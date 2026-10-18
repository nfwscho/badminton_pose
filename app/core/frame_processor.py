# 별도 QThread에서 원본 영상을 순서대로 읽어 VIDEO 관절 인식을 수행한다.
# 원본 PTS 시간과 MediaPipe용 단조 증가 보정 시간을 분리하여 배치로 전달한다.
# 캐시는 영상 파일 정보, 모델 해시 및 검출 설정으로 식별하며 취소/오류 시
# 이미 분석한 부분만 유지한다. 재생 디코더나 GUI의 캡처를 공유하지 않는다.
# 추적된 캐시의 관절 수치 계산 및 CSV 저장도 별도 작업 스레드에 위임한다.
from dataclasses import dataclass, field, replace
import hashlib
import logging
import math
from pathlib import Path
import time

import cv2
from PySide6.QtCore import QThread, Signal
from app.core.pose_detector import CONFIDENCE, DETECTION_VIEWS, MODEL_PATH, NUM_POSES, PoseDetector
from app.core.video_timeline import read_timeline, video_identity

log = logging.getLogger(__name__)


def cache_key(video_path, model_path=MODEL_PATH):
    from importlib.metadata import version
    video, model = Path(video_path), Path(model_path)
    stat = video.stat()
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    return (str(video.resolve()), stat.st_size, stat.st_mtime_ns, digest,
            version("mediapipe"), NUM_POSES, CONFIDENCE, DETECTION_VIEWS, "pts-timestamp-v2")


@dataclass
class AnalysisCache:
    video_path: str
    frames: dict = field(default_factory=dict)
    key: tuple | None = None
    state: str = "empty"
    selections: dict = field(default_factory=dict)
    anchor: tuple[int, int] | None = None
    measurements: dict = field(default_factory=dict)


class AnalysisWorker(QThread):
    batch_ready = Signal(object)
    progress = Signal(int)
    completed = Signal(str, object, float)
    error = Signal(str)

    def __init__(self, video_path, model_path=MODEL_PATH, parent=None, detector_factory=PoseDetector):
        super().__init__(parent)
        self.video_path, self.model_path = str(video_path), Path(model_path)
        self.detector_factory = detector_factory
        self.timeline = None

    def run(self):
        started = time.monotonic()
        cap, detector, key = None, None, None
        batch = []
        state = "error"
        try:
            log.info("동영상 AI 분석 시작: %s", self.video_path)
            if self.isInterruptionRequested():
                state = "cancelled"
                return
            cap = cv2.VideoCapture(self.video_path)
            fps, count_value = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if not cap.isOpened() or not math.isfinite(fps) or fps <= 0 or not math.isfinite(count_value) or count_value < 1:
                raise RuntimeError("분석할 동영상의 FPS 또는 프레임 정보를 읽을 수 없습니다.")
            count = int(count_value)
            timeline = self.timeline or read_timeline(self.video_path, self.isInterruptionRequested)
            if timeline is None:
                state = "cancelled"
                return
            if timeline.identity != video_identity(self.video_path):
                raise RuntimeError("시간축 생성 후 영상이 변경되었습니다. 영상을 다시 열고 분석하세요.")
            if len(timeline.seconds) != count:
                raise RuntimeError("원본 시간축과 재생 디코더의 프레임 수가 다릅니다. 분석을 보류합니다.")
            detector = self.detector_factory(self.model_path)
            key = cache_key(self.video_path, self.model_path)
            log.info("AI 시간 기준: 원본 PTS, fps=%s, frames=%s, FPS 대체 프레임=%s", fps, count, timeline.sources.count("fps_fallback"))
            previous_timestamp, last_progress, previous_status = -1, -1, None
            status_counts = {}
            for index in range(count):
                if self.isInterruptionRequested():
                    state = "cancelled"
                    break
                ok, bgr = cap.read()
                if not ok:
                    raise RuntimeError(f"AI 분석 프레임 읽기 실패: {index}/{count}")
                actual_seconds = timeline.seconds[index]
                timestamp = max(previous_timestamp + 1, round(actual_seconds * 1000))
                previous_timestamp = timestamp
                result = detector.detect(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), index, timestamp)
                result = replace(result, timestamp_ms=round(actual_seconds * 1000), timestamp_seconds=actual_seconds,
                                 timestamp_source=timeline.sources[index], model_timestamp_ms=timestamp)
                batch.append(result)
                status_counts[result.status] = status_counts.get(result.status, 0) + 1
                # 상태 변화만 기록하고 상세 프레임 수는 마지막에 요약한다.
                if result.status != previous_status:
                    if result.status != "detected" and (index < 5 or index % 30 == 0):
                        log.info("관절 검출 상태: frame=%s, %s", index, result.status)
                    previous_status = result.status
                if len(batch) >= 16:
                    self.batch_ready.emit(tuple(batch))
                    batch.clear()
                percent = min(99, int((index + 1) * 100 / count))
                if percent != last_progress:
                    self.progress.emit(percent)
                    if percent // 10 != last_progress // 10:
                        log.info("AI 분석 진행률: %s%%", percent)
                    last_progress = percent
            else:
                if cache_key(self.video_path, self.model_path) != key:
                    raise RuntimeError("분석 중 영상 또는 모델이 변경되었습니다. 다시 분석하세요.")
                state = "complete"
            log.info("관절 검출 상태 요약: %s", status_counts)
        except Exception as exc:
            log.exception("동영상 AI 분석 오류")
            self.error.emit(str(exc))
        finally:
            if batch:
                self.batch_ready.emit(tuple(batch))
            if detector is not None:
                try:
                    detector.close()
                except Exception:
                    state = "error"
                    log.exception("AI 모델 종료 오류")
                    self.error.emit("AI 모델 자원 해제에 실패했습니다. 로그를 확인하세요.")
            if cap is not None:
                cap.release()
            elapsed = time.monotonic() - started
            if state == "complete":
                self.progress.emit(100)
            log.info("AI 분석 %s, 소요 시간 %.2f초", {"complete": "완료", "cancelled": "취소", "error": "실패"}[state], elapsed)
            self.completed.emit(state, key, elapsed)


class MeasurementWorker(QThread):
    ready = Signal(object)
    error = Signal(str)

    def __init__(self, frames, selections, timeline, width, height, parent=None):
        super().__init__(parent)
        self.frames, self.selections = dict(frames), dict(selections)
        self.timeline, self.width, self.height = timeline, width, height

    def run(self):
        try:
            from app.analysis.joint_angles import calculate_measurements
            started = time.monotonic()
            log.info("관절 수치 계산 시작: cached_frames=%s (추론 재실행 없음)", len(self.frames))
            measurements = calculate_measurements(self.frames, self.selections, self.timeline,
                                                  self.width, self.height, self.isInterruptionRequested)
            if not self.isInterruptionRequested():
                log.info("관절 수치 계산 완료: valid_frames=%s/%s, %.3f초",
                         sum(m.analyzable for m in measurements.values()), len(measurements), time.monotonic() - started)
                self.ready.emit(measurements)
        except Exception as exc:
            log.exception("관절 수치 계산 오류")
            self.error.emit(str(exc))


class CsvExportWorker(QThread):
    saved = Signal(str)
    error = Signal(str)

    def __init__(self, path, measurements, timeline, bounds, parent=None):
        super().__init__(parent)
        self.path, self.measurements, self.timeline, self.bounds = path, dict(measurements), timeline, bounds

    def run(self):
        try:
            from app.analysis.joint_angles import export_csv
            export_csv(self.path, self.measurements, self.timeline, self.bounds)
            log.info("관절 측정 CSV 저장 완료: %s, range=%s", self.path, self.bounds)
            self.saved.emit(str(self.path))
        except Exception as exc:
            log.exception("관절 CSV 저장 오류")
            self.error.emit(str(exc))
