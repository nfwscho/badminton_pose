# 기존 관절/각도 캐시에서 동작 특징·후보·단계를 계산하는 취소 가능한 QThread.
# MediaPipe나 영상 디코더를 호출하지 않는다. 입력 스냅샷과 결과를 전달하고
# UI는 문맥/세대 검사를 통해 오래된 작업 결과를 폐기한다. 진행/제외/분류를
# 요약 로그로 남기며 동영상 재생 프레임마다 로그를 출력하지 않는다.
from collections import Counter
import logging
import time
from PySide6.QtCore import QThread, Signal
from app.analysis.motion_features import extract_features
from app.analysis.motion_detector import detect_motion
from app.analysis.motion_models import ALGORITHM_VERSION

log = logging.getLogger(__name__)


class MotionWorker(QThread):
    ready = Signal(object, object)
    progress = Signal(int)
    error = Signal(str)

    def __init__(self, frames, selections, measurements, timeline, width, height, hand, config, context, scope, parent=None):
        super().__init__(parent)
        self.frames, self.selections, self.measurements = dict(frames), dict(selections), dict(measurements)
        self.timeline, self.width, self.height = timeline, width, height
        self.hand, self.config, self.context, self.scope = hand, config, dict(context), scope

    def run(self):
        start = time.monotonic()
        try:
            log.info("스윙 분석 시작: player=%s, session=%s, hand=%s, scope=%s, frames=%s, algorithm=%s",
                     self.context["athlete_key"], self.context["player_id"], self.hand, self.scope,
                     self.scope[1] - self.scope[0] + 1, ALGORITHM_VERSION)
            self.progress.emit(0)
            features = extract_features(self.frames, self.selections, self.measurements, self.timeline,
                                        self.width, self.height, self.hand, self.config, self.isInterruptionRequested)
            if features is None or self.isInterruptionRequested():
                return
            self.progress.emit(55)
            excluded = Counter(reason for i, reason in enumerate(features.reasons)
                               if self.scope[0] <= i <= self.scope[1] and reason)
            log.info("스윙 분석 추적/관측 제외 요약: %s", dict(excluded))
            result = detect_motion(features, self.measurements, self.timeline, self.context, self.hand,
                                   self.config, self.scope, self.isInterruptionRequested)
            if result is None or self.isInterruptionRequested():
                return
            counts = Counter(event.kind for event in result.events)
            log.info("스윙 분석 종료: candidates=%s, overhead=%s, other=%s, 보류=%s, %.3f초",
                     len(result.events), counts["OVERHEAD_CANDIDATE"], counts["OTHER_SWING"], counts["UNKNOWN"], time.monotonic() - start)
            log.info("스윙 단계 경계 후보: events=%s, 단계 전체 보류=%s",
                     sum(any(p.phase != "UNKNOWN_PHASE" for p in e.phases) for e in result.events),
                     sum(all(p.phase == "UNKNOWN_PHASE" for p in e.phases) for e in result.events))
            self.progress.emit(100)
            self.ready.emit(result, features)
        except Exception as exc:
            log.exception("스윙 분석 오류")
            self.error.emit(str(exc))
        finally:
            if self.isInterruptionRequested():
                log.info("스윙 분석 취소: %.3f초", time.monotonic() - start)
