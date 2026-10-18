# 원본 동영상의 디코딩 표시 순서와 PTS를 읽는 독립 시간축 모듈.
# PyAV Frame.pts*time_base를 첫 표시 프레임 기준 초로 정규화하고 원본 PTS도
# 보존한다. 누락/역행 PTS에만 명시적 FPS 대체를 적용하며 출처와 횟수를 남긴다.
# 별도 QThread에서 인덱싱하여 기존 OpenCV 재생과 GUI를 막지 않는다.
from bisect import bisect_left
from dataclasses import dataclass
import logging
import math
from pathlib import Path

from PySide6.QtCore import QThread, Signal

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class VideoTimeline:
    seconds: tuple[float, ...]
    raw_pts_seconds: tuple[float | None, ...]
    sources: tuple[str, ...]
    duration: float
    width: int
    height: int
    identity: tuple
    origin_seconds: float = 0.0

    def nearest_frame(self, seconds):
        i = bisect_left(self.seconds, seconds)
        if i >= len(self.seconds):
            return len(self.seconds) - 1
        if i and abs(self.seconds[i - 1] - seconds) <= abs(self.seconds[i] - seconds):
            return i - 1
        return i

    def frame_end(self, index):
        return self.seconds[index + 1] if index + 1 < len(self.seconds) else self.duration


def video_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return (str(path), stat.st_size, stat.st_mtime_ns)


def read_timeline(path, cancelled=lambda: False):
    import av
    identity = video_identity(path)
    times, raw, sources = [], [], []
    origin, final_duration = None, None
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or stream.guessed_rate or 30)
        if not math.isfinite(fps) or fps <= 0:
            fps = 30.0
        for index, frame in enumerate(container.decode(stream)):
            if cancelled():
                return None
            stamp = float(frame.pts * frame.time_base) if frame.pts is not None and frame.time_base is not None else None
            if stamp is not None and not math.isfinite(stamp):
                stamp = None
            if origin is None and stamp is not None:
                origin = stamp - index / fps
            seconds = stamp - origin if stamp is not None and origin is not None else None
            if seconds is None or seconds < 0 or (times and seconds < times[-1]):
                seconds = times[-1] + 1 / fps if times else 0.0
                source = "fps_fallback"
            else:
                source = "pts"
            times.append(seconds)
            raw.append(stamp)
            sources.append(source)
            duration = getattr(frame, "duration", None)
            final_duration = float(duration * frame.time_base) if duration and frame.time_base else None
        if not times:
            raise RuntimeError("영상에서 표시 프레임의 PTS를 읽을 수 없습니다.")
        if final_duration is None or final_duration <= 0:
            final_duration = times[-1] - times[-2] if len(times) > 1 and times[-1] > times[-2] else 1 / fps
        width, height = stream.codec_context.width, stream.codec_context.height
    if video_identity(path) != identity:
        raise RuntimeError("시간축을 읽는 중 영상 파일이 변경되었습니다.")
    fallback = sources.count("fps_fallback")
    log.info("영상 PTS 시간축 완료: %s, frames=%s, duration=%.6f, FPS 대체=%s", path, len(times), times[-1] + final_duration, fallback)
    return VideoTimeline(tuple(times), tuple(raw), tuple(sources), times[-1] + final_duration,
                         width, height, identity, origin or 0.0)


class TimelineWorker(QThread):
    ready = Signal(object)
    error = Signal(str)

    def __init__(self, path, parent=None):
        super().__init__(parent)
        self.path = path

    def run(self):
        try:
            timeline = read_timeline(self.path, self.isInterruptionRequested)
            if timeline is not None and not self.isInterruptionRequested():
                self.ready.emit(timeline)
        except Exception as exc:
            log.exception("영상 PTS 인덱싱 오류")
            self.error.emit(str(exc))
