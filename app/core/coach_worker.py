# 기존 캐시 스냅샷을 독립 QThread의 순수 코칭 엔진으로 전달한다.
# MediaPipe/동영상 추론 없이 진행/취소/오류/처리 시간과 규칙 버전을 로그에 남긴다.
# GUI는 작업 세대와 현재 문맥을 재검사한 뒤 결과를 수락한다.
import logging
import time
from PySide6.QtCore import QThread, Signal
from app.analysis.coach_engine import analyze_coaching
from app.analysis.coach_models import ENGINE_VERSION

log = logging.getLogger(__name__)


class CoachWorker(QThread):
    ready = Signal(object)
    progress = Signal(int)
    error = Signal(str)

    def __init__(self, document, frames, selections, measurements, timeline, rules, confirmations,
                 event_ids=None, comparison_confirmed=False, comparison_group="", cache=None, parent=None):
        super().__init__(parent)
        self.args = (document, dict(frames), dict(selections), dict(measurements), timeline, rules,
                     dict(confirmations), event_ids, comparison_confirmed, comparison_group)
        self.cache = dict(cache or {})

    def run(self):
        started = time.monotonic()
        try:
            log.info("코칭 엔진 시작: engine=%s rules=%s/%s events=%s", ENGINE_VERSION,
                     self.args[5].version, self.args[5].key, self.args[7] or "all")
            reports = analyze_coaching(*self.args, cancelled=self.isInterruptionRequested, progress=self.progress.emit, cache=self.cache)
            if reports is not None and not self.isInterruptionRequested():
                self.ready.emit(reports)
                log.info("코칭 엔진 종료: reports=%s messages=%s %.3f초", len(reports),
                         sum(len(r.messages) for r in reports), time.monotonic() - started)
        except Exception as exc:
            log.exception("코칭 분석 오류")
            self.error.emit(str(exc))
        finally:
            if self.isInterruptionRequested():
                log.info("코칭 분석 취소: %.3f초", time.monotonic() - started)
