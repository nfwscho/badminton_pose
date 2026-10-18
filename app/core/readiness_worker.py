# 분석 준비 진단/저장 선수 연결 후보를 백그라운드에서 검증한다.
# 실제 영상·모델 캐시 키와 원본 PTS/선수 추적을 검사하고 기존 코칭 규칙으로
# 스윙별 품질/적용 가능성을 미리 확인한다. MediaPipe 추론은 수행하지 않는다.
# 연결 문서는 승인 전의 제안일 뿐이며 UI가 입력 세대와 명시적 사용자 승인을 확인한다.
import logging
from PySide6.QtCore import QThread, Signal
from app.analysis.analysis_readiness import rebind_document, validate_link
from app.analysis.coach_engine import analyze_coaching, source_key
from app.analysis.motion_features import extract_features
from app.analysis.motion_models import fingerprint
from app.core.frame_processor import cache_key
from app.core.video_timeline import video_identity

log = logging.getLogger(__name__)


class ReadinessWorker(QThread):
    ready = Signal(object)
    error = Signal(str)

    def __init__(self, video_path, cache, timeline, document, context, config, rules, confirmations, link=False, parent=None):
        super().__init__(parent)
        self.video_path, self.timeline, self.document = video_path, timeline, document
        self.frames, self.selections, self.measurements = dict(cache.frames), dict(cache.selections), dict(cache.measurements)
        self.cache_key, self.run_id = cache.key, cache.run_id
        self.context, self.config, self.rules, self.confirmations, self.link = context, config, rules, dict(confirmations), link

    def run(self):
        try:
            result = {"asset_valid": False, "asset_reason": "", "asset_action": "pose", "reports": (), "link_reasons": (), "proposal": None, "features": None}
            log.info("코칭 실행 준비 검사 시작: link=%s run=%s", self.link, self.run_id)
            try:
                if video_identity(self.video_path) != self.timeline.identity:
                    result["asset_action"] = "open"
                    raise ValueError("원본 영상 파일이 변경되었습니다. MP4를 다시 불러오고 분석하세요.")
                if cache_key(self.video_path) != self.cache_key:
                    raise ValueError("영상 또는 모델·관절 분석 설정이 현재 캐시와 다릅니다. 관절 분석을 다시 실행하세요.")
                result["asset_valid"] = True
            except Exception as exc:
                result["asset_reason"] = str(exc)
            if self.isInterruptionRequested():
                return
            if not result["asset_valid"]:
                self.ready.emit(result)
                return
            if self.document and self.context:
                reasons, mapping = validate_link(self.document, self.frames, self.selections, self.measurements, self.timeline, self.context)
                result["link_reasons"] = reasons
                if self.link and not reasons:
                    if self.config != self.document.automatic.config:
                        result["link_reasons"] = ("스윙 검출 설정이 바뀌었습니다. 변경 내용을 확인하고 영향받는 스윙을 갱신하세요.",)
                    else:
                        features = extract_features(self.frames, self.selections, self.measurements, self.timeline,
                                                    self.timeline.width, self.timeline.height, self.context["racket_hand"] if "racket_hand" in self.context else self.document.automatic.racket_hand,
                                                    self.config, self.isInterruptionRequested)
                        if features is not None:
                            provenance = {"pose_run_id": self.run_id, "pose_cache_key": fingerprint(self.cache_key),
                                          "measurement_key": source_key(self.frames, self.selections, self.measurements, self.timeline, self.config, self.isInterruptionRequested)}
                            result["proposal"] = rebind_document(self.document, features, self.measurements, self.timeline,
                                                                  self.context, mapping, provenance)
                            result["features"] = features
                document = result["proposal"] or self.document
                # 다른 세션의 과거 문서는 현재 데이터의 품질로 가장하지 않는다.
                if (self.rules and (result["proposal"] or not reasons) and document.automatic.player_id == self.context["player_id"]
                        and self.config == document.automatic.config):
                    result["reports"] = analyze_coaching(document, self.frames, self.selections, self.measurements,
                                                         self.timeline, self.rules, self.confirmations, cancelled=self.isInterruptionRequested) or ()
            if not self.isInterruptionRequested():
                self.ready.emit(result)
        except Exception as exc:
            log.exception("분석 준비 진단 오류")
            self.error.emit(str(exc))
