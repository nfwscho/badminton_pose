# 영상 식별 정보와 선수/라켓 손 문맥에 연결한 스윙 결과 JSON 저장소.
# 원본 자동 결과와 수동 이벤트/삭제/수정 이력을 분리하고 변경 전 문서를 history에
# 보존한다. 임시 파일+replace로 저장하며 소스 영상/PTS/경계가 다른 파일은 거부한다.
# 저장 결과는 과거 관측 스냅샷이며 현재 선수 캐시에 자동으로 합치지 않는다.
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile

from app.analysis.motion_models import (ALGORITHM_VERSION, KINDS, MotionConfig, MotionDocument, MotionEvent,
                                       MotionInterval, MotionResult, PhaseSpan, fingerprint, validate_event)


def _event(data):
    data = dict(data)
    data["phases"] = tuple(PhaseSpan(**phase) for phase in data["phases"])
    data["evidence"], data["uncertainty"] = tuple(data["evidence"]), tuple(data["uncertainty"])
    return MotionEvent(**data)


def document_from_dict(data, timeline):
    automatic = dict(data["automatic"])
    automatic["video_identity"] = tuple(automatic["video_identity"])
    if automatic["video_identity"] != tuple(timeline.identity):
        raise ValueError("저장된 동작 결과와 현재 영상 식별 정보가 다릅니다.")
    if automatic["algorithm_version"] != ALGORITHM_VERSION:
        raise ValueError("저장된 동작 알고리즘 버전이 다릅니다. 원본을 보존하고 재검토하세요.")
    automatic["scope"] = tuple(automatic["scope"])
    if not 0 <= automatic["scope"][0] <= automatic["scope"][1] < len(timeline.seconds):
        raise ValueError("저장된 분석 범위가 올바르지 않습니다.")
    automatic["config"] = MotionConfig(**automatic["config"])
    automatic["events"] = tuple(_event(event) for event in automatic["events"])
    automatic["coverage"] = tuple(MotionInterval(**interval) for interval in automatic["coverage"])
    result = MotionResult(**automatic)
    current = tuple(_event(event) for event in data["events"])
    if result.video_id != fingerprint(timeline.identity):
        raise ValueError("동작 결과의 영상 ID가 다릅니다.")
    if (result.athlete_key != fingerprint((result.video_id, result.source_anchor))
            or result.context_id != fingerprint((result.athlete_key, result.racket_hand, ALGORITHM_VERSION))):
        raise ValueError("저장된 선수 문맥 키가 일치하지 않습니다.")
    for events in (result.events, current):
        if len({event.event_id for event in events}) != len(events):
            raise ValueError("중복 스윙 이벤트 ID가 있습니다.")
        for event in events:
            if (event.video_id != result.video_id or event.athlete_key != result.athlete_key
                    or event.racket_hand != result.racket_hand or event.player_id != result.player_id):
                raise ValueError("서로 다른 선수/라켓 손의 이벤트가 섞여 있습니다.")
            validate_event(event, timeline, events)
    for interval in result.coverage:
        if (interval.kind not in KINDS or not result.scope[0] <= interval.start_frame <= interval.end_frame <= result.scope[1]):
            raise ValueError("관측 구간 범위가 올바르지 않습니다.")
    if (not result.coverage or result.coverage[0].start_frame != result.scope[0] or result.coverage[-1].end_frame != result.scope[1]
            or any(a.end_frame + 1 != b.start_frame for a, b in zip(result.coverage, result.coverage[1:]))):
        raise ValueError("관측/보류 구간은 분석 범위를 빠짐없이 분할해야 합니다.")
    return MotionDocument(result, current, tuple(data.get("operations", ())))


class MotionStore:
    def __init__(self, directory):
        self.directory = Path(directory)

    def path_for(self, identity):
        return self.directory / (fingerprint(identity) + ".json")

    def _read(self, identity):
        path = self.path_for(identity)
        if not path.exists():
            return {"schema_version": 1, "video_identity": list(identity), "documents": {}, "history": [], "last_context": None}
        def reject_constant(value):
            raise ValueError("JSON에 비정상 숫자가 있습니다: " + value)
        data = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
        if data.get("schema_version") != 1 or tuple(data["video_identity"]) != tuple(identity):
            raise ValueError("동작 JSON의 버전/영상 정보가 다릅니다.")
        return data

    def load(self, timeline, context_id=None):
        data = self._read(timeline.identity)
        key = context_id or data.get("last_context")
        document = data["documents"].get(key) if key else None
        return document_from_dict(document, timeline) if document else None

    def available(self, timeline):
        data = self._read(timeline.identity)
        return [document_from_dict(document, timeline) for document in data["documents"].values()]

    def save(self, document, timeline):
        # 저장 전에도 로드와 동일한 유효성 검사를 적용한다.
        payload = asdict(document)
        document_from_dict(payload, timeline)
        data = self._read(timeline.identity)
        key = document.automatic.context_id
        previous = data["documents"].get(key)
        if previous is not None and previous != payload:
            # tuple/list 표현은 JSON 정규화 후 비교한다.
            if json.dumps(previous, sort_keys=True, ensure_ascii=False) != json.dumps(payload, sort_keys=True, ensure_ascii=False):
                data["history"].append({"saved_at": datetime.now(timezone.utc).isoformat(), "document": previous})
        data["documents"][key], data["last_context"] = payload, key
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.directory, delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            os.replace(temporary, self.path_for(timeline.identity))
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        return self.path_for(timeline.identity)
