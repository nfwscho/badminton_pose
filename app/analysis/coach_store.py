# 기존 스윙 JSON과 독립적인 코칭 스냅샷/검토/메모/확인 이력 저장소.
# 스키마 1 JSON을 임시 파일+replace로 저장하고 원본 영상/프레임/PTS/선수 문맥을
# 검증한다. 과거 보고서는 현재 캐시와 재검증되기 전 현재 결과로 표시하지 않는다.
# CSV/JSON 내보내기는 수치·품질·출처·검토 이력을 포함하며 결측은 빈 값으로 보존한다.
import csv
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import tempfile

from app.analysis.coach_models import (CoachMessage, CoachQuality, CoachReport, CoachReview, Observation,
                                       ENGINE_VERSION, LEVELS, REVIEW_STATES, event_key)
from app.analysis.motion_models import fingerprint, validate_event
from app.analysis.motion_store import _event


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)
    handle, temporary = tempfile.mkstemp(prefix=path.stem + "-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def report_from_dict(data, timeline):
    def quality(value):
        q = CoachQuality(**{**value, "reasons": tuple(value["reasons"])})
        if (q.state not in ("COACHING_DEFERRED", "MEASUREMENT_ONLY", "COACHING_AVAILABLE")
                or not 0 <= q.valid_frames <= q.total_frames or not math.isfinite(q.valid_ratio)
                or not 0 <= q.valid_ratio <= 1):
            raise ValueError("코칭 품질 정보가 잘못되었습니다.")
        return q
    data = dict(data)
    data["video_identity"] = tuple(data["video_identity"])
    if data["video_identity"] != tuple(timeline.identity) or data["video_id"] != fingerprint(timeline.identity):
        raise ValueError("코칭 저장 결과와 현재 영상이 다릅니다.")
    event = _event(data["event_snapshot"])
    data["event_snapshot"] = asdict(event)
    validate_event(event, timeline)
    if data["event_id"] != event.event_id or data["event_fingerprint"] != event_key(event):
        raise ValueError("코칭 이벤트 스냅샷 지문이 잘못되었습니다.")
    data["quality"] = quality(data["quality"])
    data["features"] = {key: Observation(**value) for key, value in data["features"].items()}
    messages = []
    for raw in data["messages"]:
        raw = dict(raw)
        raw["quality"] = quality(raw["quality"])
        raw["measurements"] = tuple(Observation(**value) for value in raw["measurements"])
        for name in ("phases", "deferred_reasons", "graph_metrics"):
            raw[name] = tuple(raw[name])
        message = CoachMessage(**raw)
        if (message.event_id != event.event_id or message.video_id != event.video_id
                or message.athlete_key != event.athlete_key or message.player_id != event.player_id
                or message.track_segment != event.track_segment or message.level not in LEVELS
                or not event.start_frame <= message.start_frame <= message.end_frame <= event.end_frame
                or message.start_pts != timeline.raw_pts_seconds[message.start_frame]
                or message.end_pts != timeline.raw_pts_seconds[message.end_frame]):
            raise ValueError("코칭 근거의 선수/프레임/PTS가 잘못되었습니다.")
        messages.append(message)
    data["messages"] = tuple(messages)
    report = CoachReport(**data)
    for observation in (*report.features.values(), *(v for m in report.messages for v in m.measurements)):
        if (observation.level not in LEVELS or not event.start_frame <= observation.start_frame <= observation.end_frame <= event.end_frame
                or (observation.peak_frame is not None and not observation.start_frame <= observation.peak_frame <= observation.end_frame)
                or (observation.valid and (observation.value is None or not math.isfinite(observation.value)))
                or (not observation.valid and observation.value is not None)):
            raise ValueError("코칭 관측값/근거 범위가 잘못되었습니다.")
    return report


class CoachStore:
    def __init__(self, directory):
        self.directory = Path(directory)

    def path_for(self, identity):
        return self.directory / (fingerprint(identity) + ".json")

    def load(self, timeline):
        path = self.path_for(timeline.identity)
        if not path.exists():
            return {}, {}, {}, []
        def reject(value):
            raise ValueError("코칭 JSON 비정상 숫자: " + value)
        data = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)
        if data.get("schema_version") != 1 or tuple(data["video_identity"]) != tuple(timeline.identity):
            raise ValueError("코칭 JSON 버전/영상 식별 불일치")
        reports = {key: report_from_dict(value, timeline) for key, value in data["reports"].items()}
        if any(key != report.event_id for key, report in reports.items()):
            raise ValueError("코칭 보고서 키 불일치")
        reviews = {key: CoachReview(**{**value, "history": tuple(value.get("history", ()))}) for key, value in data["reviews"].items()}
        for review in reviews.values():
            if review.status not in REVIEW_STATES or len(review.note) > 10000 or len(review.error_reason) > 2000:
                raise ValueError("잘못된 사용자 검토 정보")
        return reports, reviews, data.get("confirmations", {}), data.get("history", [])

    def save(self, timeline, reports, reviews, confirmations, history):
        data = {"schema_version": 1, "engine_version": ENGINE_VERSION, "video_identity": timeline.identity,
                "reports": {key: asdict(report) for key, report in reports.items()},
                "reviews": {key: asdict(review) for key, review in reviews.items()},
                "confirmations": confirmations, "history": history}
        for report in reports.values():
            report_from_dict(asdict(report), timeline)
        path = self.path_for(timeline.identity)
        atomic_json(path, data)
        return path


def export_reports(path, reports, reviews, stale_ids=()):
    path = Path(path)
    reports = tuple(reports)
    payload = {"schema_version": 1, "engine_version": ENGINE_VERSION,
               "reports": [{**asdict(report), "stale": report.event_id in stale_ids} for report in reports],
               "reviews": {key: asdict(value) for key, value in reviews.items()}}
    if path.suffix.lower() != ".csv":
        atomic_json(path, payload)
        return
    # 메모/라벨을 스프레드시트 수식으로 실행하지 않도록 텍스트 접두를 적용한다.
    def cell(value):
        return "'" + value if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")) else value
    columns = ("result_id", "event_id", "video_id", "athlete_key", "player_id", "track_segment", "rule_id", "rule_version",
               "start_frame", "end_frame", "start_pts", "end_pts", "status", "level", "observation", "interpretation",
               "suggestion", "measurements_json", "quality_json", "deferred_reasons_json", "engine_version", "rules_version",
               "measurement_version", "motion_version", "input_key", "generated_at", "stale", "review_json")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.stem + "-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for report in reports:
                for message in report.messages:
                    row = {name: getattr(message, name) for name in columns if hasattr(message, name)}
                    row.update({name: getattr(report, name) for name in ("engine_version", "rules_version", "measurement_version", "motion_version", "input_key")})
                    row.update(measurements_json=json.dumps([asdict(o) for o in message.measurements], ensure_ascii=False, allow_nan=False),
                               quality_json=json.dumps(asdict(message.quality), ensure_ascii=False),
                               deferred_reasons_json=json.dumps(message.deferred_reasons, ensure_ascii=False),
                               stale=report.event_id in stale_ids,
                               review_json=json.dumps(asdict(reviews.get(message.event_id + ":" + message.rule_id, CoachReview())), ensure_ascii=False))
                    writer.writerow({key: cell(value) for key, value in row.items()})
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
