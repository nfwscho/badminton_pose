# Qt/추론과 독립적인 근거 기반 로컬 관찰 엔진.
# 실제 캐시에서 단계별 관절 변화/피크/경로/안정화 특징을 계산하고 품질 검사 후
# 허용된 JSON 규칙을 적용한다. 불확실한 단계는 후보로 보존하고 결측을 채우지 않는다.
# 같은 선수·손·동작·고정 촬영/기술 확인을 통과한 경우만 반복 차이를 제공한다.
# 모든 수치는 실제 근거 프레임에 연결하며 기술 점수/표준 각도/타격 판정은 생성하지 않는다.
from dataclasses import asdict, replace
import logging
import math
import uuid
import numpy as np

from app.analysis.coach_models import (CoachMessage, CoachReport, ENGINE_VERSION, MEASUREMENT_VERSION,
                                       Observation, event_key, feature_label, now)
from app.analysis.coach_quality import assess_quality
from app.analysis.motion_features import extract_features, smooth_signal, temporal_derivative
from app.analysis.motion_models import fingerprint, validate_event

log = logging.getLogger(__name__)


def source_key(frames, selections, measurements, timeline, motion_config, cancelled=lambda: False):
    # 표본의 반올림 없이 입력 내용을 지문에 포함한다. 캐시 자체는 수정하지 않는다.
    import hashlib
    import json
    digest = hashlib.sha256()
    def add(value):
        # 무효 원시 NaN도 지문에만 기록한다. 출력 관측값에는 유한 값만 허용한다.
        digest.update(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=True).encode("utf-8"))
    add((timeline.identity, timeline.seconds, timeline.raw_pts_seconds, timeline.sources, asdict(motion_config)))
    for i in range(len(timeline.seconds)):
        if cancelled():
            return None
        add((i, asdict(frames[i]) if i in frames else None, asdict(selections[i]) if i in selections else None,
             asdict(measurements[i]) if i in measurements else None))
    return digest.hexdigest()


def _raw(measurements, name, n, valid):
    values = np.full(n, np.nan)
    for i, m in measurements.items():
        metric = m.values.get(name)
        if 0 <= i < n and valid[i] and metric and metric.valid and metric.raw is not None and math.isfinite(metric.raw):
            values[i] = metric.raw
    return values


def summarize_features(event, features, measurements, timeline, config, integrity=True):
    result, statuses = {}, {}
    n, hand = len(features.times), event.racket_hand
    valid = features.valid.copy() if integrity else np.zeros(n, dtype=bool)
    series = {name: _raw(measurements, metric, n, valid) for name, metric in (
        ("elbow", hand + "_elbow"), ("shoulder", hand + "_shoulder"), ("trunk", "trunk_tilt"),
        ("left_knee", "left_knee"), ("right_knee", "right_knee"), ("wrist_height", hand + "_wrist_height"))}
    # 코칭 미분/평활화는 현재 이벤트 안의 관측만 사용한다. 단계 경계의 미분에는
    # 같은 이벤트의 이웃 단계가 필요할 수 있으므로 메시지 근거 범위를 아래에서 확장한다.
    ea, eb = event.start_frame, event.end_frame
    scope = slice(ea, eb + 1)
    times, labels = features.times[scope], features.runs[scope]
    wrist_local = np.where(valid[scope, None], features.wrist_raw[scope], np.nan)
    wrist_local = smooth_signal(wrist_local, times, labels, config)
    wrist_positions = np.full((n, 2), np.nan)
    wrist_positions[scope] = wrist_local
    derivatives = {"wrist_speed": np.linalg.norm(temporal_derivative(wrist_local, times, labels, config.max_time_gap), axis=1)}
    for name in ("elbow", "shoulder", "trunk"):
        smoothed = smooth_signal(series[name][scope], times, labels, config)
        derivatives[name + "_rate"] = temporal_derivative(smoothed, times, labels, config.max_time_gap)
    for name, values in derivatives.items():
        series[name] = np.full(n, np.nan)
        series[name][scope] = values
    spans = [("SWING", event.start_frame, event.end_frame)]
    spans += [(p.phase, p.start_frame, p.end_frame) for p in event.phases if p.phase != "UNKNOWN_PHASE"]
    for phase, a, b in spans:
        statuses[phase] = "USER_REVIEW_REQUIRED_CANDIDATE" if phase != "SWING" else "EVENT_RANGE"
        def put(name, value, unit, level="DERIVED", peak=None, reason=""):
            good = value is not None and math.isfinite(float(value))
            result[phase + "." + name] = Observation(phase + "." + name, float(value) if good else None,
                                                       unit, good, reason if not good else "", level, a, b, peak)
        connected = (integrity and bool(np.all(valid[a:b + 1]))
                     and len(set(features.runs[a:b + 1])) == 1 and features.runs[a] >= 0)
        put("duration", timeline.frame_end(b) - timeline.seconds[a], "초")
        put("valid_ratio", float(np.mean(valid[a:b + 1])), "유효 관측 비율")
        for name, values in series.items():
            unit = "몸통길이/초" if name == "wrist_speed" else "도/초" if name.endswith("rate") else "몸통길이" if name == "wrist_height" else "도"
            chunk = values[a:b + 1]
            finite = np.flatnonzero(np.isfinite(chunk))
            enough = len(finite) >= 2
            complete = connected and len(finite) == len(chunk) and enough
            put(name + "_start", chunk[0] if np.isfinite(chunk[0]) else None, unit, "MEASURED" if name in ("elbow", "shoulder", "trunk", "left_knee", "right_knee", "wrist_height") else "DERIVED", a,
                "시작 프레임 관측 없음")
            put(name + "_change", chunk[-1] - chunk[0] if complete else None, unit, reason="연속적인 시작·끝 관측 부족")
            put(name + "_range", float(np.ptp(chunk[finite])) if enough else None, unit, reason="유효 관측 2개 미만")
            for extreme, choose in (("min", np.argmin), ("max", np.argmax)):
                frame = a + int(finite[choose(chunk[finite])]) if enough else None
                put(name + "_" + extreme, values[frame] if frame is not None else None, unit,
                    peak=frame, reason="유효 관측 2개 미만")
        wrist = wrist_positions[a:b + 1]
        path = float(np.linalg.norm(np.diff(wrist, axis=0), axis=1).sum()) if connected and len(wrist) >= 2 and np.isfinite(wrist).all() else None
        put("wrist_path", path, "몸통길이", reason="손목 경로 연속 관측 부족")
        for name in ("wrist_speed", "elbow_rate", "shoulder_rate"):
            chunk = series[name][a:b + 1]
            finite = np.flatnonzero(np.isfinite(chunk))
            frame = a + int(finite[np.argmax(np.abs(chunk[finite]))]) if connected and len(finite) >= 2 else None
            alias = "wrist" if name == "wrist_speed" else name.split("_")[0]
            put(alias + "_peak_time", timeline.seconds[frame] if frame is not None else None, "초", peak=frame,
                reason="피크 계산을 위한 연속 관측 부족")
            put(name + "_peak", abs(series[name][frame]) if frame is not None else None,
                "몸통길이/초" if name == "wrist_speed" else "도/초", peak=frame, reason="피크 미관측")
        wrist_peak, elbow_peak = result[phase + ".wrist_peak_time"], result[phase + ".elbow_peak_time"]
        put("peak_time_difference", wrist_peak.value - elbow_peak.value if wrist_peak.valid and elbow_peak.valid else None,
            "초", reason="동일 단계의 두 피크 미관측")
    # 실제로 관측된 복귀 후보의 안정화만 요약한다. 다음 준비/완전 복귀 시점은 만들지 않는다.
    if "RECOVERY" in statuses:
        span = next(p for p in event.phases if p.phase == "RECOVERY")
        result["RECOVERY.next_ready_time"] = Observation("RECOVERY.next_ready_time", None, "초", False,
            "다음 준비 자세의 확정 관측 없음; 복귀 후보 지속 시간과 다름", "UNDETERMINED", span.start_frame, span.end_frame)
    return result, statuses


def compare_reports_data(event, observations, quality, peers, comparison_confirmed, group):
    result = {"reference_event_id": None, "group": group, "differences": {}, "reasons": []}
    if not comparison_confirmed or not group.strip():
        result["reasons"].append("동일 촬영 각도·고정 카메라·동일 기술 그룹의 사용자 확인 필요")
        return result
    if not quality.user_confirmed or quality.state != "MEASUREMENT_ONLY":
        result["reasons"].append("현재 스윙/단계 또는 데이터 품질 미확인")
        return result
    if event.kind not in ("OVERHEAD_CANDIDATE", "OTHER_SWING"):
        result["reasons"].append("동작 종류 불명")
        return result
    reference = None
    for other, values, q in peers:
        if other.event_id == event.event_id:
            continue
        compatible = all(getattr(event, key) == getattr(other, key) for key in
                         ("video_id", "athlete_key", "player_id", "track_segment", "racket_hand", "kind"))
        if compatible and q.state == "MEASUREMENT_ONLY" and q.user_confirmed:
            reference = other, values
            break
    if reference is None:
        result["reasons"].append("같은 영상·선수·추적 구간·손·종류의 확인된 비교 스윙 부족")
        return result
    other, values = reference
    result["reference_event_id"] = other.event_id
    for key in ("SWING.duration", "FORWARD_SWING.duration", "SWING.elbow_range", "SWING.shoulder_range",
                "SWING.wrist_peak_time", "SWING.trunk_change", "SWING.valid_ratio"):
        current, prior = observations.get(key), values.get(key)
        if not current or not prior or not current.valid or not prior.valid:
            result["differences"][key] = {"valid": False, "reason": "양쪽 이벤트의 해당 관측 부족"}
            continue
        x, y = current.value, prior.value
        if key == "SWING.wrist_peak_time":
            x, y = x - event.start_seconds, y - other.start_seconds
        result["differences"][key] = {"valid": True, "current": x, "reference": y,
                                      "difference": x - y, "unit": current.unit}
    return result


def analyze_coaching(document, frames, selections, measurements, timeline, rules, confirmations=None,
                     event_ids=None, comparison_confirmed=False, comparison_group="", cancelled=lambda: False,
                     progress=lambda value: None, cache=None):
    confirmations = confirmations or {}
    source = document.automatic
    if tuple(timeline.identity) != source.video_identity:
        raise ValueError("코칭 입력 영상과 스윙 문서가 다릅니다.")
    if any(e.video_id != source.video_id or e.athlete_key != source.athlete_key or e.player_id != source.player_id
           or e.racket_hand != source.racket_hand for e in document.events):
        raise ValueError("코칭 문서에 다른 선수/라켓 손 이벤트가 섞였습니다.")
    for event in document.events:
        validate_event(event, timeline, document.events)
    data_key = source_key(frames, selections, measurements, timeline, source.config, cancelled)
    if data_key is None:
        return None
    document_data = asdict(document)
    targets = [e for e in document.events if event_ids is None or e.event_id in event_ids]
    input_keys = {e.event_id: fingerprint((ENGINE_VERSION, MEASUREMENT_VERSION, data_key, document_data, rules.key,
                                          confirmations, comparison_confirmed, comparison_group, e.event_id)) for e in targets}
    cache = cache or {}
    if targets and all(e.event_id in cache and cache[e.event_id].input_key == input_keys[e.event_id] for e in targets):
        log.info("코칭 캐시 재사용: events=%s rules=%s", len(targets), rules.version)
        progress(100)
        return tuple(cache[e.event_id] for e in targets)
    log.info("코칭 캐시 재계산: events=%s", len(targets))
    features = extract_features(frames, selections, measurements, timeline, timeline.width, timeline.height,
                                source.racket_hand, source.config, cancelled)
    if features is None or cancelled():
        return None
    peers = []
    summaries = {}
    for event in document.events:
        if cancelled():
            return None
        quality = assess_quality(event, features, measurements, selections, timeline,
                                 confirmations.get(event.event_id) == event_key(event), 0.8, source.config.max_time_gap, frames)
        values, statuses = summarize_features(event, features, measurements, timeline, source.config, quality.integrity_ok)
        summaries[event.event_id] = quality, values, statuses
        peers.append((event, values, quality))
    reports, created = [], now()
    for index, event in enumerate(targets):
        if cancelled():
            return None
        quality, values, statuses = summaries[event.event_id]
        comparison = compare_reports_data(event, values, quality, peers, comparison_confirmed, comparison_group)
        key = input_keys[event.event_id]
        messages = []
        log.info("코칭 품질: event=%s player=%s hand=%s state=%s valid=%s/%s phase=%s reasons=%s",
                 event.event_id, event.player_id, event.racket_hand, quality.state, quality.valid_frames,
                 quality.total_frames, quality.phase, quality.reasons)
        for rule in rules.rules:
            if not rule["enabled"]:
                log.info("코칭 규칙 제외: event=%s rule=%s disabled", event.event_id, rule["id"])
                continue
            reasons = list(quality.reasons)
            base_reasons = len(reasons)
            observations = tuple(values[name] for name in rule["features"] if name in values)
            missing = [name for name in rule["features"] if name not in values or not values[name].valid]
            if rule["kinds"] and event.kind not in rule["kinds"]:
                reasons.append("규칙의 동작 종류 적용 조건 불충족")
            if any(not any(p.phase == phase for p in event.phases) for phase in rule["phases"]):
                reasons.append("규칙에 필요한 단계 미관측")
            if missing:
                reasons.append("필요 특징 부족: " + ", ".join(feature_label(name) for name in missing))
            if quality.valid_ratio < rule["min_valid_ratio"]:
                reasons.append("규칙의 최소 관측 비율 미달")
            if rule["id"] == "repeat_consistency":
                reasons.extend(comparison["reasons"])
            rule_quality = replace(quality, applicability="INSUFFICIENT_RULE_EVIDENCE" if len(reasons) > base_reasons
                                   else "OBSERVATION_APPLICABLE")
            if rule["id"] == "quality_notice":
                observation = f"필요 팔/몸통 유효 관측 {quality.valid_frames}/{quality.total_frames} 프레임; 단계 {quality.phase}."
            else:
                available = [f"{feature_label(o.name)}: {o.value:.4g} {o.unit}" for o in observations if o.valid]
                observation = "; ".join(available) if available else "적용 가능한 실제 측정 근거가 없어 관찰을 보류합니다."
                if rule["id"] == "repeat_consistency" and not comparison["reasons"]:
                    observation += "; 기준 이벤트 " + str(comparison["reference_event_id"]) + " 대비 차이는 반복 비교 표 참조."
            deferred = bool(reasons)
            level = "UNDETERMINED" if not any(o.valid for o in observations) else "DERIVED"
            if not deferred:
                level = "INFERRED" if rule["id"] != "quality_notice" else "MEASURED"
            bounds = [(p.start_frame, p.end_frame) for p in event.phases if p.phase in rule["phases"]]
            a, b = (min(x[0] for x in bounds), max(x[1] for x in bounds)) if bounds else (event.start_frame, event.end_frame)
            padding = (source.config.window // 2 if source.config.smoothing == "median" else 0) + 1
            a, b = max(event.start_frame, a - padding), min(event.end_frame, b + padding)
            interpretation = ("평가 보류 — " + "; ".join(dict.fromkeys(reasons))) if deferred else ("잠정적 검토: " + rule["interpretation"])
            graphs = (event.racket_hand + "_elbow", event.racket_hand + "_shoulder", "trunk_tilt")
            message = CoachMessage(str(uuid.uuid5(uuid.NAMESPACE_URL, f"coach:{event.video_id}:{event.athlete_key}:{event.event_id}:{rule['id']}")),
                                   event.video_id, event.athlete_key, event.player_id, event.track_segment, event.event_id,
                                   tuple(rule["phases"]), rule["id"], rule["version"], level,
                                   "COACHING_DEFERRED" if deferred else "OBSERVATION_ONLY", observations, a, b,
                                   timeline.raw_pts_seconds[a], timeline.raw_pts_seconds[b], observation, interpretation,
                                   rule["suggestion"], rule_quality, tuple(dict.fromkeys(reasons)), graphs, created,
                                   rule_name=rule["name"])
            messages.append(message)
            log.info("코칭 규칙: event=%s rule=%s version=%s status=%s", event.event_id, rule["id"], rule["version"], message.status)
        reports.append(CoachReport(tuple(timeline.identity), source.video_id, source.context_id, event.event_id, event_key(event), key,
                                   rules.key, rules.version, ENGINE_VERSION, MEASUREMENT_VERSION, event.algorithm_version,
                                   created, quality, tuple(messages), values, statuses, comparison, asdict(event)))
        progress(round(100 * (index + 1) / max(1, len(targets))))
    return tuple(reports)
