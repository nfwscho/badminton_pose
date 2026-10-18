# 선택 선수의 연속 관측 구간에서 적응 임계값/국소 피크/이동량/팔 각도 변화로
# 독립 스윙 후보를 검출한다. A/B는 별도 범위로 적용하며 추적/시간 단절을 병합하지 않는다.
# 오버헤드의 화면상 증거가 충분할 때만 단계 규칙을 적용하고 모든 결과는 검토 대상이다.
# 이벤트 ID는 입력 문맥/피크에 결정적이며 수동 편집은 원본 이벤트를 변경하지 않는다.
from dataclasses import asdict, replace
import numpy as np
import uuid

from app.analysis.motion_features import local_extrema
from app.analysis.motion_models import (ALGORITHM_VERSION, MotionDocument, MotionEvent, MotionInterval,
                                        MotionResult, fingerprint, normalize_phases, validate_event)
from app.analysis.motion_phases import classify_phases


def _finite_range(values):
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    return float(np.ptp(finite)) if len(finite) >= 2 else None


def phase_durations(phases):
    durations = {}
    for phase in phases:
        durations[phase.phase] = durations.get(phase.phase, 0.0) + phase.end_seconds - phase.start_seconds
    return durations


def event_statistics(features, measurements, start, end, phases, hand):
    stats = {"wrist_path_torso_lengths": None, "elbow_min": None, "elbow_max": None,
             "shoulder_min": None, "shoulder_max": None, "forward_elbow_change": None,
             "peak_energy_frame": None, "peak_energy_seconds": None,
             "phase_durations": phase_durations(phases)}
    if features is None:
        return stats
    position, runs = features.wrist_smooth, features.runs
    steps = [np.linalg.norm(position[i] - position[i - 1]) for i in range(start + 1, end + 1)
             if features.valid[i] and features.valid[i - 1] and runs[i] >= 0 and runs[i] == runs[i - 1]
             and np.isfinite(position[i]).all() and np.isfinite(position[i - 1]).all()]
    if steps:
        stats["wrist_path_torso_lengths"] = float(sum(steps))
    for name, values in (("elbow", features.elbow_angle), ("shoulder", features.shoulder_angle)):
        finite = values[start:end + 1][np.isfinite(values[start:end + 1])]
        if len(finite) >= 2:
            stats[name + "_min"], stats[name + "_max"] = float(finite.min()), float(finite.max())
    for phase in phases:
        if phase.phase == "FORWARD_SWING":
            values = features.elbow_angle[phase.start_frame:phase.end_frame + 1]
            if len(values) >= 2 and np.isfinite(values).all():
                stats["forward_elbow_change"] = float(values[-1] - values[0])
    energy = features.energy[start:end + 1]
    if np.isfinite(energy).any():
        peak = start + int(np.nanargmax(energy))
        stats["peak_energy_frame"], stats["peak_energy_seconds"] = peak, float(features.times[peak])
    return stats


def _make_event(features, measurements, timeline, context, hand, start, end, peak, kind,
                phases, indicators, evidence, uncertainty, clipped, event_id=None, manual=False, user_label=""):
    ratio = float(np.mean(features.valid[start:end + 1])) if features is not None else None
    measured = measurements.get(peak) if peak is not None and measurements else None
    angles = {name: value.raw if value.valid else None for name, value in measured.values.items()} if measured else {}
    segment = features.track_segments[peak] if features is not None and peak is not None else context.get("track_segment", "manual_unverified")
    event = MotionEvent(event_id or str(uuid.uuid5(uuid.NAMESPACE_URL, f"{context['context_id']}:{peak}:{ALGORITHM_VERSION}")),
                        context["video_id"], context["athlete_key"], context["player_id"], segment or "unverified",
                        hand, kind, start, end, timeline.seconds[start], timeline.frame_end(end),
                        timeline.raw_pts_seconds[start], timeline.raw_pts_seconds[end],
                        timeline.frame_end(end) - timeline.seconds[start], tuple(phases), peak,
                        timeline.seconds[peak] if peak is not None else None,
                        timeline.raw_pts_seconds[peak] if peak is not None else None,
                        angles, indicators, event_statistics(features, measurements, start, end, phases, hand), ratio,
                        tuple(evidence), tuple(uncertainty), "manual" if manual else "automatic", manual, user_label, clipped)
    validate_event(event, timeline)
    return event


def _active_windows(speed, times, start, end, low, merge_gap):
    windows, left = [], None
    for i in range(start, end + 1):
        if np.isfinite(speed[i]) and speed[i] >= low:
            left = i if left is None else left
        elif left is not None:
            windows.append([left, i - 1])
            left = None
    if left is not None:
        windows.append([left, end])
    merged = []
    for a, b in windows:
        if merged and times[a] - times[merged[-1][1]] <= merge_gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return merged


def detect_motion(features, measurements, timeline, context, hand, config, scope=None, cancelled=lambda: False):
    scope = scope or (0, len(timeline.seconds) - 1)
    if not 0 <= scope[0] <= scope[1] < len(timeline.seconds):
        raise ValueError("동작 분석 A/B 범위가 올바르지 않습니다.")
    events, eligible_runs = [], set()
    for run in sorted(set(features.runs[scope[0]:scope[1] + 1]) - {-1}):
        if cancelled():
            return None
        indices = np.flatnonzero((features.runs == run) & features.valid & np.isfinite(features.speed))
        indices = indices[(indices >= scope[0]) & (indices <= scope[1])]
        if len(indices) < 3:
            continue
        start, end = int(indices[0]), int(indices[-1])
        if features.player_ids[start] != context["player_id"]:
            continue
        eligible_runs.add(run)
        speeds = features.speed[indices]
        baseline = float(np.percentile(speeds, 20))
        lower = speeds[speeds <= np.median(speeds)]
        noise = float(np.median(np.abs(lower - baseline)))
        high = max(config.speed_floor / config.sensitivity, baseline + 3 * max(noise, 0.05) / config.sensitivity)
        low = max(0.18, high * 0.4)
        windows = _active_windows(features.speed, features.times, start, end, low, config.merge_gap)
        candidates = []
        for a, b in windows:
            local = features.speed[a:b + 1]
            peaks = [a + i for i in local_extrema(local) if local[i] >= high]
            if not peaks:
                continue
            # 저속 골짜기를 동반한 독립 피크를 하나의 이벤트로 고정하지 않는다.
            splits = []
            strong = []
            for peak in peaks:
                if not strong or features.times[peak] - features.times[strong[-1]] >= config.min_duration:
                    strong.append(peak)
                elif features.speed[peak] > features.speed[strong[-1]]:
                    strong[-1] = peak
            for p, q in zip(strong, strong[1:]):
                valley = p + int(np.argmin(features.speed[p:q + 1]))
                valley_threshold = min(features.speed[p], features.speed[q]) * 0.5
                va, vb = valley, valley
                while va > p and features.speed[va - 1] < valley_threshold:
                    va -= 1
                while vb < q and features.speed[vb + 1] < valley_threshold:
                    vb += 1
                if (features.speed[valley] < valley_threshold and features.times[vb] - features.times[va] >= max(0.12, config.merge_gap)
                        and features.times[q] - features.times[p] >= max(0.4, config.min_duration * 2)):
                    splits.append(valley)
            left = a
            for right in (*splits, b):
                if right < left:
                    continue
                peak = left + int(np.argmax(features.speed[left:right + 1]))
                duration = timeline.frame_end(right) - timeline.seconds[left]
                excursion = float(np.linalg.norm(np.ptp(features.wrist_smooth[left:right + 1], axis=0)))
                elbow_change = _finite_range(features.elbow_angle[left:right + 1]) or 0
                shoulder_change = _finite_range(features.shoulder_angle[left:right + 1]) or 0
                height_change = _finite_range(features.wrist_height[left:right + 1]) or 0
                edge = (left == scope[0] and left > 0) or (right == scope[1] and right + 1 < len(features.times))
                combined = elbow_change >= 12 or shoulder_change >= 12 or height_change >= 0.2
                if (features.speed[peak] >= high and excursion >= config.min_excursion
                        and (duration >= config.min_duration or edge) and combined):
                    candidates.append((left, right, peak, excursion, elbow_change, shoulder_change))
                left = right + 1
        for ci, (a, b, peak, excursion, elbow_change, shoulder_change) in enumerate(candidates):
            event_start, event_end = a, b
            left_limit = start if ci == 0 else (candidates[ci - 1][1] + a) // 2 + 1
            right_limit = end if ci + 1 == len(candidates) else (b + candidates[ci + 1][0]) // 2
            while event_start > left_limit and features.times[a] - features.times[event_start - 1] <= config.context_seconds:
                event_start -= 1
            while event_end < right_limit and features.times[event_end + 1] - features.times[b] <= config.context_seconds:
                event_end += 1
            heights = features.wrist_height[event_start:event_end + 1]
            finite = heights[np.isfinite(heights)]
            raised_time = sum(timeline.frame_end(i) - timeline.seconds[i] for i in range(event_start, event_end + 1)
                              if np.isfinite(features.wrist_height[i]) and features.wrist_height[i] > 0.1)
            shoulder = features.shoulder_angle[event_start:event_end + 1]
            overhead = (raised_time >= 0.06 and len(finite) >= 3 and
                        ((np.isfinite(shoulder).any() and float(np.nanmax(shoulder)) >= 70)
                         or float(np.max(finite)) >= 0.3))
            kind = "OVERHEAD_CANDIDATE" if overhead else "OTHER_SWING" if len(finite) >= 3 and finite.max() < -0.15 else "UNKNOWN"
            if len(finite) / len(heights) < config.min_valid_ratio:
                kind = "UNKNOWN"
            short_partial = timeline.frame_end(b) - timeline.seconds[a] < config.min_duration
            if short_partial:
                kind = "UNKNOWN"
            clipped = ((event_start == scope[0] and scope[0] > 0) or
                       (event_end == scope[1] and scope[1] < len(features.times) - 1))
            phases = classify_phases(features, timeline, event_start, event_end, a, b, peak, kind, low)
            indicators = {"peak_wrist_speed_torso_per_second": float(features.speed[peak]),
                          "relative_wrist_excursion": excursion, "elbow_angle_range": elbow_change,
                          "shoulder_angle_range": shoulder_change, "raised_arm_seconds": float(raised_time),
                          "adaptive_high_threshold": high, "adaptive_low_threshold": low,
                          "peak_motion_energy": float(features.energy[peak])}
            uncertainty = ["규칙 기반 후보 — 실제 영상 수동 검토 필요", "화면상 상대 운동이며 구질·라켓속도·임팩트 시점 미확정"]
            if clipped:
                uncertainty.append("A/B 경계에서 동작 또는 전후 문맥이 잘림")
            if short_partial:
                uncertainty.append("잘린 활동 구간이 최소 지속 시간보다 짧아 분류 보류")
            if event_start == start or event_end == end:
                uncertainty.append("관측 구간 가장자리: 촬영/추적 밖의 단계는 판단 불가")
            events.append(_make_event(features, measurements, timeline, context, hand, event_start, event_end, peak, kind,
                                      phases, indicators, ("국소 속도 피크 + 상대 이동량 + 팔 각도/높이 변화",), uncertainty, clipped))
    events.sort(key=lambda event: event.start_frame)
    for event in events:
        validate_event(event, timeline, events)
    coverage = []
    event_index = 0
    for i in range(scope[0], scope[1] + 1):
        valid = (features.valid[i] and np.isfinite(features.speed[i]) and features.player_ids[i] == context["player_id"]
                 and features.runs[i] in eligible_runs)
        while event_index < len(events) and events[event_index].end_frame < i:
            event_index += 1
        candidate = events[event_index] if event_index < len(events) and events[event_index].start_frame <= i else None
        if candidate:
            kind, reason = candidate.kind, "swing_candidate"
        elif valid:
            kind, reason = "NO_SWING", "유효 관측, 이벤트 외 구간"
        else:
            kind, reason = "UNKNOWN", features.reasons[i] or "insufficient_temporal_samples_or_foreign_player"
        if coverage and coverage[-1].kind == kind and coverage[-1].reason == reason and coverage[-1].end_frame + 1 == i:
            coverage[-1] = replace(coverage[-1], end_frame=i)
        else:
            coverage.append(MotionInterval(i, i, kind, reason))
    key = fingerprint((context, hand, asdict(config), scope, ALGORITHM_VERSION,
                       [(int(i), round(float(features.speed[i]), 6)) for i in range(scope[0], scope[1] + 1) if np.isfinite(features.speed[i])]))
    return MotionResult(tuple(timeline.identity), context["video_id"], context["athlete_key"], context["context_id"],
                        context["player_id"], hand, tuple(scope), config, tuple(events), tuple(coverage), key, context["source_anchor"])


def manual_event(document, timeline, start, end, kind, spans, user_label="", existing=None,
                 features=None, measurements=None):
    if type(start) is not int or type(end) is not int or not 0 <= start <= end < len(timeline.seconds):
        raise ValueError("시작/종료 프레임을 확인하세요.")
    phases = normalize_phases(spans, start, end, timeline)
    result = document.automatic
    if features is not None and any(features.valid[i] and features.player_ids[i] != result.player_id for i in range(start, end + 1)):
        raise ValueError("다른 선택 세션의 관측을 수동 이벤트에 섞을 수 없습니다.")
    context = {"context_id": result.context_id, "video_id": result.video_id, "athlete_key": result.athlete_key,
               "player_id": result.player_id, "track_segment": existing.track_segment if existing else "manual_unverified"}
    peak = None
    if features is not None and np.isfinite(features.speed[start:end + 1]).any():
        peak = start + int(np.nanargmax(features.speed[start:end + 1]))
    event = _make_event(features, measurements or {}, timeline, context, result.racket_hand, start, end, peak, kind, phases,
                        {}, ("사용자가 라벨/경계를 지정함",), ("수동 수정은 자동 확정 판정이 아님",),
                        start < result.scope[0] or end > result.scope[1],
                        existing.event_id if existing else "manual-" + str(uuid.uuid4()), True, user_label.strip())
    if features is None and existing is not None and (start, end) == (existing.start_frame, existing.end_frame):
        event = replace(event, peak_frame=existing.peak_frame, peak_seconds=existing.peak_seconds, peak_pts=existing.peak_pts,
                        peak_angles=existing.peak_angles, indicators=existing.indicators, valid_ratio=existing.valid_ratio,
                        details={**existing.details, "phase_durations": phase_durations(phases)})
    validate_event(event, timeline, document.events)
    return event
