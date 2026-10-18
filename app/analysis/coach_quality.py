# 스윙별 코칭 전제 검사: 실제 PTS, 같은 선수/추적 구간, 관절 유효 관측과 단계 확인.
# 내부 비율은 관측 수 비율이며 정확도 확률이 아니다. 자동 단계와 사용자 확인을 구분한다.
# 규칙별 필요 특징 검사는 엔진에서 추가한다. 기술 표준 미검증인 V1은 측정 관찰만 허용한다.
# 신원/시간 불일치는 부분 관측으로도 우회하지 못하도록 무결성 실패로 표시한다.
import math
from app.analysis.coach_models import CoachQuality
from app.analysis.motion_models import validate_event


def assess_quality(event, features, measurements, selections, timeline, confirmed, minimum, max_gap, frames=None):
    reasons, integrity = [], True
    try:
        validate_event(event, timeline)
    except (ValueError, IndexError, TypeError):
        reasons.append("이벤트 경계/PTS 유효성 실패")
        integrity = False
    good, total = 0, event.end_frame - event.start_frame + 1
    if not 0 <= event.start_frame <= event.end_frame < len(timeline.seconds):
        return CoachQuality("COACHING_DEFERRED", "INVALID", 0, max(total, 0), 0,
                            "UNKNOWN", "NOT_APPLICABLE", confirmed, tuple(reasons), False)
    segments, runs = set(), set()
    for i in range(event.start_frame, event.end_frame + 1):
        m, s = measurements.get(i), selections.get(i)
        frame = frames.get(i) if frames is not None else None
        if (s and s.frame_index != i) or (frame and (frame.frame_index != i or frame.timestamp_seconds is None
                or not math.isfinite(frame.timestamp_seconds) or abs(frame.timestamp_seconds - timeline.seconds[i]) > 1e-6)):
            integrity = False
            reasons.append("관절/선택 프레임의 원본 시간 불일치")
        if ((m and (m.player_id != event.player_id or m.track_segment != event.track_segment))
                or (s and s.player_id != event.player_id)
                or (features.valid[i] and (features.player_ids[i] != event.player_id
                                          or features.track_segments[i] != event.track_segment))):
            integrity = False
            reasons.append("다른 선수/추적 구간 데이터 또는 세션 불일치")
        if (timeline.sources[i] != "pts" or timeline.raw_pts_seconds[i] is None
                or not math.isfinite(timeline.raw_pts_seconds[i])
                or not math.isfinite(timeline.seconds[i])
                or (i > event.start_frame and not 1e-6 < timeline.seconds[i] - timeline.seconds[i - 1] <= max_gap)):
            integrity = False
            reasons.append("원본 PTS 누락/역행/시간 단절")
        if i > event.start_frame and timeline.raw_pts_seconds[i] is not None and timeline.raw_pts_seconds[i - 1] is not None:
            if not 1e-6 < timeline.raw_pts_seconds[i] - timeline.raw_pts_seconds[i - 1] <= max_gap:
                integrity = False
                reasons.append("원본 PTS 시간 단절")
        if m and (m.frame_index != i or m.timestamp_source != "pts"
                  or not math.isfinite(m.timestamp_seconds) or abs(m.timestamp_seconds - timeline.seconds[i]) > 1e-6):
            integrity = False
            reasons.append("측정 프레임/시간 불일치")
        valid = (features.valid[i] and m is not None and s is not None
                 and m.tracking_state in ("selected", "tracked") and s.state in ("selected", "tracked")
                 and s.selected_pose_index is not None)
        if valid:
            good += 1
            segments.add(m.track_segment)
            runs.add(int(features.runs[i]))
    ratio = good / total
    if good != total or len(segments) != 1 or len(runs) != 1 or -1 in runs:
        reasons.append("관절 결측/가림 또는 추적 연속성 부족")
    if ratio < minimum:
        reasons.append("필요 관절의 유효 관측 비율 부족")
    known = any(p.phase != "UNKNOWN_PHASE" for p in event.phases)
    partial = any(p.phase == "UNKNOWN_PHASE" for p in event.phases)
    if not confirmed:
        reasons.append("사용자의 현재 스윙/단계 확인 전 — 자동 후보임")
    if not known or partial:
        reasons.append("단계 구분 전체 또는 일부 불확실")
    if event.kind != "OVERHEAD_CANDIDATE":
        reasons.append("오버헤드 규칙의 동작 적용 근거 부족")
    deferred = (not integrity or good != total or len(runs) != 1 or ratio < minimum
                or not confirmed or not known or partial or event.kind != "OVERHEAD_CANDIDATE")
    return CoachQuality("COACHING_DEFERRED" if deferred else "MEASUREMENT_ONLY",
                        "SUFFICIENT" if integrity and ratio >= minimum else "INSUFFICIENT", good, total, ratio,
                        "UNKNOWN" if not known else "PARTIAL_CANDIDATE" if partial else "USER_REVIEWED_CANDIDATE" if confirmed else "CANDIDATE",
                        "OBSERVATION_RULES_ONLY", confirmed, tuple(dict.fromkeys(reasons)), integrity)
