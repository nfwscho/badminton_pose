# 오버헤드 후보의 관측 신호에서 단계 경계를 추정하는 규칙 모듈.
# 저속 준비, 상방 손목 이동/방향 전환, 가속·팔 각도 변화, 피크 이후 감속,
# 저속 안정화를 근거로 사용한다. 균등 분할하지 않으며 관찰 근거가 없으면
# UNKNOWN_PHASE로 남긴다. 손목 움직임 피크를 셔틀콕 임팩트라고 해석하지 않는다.
import numpy as np
from app.analysis.motion_models import normalize_phases, phase_span


def _stable_start(features, start, end, threshold, minimum=0.08):
    run_start = None
    for i in range(start, end + 1):
        if np.isfinite(features.speed[i]) and features.speed[i] <= threshold:
            run_start = i if run_start is None else run_start
            if features.times[i] - features.times[run_start] >= minimum:
                return run_start
        else:
            run_start = None
    return None


def classify_phases(features, timeline, start, end, active_start, active_end, peak, kind, low_threshold):
    if kind != "OVERHEAD_CANDIDATE":
        return normalize_phases([], start, end, timeline)
    spans = []
    if active_start > start and timeline.seconds[active_start] - timeline.seconds[start] >= 0.08:
        if np.nanmedian(features.speed[start:active_start]) <= low_threshold:
            spans.append(phase_span("READY", start, active_start - 1, timeline, "스윙 직전 손목 저속 안정 구간"))
    heights = features.wrist_height[active_start:peak + 1]
    turn = None
    if len(heights) >= 3 and np.isfinite(heights).all():
        candidate = active_start + int(np.argmax(heights))
        if active_start < candidate < peak and heights.max() - heights[0] >= 0.12:
            turn = candidate
            spans.append(phase_span("BACKSWING", active_start, turn, timeline,
                                   "피크 이전 손목의 화면상 상승과 높이 최대/방향 전환 (실제 후방 방향 미확정)"))
    forward_start = turn + 1 if turn is not None else active_start
    if peak > forward_start:
        speeds = features.speed[forward_start:peak + 1]
        elbow = features.elbow_angle[forward_start:peak + 1]
        shoulder = features.shoulder_angle[forward_start:peak + 1]
        angle_change = max(float(np.ptp(elbow)) if np.isfinite(elbow).all() else 0,
                           float(np.ptp(shoulder)) if np.isfinite(shoulder).all() else 0)
        if speeds[-1] - speeds[0] >= max(0.2, speeds[-1] * 0.12) and angle_change >= 8:
            spans.append(phase_span("FORWARD_SWING", forward_start, peak, timeline,
                                   "손목 화면상 속도 증가와 팔 각도 변화가 함께 관측됨; 끝은 움직임 피크 후보"))
    stable = _stable_start(features, peak + 1, end, low_threshold)
    follow_end = stable - 1 if stable is not None else min(active_end, end)
    if follow_end > peak and np.isfinite(features.speed[peak:follow_end + 1]).all():
        if features.speed[follow_end] < features.speed[peak] * 0.85:
            spans.append(phase_span("FOLLOW_THROUGH", peak + 1, follow_end, timeline, "주요 움직임 피크 이후 손목 감속 관측"))
    if stable is not None and stable <= end:
        spans.append(phase_span("RECOVERY", stable, end, timeline,
                               "피크 이후 저속 안정화; 실제 준비 자세로의 복귀 여부는 수동 검토 필요"))
    return normalize_phases(spans, start, end, timeline)
