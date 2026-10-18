# 원본 PTS와 선택 선수 캐시에서 신호처리 특징을 추출하는 순수 NumPy 모듈.
# 몸통 중심 상대 손목 좌표를 관측 구간의 중앙 몸통 길이로 정규화한다.
# 분석용 median 평활화/고립 좌표 이상치 배제는 3단계 표시 평활화와 독립적이다.
# 결측/다른 선수/동일·역전 PTS/큰 시간 간격을 넘어 평활화·미분하지 않는다.
# 속도는 화면상 몸통 길이/초이며 라켓 헤드 또는 실제 3D 속도가 아니다.
from dataclasses import dataclass
import numpy as np

MIN_TIME_STEP = 1e-6


@dataclass(frozen=True)
class MotionFeatures:
    times: np.ndarray
    valid: np.ndarray
    runs: np.ndarray
    player_ids: tuple
    track_segments: tuple
    reasons: tuple
    wrist_raw: np.ndarray
    wrist_smooth: np.ndarray
    velocity: np.ndarray
    speed: np.ndarray
    elbow_angle: np.ndarray
    shoulder_angle: np.ndarray
    wrist_height: np.ndarray
    elbow_rate: np.ndarray
    shoulder_rate: np.ndarray
    trunk_rate: np.ndarray
    energy: np.ndarray


def contiguous_runs(valid, times, labels, max_gap):
    runs = []
    start = None
    for i in range(len(times)):
        connected = (i > 0 and valid[i - 1] and labels[i] == labels[i - 1]
                     and MIN_TIME_STEP < times[i] - times[i - 1] <= max_gap)
        if not valid[i] or not connected:
            if start is not None:
                runs.append((start, i - 1))
            start = i if valid[i] else None
        elif start is None:
            start = i
    if start is not None:
        runs.append((start, len(times) - 1))
    return runs


def smooth_signal(values, times, labels, config, cancelled=lambda: False):
    values = np.asarray(values, dtype=float)
    output = values.copy()
    valid = np.isfinite(values).all(axis=1) if values.ndim == 2 else np.isfinite(values)
    if config.smoothing == "none" or config.window == 1:
        return output
    radius = config.window // 2
    for start, end in contiguous_runs(valid, times, labels, config.max_time_gap):
        for i in range(start, end + 1):
            if cancelled():
                return None
            output[i] = np.median(values[max(start, i - radius):min(end + 1, i + radius + 1)], axis=0)
    return output


def temporal_derivative(values, times, labels, max_gap):
    values, times = np.asarray(values, dtype=float), np.asarray(times, dtype=float)
    result = np.full(values.shape, np.nan)
    valid = np.isfinite(values).all(axis=1) if values.ndim == 2 else np.isfinite(values)
    for start, end in contiguous_runs(valid, times, labels, max_gap):
        if end > start:
            result[start:end + 1] = np.gradient(values[start:end + 1], times[start:end + 1], axis=0)
    return result


def local_extrema(values, maximum=True):
    values = np.asarray(values)
    indices = []
    for i, value in enumerate(values):
        if not np.isfinite(value):
            continue
        left = values[i - 1] if i else -np.inf if maximum else np.inf
        right = values[i + 1] if i + 1 < len(values) else -np.inf if maximum else np.inf
        if (value >= left and value > right) if maximum else (value <= left and value < right):
            indices.append(i)
    return indices


def extract_features(frames, selections, measurements, timeline, width, height, hand, config, cancelled=lambda: False):
    if hand not in ("left", "right") or not np.isfinite((width, height)).all() or min(width, height) <= 0:
        raise ValueError("라켓 손 또는 원본 영상 크기가 올바르지 않습니다.")
    n = len(timeline.seconds)
    times = np.asarray(timeline.seconds, dtype=float)
    raw = np.full((n, 2), np.nan)
    torso_size, angles, shoulder_angles, heights, tilt = (np.full(n, np.nan) for _ in range(5))
    valid, ids, segments, reasons = np.zeros(n, dtype=bool), [None] * n, [None] * n, ["unselected"] * n
    wrist_id, elbow_id = (15, 13) if hand == "left" else (16, 14)
    for i in range(n):
        if cancelled():
            return None
        frame, selected, measured = frames.get(i), selections.get(i), measurements.get(i)
        if selected is None or measured is None or frame is None:
            reasons[i] = "pose_frame_missing" if frame is None else "unselected" if selected is None else "measurement_missing"
            continue
        ids[i], segments[i] = selected.player_id, measured.track_segment
        if (selected.state not in ("selected", "tracked") or selected.selected_pose_index is None
                or measured.player_id != selected.player_id or measured.track_segment is None
                or not 0 <= selected.selected_pose_index < len(frame.poses)):
            reasons[i] = "tracking_unavailable"
            continue
        if timeline.sources[i] != "pts" or not np.isfinite(times[i]):
            reasons[i] = "original_pts_unavailable"
            continue
        if i and (times[i] - times[i - 1] <= MIN_TIME_STEP or times[i] - times[i - 1] > config.max_time_gap):
            reasons[i] = "time_discontinuity"
            continue
        if abs(measured.timestamp_seconds - times[i]) > 1e-6 or frame.frame_index != i:
            reasons[i] = "timestamp_mismatch"
            continue
        pose = frame.poses[selected.selected_pose_index]
        required = (11, 12, 23, 24, elbow_id, wrist_id)
        if not all(pose.joints[j].reliable for j in required):
            reasons[i] = "low_confidence"
            continue
        points = {j: np.array((pose.joints[j].x * width, pose.joints[j].y * height)) for j in required}
        shoulder_center, hip_center = (points[11] + points[12]) / 2, (points[23] + points[24]) / 2
        size = np.linalg.norm(shoulder_center - hip_center)
        elbow_metric = measured.values[f"{hand}_elbow"]
        if size < 1 or not elbow_metric.valid or elbow_metric.raw is None or not np.isfinite(elbow_metric.raw):
            reasons[i] = "degenerate_or_missing_arm"
            continue
        raw[i], torso_size[i] = points[wrist_id] - shoulder_center, size
        angles[i] = measured.values[f"{hand}_elbow"].raw
        shoulder_angles[i] = measured.values[f"{hand}_shoulder"].raw if measured.values[f"{hand}_shoulder"].valid else np.nan
        heights[i] = measured.values[f"{hand}_wrist_height"].raw if measured.values[f"{hand}_wrist_height"].valid else np.nan
        tilt[i] = measured.values["trunk_tilt"].raw if measured.values["trunk_tilt"].valid else np.nan
        valid[i], reasons[i] = True, ""
    labels = [(ids[i], segments[i]) for i in range(n)]
    for start, end in contiguous_runs(valid, times, labels, config.max_time_gap):
        raw[start:end + 1] /= np.median(torso_size[start:end + 1])
        # 고립된 큰 좌표 튐만 제외한다. 결측을 중앙값으로 대체하지 않는다.
        for i in range(start + 1, end):
            if cancelled():
                return None
            neighborhood = raw[max(start, i - 3):min(end + 1, i + 4)]
            center = np.median(neighborhood, axis=0)
            residuals = np.linalg.norm(neighborhood - center, axis=1)
            threshold = max(0.65, 6 * float(np.median(residuals)))
            if np.linalg.norm(raw[i] - center) > threshold:
                valid[i], reasons[i] = False, "coordinate_spike"
    raw[~valid], angles[~valid], shoulder_angles[~valid], heights[~valid], tilt[~valid] = np.nan, np.nan, np.nan, np.nan, np.nan
    runs = np.full(n, -1, dtype=int)
    for run_id, (start, end) in enumerate(contiguous_runs(valid, times, labels, config.max_time_gap)):
        runs[start:end + 1] = run_id
    signals = []
    for values in (raw, angles, shoulder_angles, tilt):
        signal = smooth_signal(values, times, runs, config, cancelled)
        if signal is None:
            return None
        signals.append(signal)
    smoothed, smooth_elbow, smooth_shoulder, smooth_tilt = signals
    velocity = temporal_derivative(smoothed, times, runs, config.max_time_gap)
    speed = np.linalg.norm(velocity, axis=1)
    elbow_rate = temporal_derivative(smooth_elbow, times, runs, config.max_time_gap)
    shoulder_rate = temporal_derivative(smooth_shoulder, times, runs, config.max_time_gap)
    trunk_rate = temporal_derivative(smooth_tilt, times, runs, config.max_time_gap)
    energy = speed ** 2 + 0.25 * np.nan_to_num(elbow_rate / 180) ** 2 + 0.15 * np.nan_to_num(shoulder_rate / 180) ** 2 + 0.1 * np.nan_to_num(trunk_rate / 90) ** 2
    return MotionFeatures(times, valid, runs, tuple(ids), tuple(segments), tuple(reasons), raw, smoothed,
                          velocity, speed, angles, shoulder_angles, heights, elbow_rate, shoulder_rate, trunk_rate, energy)
