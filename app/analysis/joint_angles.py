# 선택 선수의 검증된 관절 캐시에서 영상 평면의 각도/높이를 계산하는 순수 모듈.
# NumPy로 원본 해상도의 x/y 축척을 적용하고 결측/저신뢰도/영벡터를 배제한다.
# 신원 세션 및 연속 구간을 보존하며 원시값과 표시용 3점 평활화 값을 구분한다.
# 통계는 원시값만 사용하고 CSV에는 원시/표시값, 유효성/사유 및 영상/PTS를 기록한다.
# 실제 3D 관절 회전, 자세 품질, 운동 분류 또는 코칭 판단은 수행하지 않는다.
import csv
from dataclasses import dataclass, replace
import os
from pathlib import Path
import tempfile

import numpy as np

METRICS = {
    "left_elbow": ("왼쪽 팔꿈치", "도"), "right_elbow": ("오른쪽 팔꿈치", "도"),
    "left_knee": ("왼쪽 무릎", "도"), "right_knee": ("오른쪽 무릎", "도"),
    "left_shoulder": ("왼쪽 어깨(화면)", "도"), "right_shoulder": ("오른쪽 어깨(화면)", "도"),
    "trunk_tilt": ("몸통 기울기(화면)", "도"),
    "left_wrist_height": ("왼쪽 손목 상대높이", "몸통길이"),
    "right_wrist_height": ("오른쪽 손목 상대높이", "몸통길이"),
}
TRIPLETS = {"left_elbow": (11, 13, 15), "right_elbow": (12, 14, 16),
            "left_knee": (23, 25, 27), "right_knee": (24, 26, 28),
            "left_shoulder": (13, 11, 23), "right_shoulder": (14, 12, 24)}
EPSILON = 1e-6


@dataclass(frozen=True, slots=True)
class MetricValue:
    raw: float | None
    valid: bool
    reason: str = ""
    display: float | None = None


@dataclass(frozen=True, slots=True)
class FrameMeasurement:
    frame_index: int
    timestamp_seconds: float
    timestamp_source: str
    player_id: int | None
    track_segment: str | None
    tracking_state: str
    values: dict[str, MetricValue]
    stop_frame: int | None = None
    stop_reason: str | None = None

    @property
    def analyzable(self):
        return any(value.valid for value in self.values.values())


def angle_degrees(a, b, c):
    a, b, c = (np.asarray(p, dtype=float) for p in (a, b, c))
    if not all(p.shape == (2,) and np.isfinite(p).all() for p in (a, b, c)):
        return None
    u, v = a - b, c - b
    lu, lv = np.linalg.norm(u), np.linalg.norm(v)
    if not np.isfinite((lu, lv)).all() or min(lu, lv) <= EPSILON:
        return None
    return float(np.clip(np.degrees(np.arccos(np.clip(np.dot(u / lu, v / lv), -1, 1))), 0, 180))


def measure_pose(pose, width, height):
    if not np.isfinite((width, height)).all() or min(width, height) <= 0:
        raise ValueError("측정할 원본 영상 크기가 올바르지 않습니다.")
    values = {}

    def point(i):
        joint = pose.joints[i]
        return np.array((joint.x * width, joint.y * height), dtype=float) if joint.reliable else None

    for name, indices in TRIPLETS.items():
        points = [point(i) for i in indices]
        result = angle_degrees(*points) if all(p is not None for p in points) else None
        values[name] = MetricValue(result, result is not None,
                                   "" if result is not None else "missing_or_degenerate_joint", result)
    torso = [point(i) for i in (11, 12, 23, 24)]
    shoulder_center = hip_center = None
    length = None
    if all(p is not None for p in torso):
        shoulder_center, hip_center = (torso[0] + torso[1]) / 2, (torso[2] + torso[3]) / 2
        length = float(np.linalg.norm(shoulder_center - hip_center))
        if length <= EPSILON:
            length = None
    tilt = angle_degrees(shoulder_center, hip_center, hip_center + np.array((0, -1))) if length else None
    values["trunk_tilt"] = MetricValue(tilt, tilt is not None, "" if tilt is not None else "missing_or_degenerate_torso", tilt)
    for side, wrist, shoulder in (("left", 15, 11), ("right", 16, 12)):
        w, s = point(wrist), point(shoulder)
        # 화면 Y는 아래쪽이 양수이므로 어깨보다 위에 있는 손목 높이가 양수다.
        value = float((s[1] - w[1]) / length) if length and w is not None and s is not None else None
        values[f"{side}_wrist_height"] = MetricValue(value, value is not None,
                                                     "" if value is not None else "missing_or_degenerate_joint", value)
    return values


def calculate_measurements(frames, selections, timeline, width, height, cancelled=lambda: False):
    if not np.isfinite((width, height)).all() or min(width, height) <= 0:
        raise ValueError("측정할 원본 영상 크기가 올바르지 않습니다.")
    measurements = {}
    segment_id, previous_index, previous_player = None, -2, None
    for index, seconds in enumerate(timeline.seconds):
        if cancelled():
            return {}
        frame, selected = frames.get(index), selections.get(index)
        player = selected.player_id if selected else None
        state = selected.state if selected else "unselected"
        valid = (frame is not None and selected is not None and state in ("selected", "tracked")
                 and selected.selected_pose_index is not None and 0 <= selected.selected_pose_index < len(frame.poses)
                 and frame.poses[selected.selected_pose_index].trackable)
        if valid:
            if index != previous_index + 1 or player != previous_player:
                segment_id = f"{player}:{index}"
            previous_index, previous_player = index, player
            values = measure_pose(frame.poses[selected.selected_pose_index], width, height)
        else:
            values = {name: MetricValue(None, False, "tracking_unavailable" if selected else "unselected") for name in METRICS}
        measurements[index] = FrameMeasurement(index, seconds, timeline.sources[index], player,
                                                segment_id if valid else None, state, values,
                                                selected.stop_frame if selected else None,
                                                selected.stop_reason if selected else None)
    # 결측을 넘어 연결하거나 다른 세션의 값을 평활화하지 않는다.
    for index, measurement in list(measurements.items()):
        if cancelled():
            return {}
        values = dict(measurement.values)
        for name, metric in values.items():
            if not metric.valid:
                continue
            neighbors = [metric.raw]
            for neighbor_index in (index - 1, index + 1):
                neighbor = measurements.get(neighbor_index)
                if neighbor and neighbor.track_segment == measurement.track_segment and neighbor.values[name].valid:
                    neighbors.append(neighbor.values[name].raw)
            values[name] = replace(metric, display=float(np.mean(neighbors)))
        measurements[index] = replace(measurement, values=values)
    return measurements


def measurement_bounds(measurements, count, a=None, b=None):
    if not count:
        return None
    if a is not None or b is not None:
        return (a if a is not None else 0, b if b is not None else count - 1)
    tracked = [i for i, m in measurements.items() if m.track_segment is not None]
    return (min(tracked), max(tracked)) if tracked else None


def summarize(measurements, timeline, bounds):
    selected = [m for i, m in measurements.items() if bounds and bounds[0] <= i <= bounds[1]]
    total = bounds[1] - bounds[0] + 1 if bounds else 0
    stats = {}
    for name in METRICS:
        values = [m.values[name].raw for m in selected if m.values[name].valid]
        enough = len(values) >= 2
        stats[name] = {"count": len(values), "min": min(values) if enough else None,
                       "max": max(values) if enough else None,
                       "range": max(values) - min(values) if enough else None}
    usable = sum(m.analyzable for m in selected)
    return {"metrics": stats, "total_frames": total, "usable_frames": usable,
            "ratio": usable / total if total else None,
            "duration": timeline.frame_end(bounds[1]) - timeline.seconds[bounds[0]] if bounds else None}


def export_csv(path, measurements, timeline, bounds):
    path = Path(path)
    fields = ["video_path", "video_size", "video_mtime_ns", "width", "height", "frame_index", "timestamp_seconds",
              "original_pts_seconds", "timestamp_source", "player_session_id", "track_segment", "tracking_state",
              "tracking_stop_frame", "tracking_stop_reason"]
    for name in METRICS:
        fields.extend((name, f"{name}_display_smoothed", f"{name}_valid", f"{name}_reason"))
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8-sig", newline="", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            for index in sorted(measurements):
                if not bounds or not bounds[0] <= index <= bounds[1]:
                    continue
                m = measurements[index]
                row = dict(zip(fields[:5], (*timeline.identity, timeline.width, timeline.height)))
                row.update(frame_index=index, timestamp_seconds=m.timestamp_seconds,
                           original_pts_seconds=timeline.raw_pts_seconds[index], timestamp_source=m.timestamp_source,
                           player_session_id=m.player_id, track_segment=m.track_segment, tracking_state=m.tracking_state,
                           tracking_stop_frame=m.stop_frame, tracking_stop_reason=m.stop_reason)
                for name, value in m.values.items():
                    row.update({name: value.raw, f"{name}_display_smoothed": value.display,
                                f"{name}_valid": int(value.valid), f"{name}_reason": value.reason})
                writer.writerow(row)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
