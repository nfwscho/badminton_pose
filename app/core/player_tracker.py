# 선수 선택과 보수적인 프레임 간 신원 연결. MediaPipe 출력 순서를 ID로 사용하지
# 않고 몸통 위치/크기/관절 이동을 비교한다. 겹침, 누락, 큰 이동 또는 경쟁 후보가
# 있으면 연결을 중단하고 이후 자동 복구하지 않는다. 선택 프레임에서 앞뒤로
# 독립 추적하며 별도 QThread에서 캐시를 처리하여 재생/GUI 응답성을 유지한다.
from dataclasses import dataclass
import logging
import math

from PySide6.QtCore import QThread, Signal

log = logging.getLogger(__name__)
TORSO = (11, 12, 23, 24)


@dataclass(frozen=True, slots=True)
class FrameSelection:
    frame_index: int
    selected_pose_index: int | None
    state: str
    player_id: int = 1  # 이 선택 세션 안에서만 유효하며 영구 신원 ID가 아니다.


def contains(box, x, y):
    return box is not None and box[0] <= x <= box[2] and box[1] <= y <= box[3]


def pick_person(poses, x, y):
    candidates = [i for i, p in enumerate(poses) if p.trackable and contains(p.bbox, x, y)]
    # 영역이 겹치면 임의의 가까운 선수를 택하지 않는다.
    if len(candidates) != 1:
        return None
    chosen = candidates[0]
    return None if crowded(poses[chosen], poses, chosen) else chosen


def descriptor(pose):
    points = [pose.joints[i] for i in TORSO]
    cx = sum(p.x for p in points) / 4
    cy = sum(p.y for p in points) / 4
    scale = math.hypot(points[0].x - points[2].x, points[0].y - points[2].y)
    return cx, cy, max(scale, 0.01)


def distance(a, b):
    ax, ay, scale = descriptor(a)
    bx, by, other_scale = descriptor(b)
    if not 0.65 <= other_scale / scale <= 1.55:
        return float("inf")
    if math.hypot(ax - bx, ay - by) > scale * 0.6:
        return float("inf")
    return sum(math.hypot(a.joints[i].x - b.joints[i].x,
                          a.joints[i].y - b.joints[i].y) for i in TORSO) / (4 * scale)


def crowded(target, poses, target_index):
    cx, cy, scale = descriptor(target)
    for i, pose in enumerate(poses):
        if i == target_index or not pose.trackable:
            continue
        ox, oy, other_scale = descriptor(pose)
        if math.hypot(cx - ox, cy - oy) < max(scale, other_scale) * 1.1:
            return True
    return False


def match_person(previous, previous_poses, previous_index, poses):
    candidates = sorted((distance(previous, p), i) for i, p in enumerate(poses) if p.trackable)
    if not candidates or candidates[0][0] > 0.55:
        return None, "lost"
    score, index = candidates[0]
    if len(candidates) > 1 and candidates[1][0] - score < 0.25:
        return None, "ambiguous"
    target = poses[index]
    if crowded(target, poses, index):
        return None, "ambiguous"
    # 다른 선수의 이전 위치와도 비슷하면 선택 선수로 확정하지 않는다.
    for i, competitor in enumerate(previous_poses):
        if i != previous_index and competitor.trackable and distance(competitor, target) <= score + 0.2:
            return None, "ambiguous"
    return index, "tracked"


def track_selection(frames, anchor_frame, anchor_pose, cancelled=lambda: False):
    anchor = frames[anchor_frame]
    target = anchor.poses[anchor_pose]
    if not target.trackable:
        raise ValueError("선택한 선수의 몸통 관절 신뢰도가 부족합니다.")
    selections = {anchor_frame: FrameSelection(anchor_frame, anchor_pose, "selected")}
    for direction in (1, -1):
        previous, previous_poses, previous_index = target, anchor.poses, anchor_pose
        stopped = crowded(target, anchor.poses, anchor_pose)
        index = anchor_frame + direction
        lower, upper = min(frames), max(frames)
        while lower <= index <= upper:
            if cancelled():
                return {}
            frame = frames.get(index)
            if stopped:
                selections[index] = FrameSelection(index, None, "reselect_required")
            elif frame is None:
                stopped = True
                selections[index] = FrameSelection(index, None, "not_analyzed")
            else:
                match, state = match_person(previous, previous_poses, previous_index, frame.poses)
                selections[index] = FrameSelection(index, match, state)
                if match is None:
                    stopped = True
                    log.warning("선수 추적 실패: frame=%s, state=%s, direction=%s; 재선택 필요", index, state, direction)
                else:
                    previous, previous_poses, previous_index = frame.poses[match], frame.poses, match
            index += direction
    return selections


class TrackingWorker(QThread):
    ready = Signal(object)
    error = Signal(str)

    def __init__(self, frames, anchor_frame, anchor_pose, parent=None):
        super().__init__(parent)
        self.frames = dict(frames)
        self.anchor_frame, self.anchor_pose = anchor_frame, anchor_pose

    def run(self):
        try:
            result = track_selection(self.frames, self.anchor_frame, self.anchor_pose, self.isInterruptionRequested)
            if not self.isInterruptionRequested():
                self.ready.emit(result)
        except Exception as exc:
            log.exception("선수 추적 계산 오류")
            self.error.emit(str(exc))
