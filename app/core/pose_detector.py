# MediaPipe Tasks VIDEO 검출 어댑터 및 불변 관절 데이터 모델.
# 33개 관절의 이름/정규화 좌표/깊이/visibility/presence/월드 좌표를 보존한다.
# 낮은 신뢰도 및 화면 밖 관절을 시각화/선택에서 제외하며 이전 값을 채우지 않는다.
# MediaPipe는 지연 import하여 모델/패키지 오류가 기존 플레이어를 막지 않는다.
# 와이드 영상의 작은 선수를 위해 전체/좌/우 겹침 영역의 독립 VIDEO 인스턴스를
# 사용하고 원본 좌표로 역변환한다. 중복 후보를 제거하고 최대 num_poses를 유지한다.
from dataclasses import dataclass
import logging
import math
from pathlib import Path

log = logging.getLogger(__name__)
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "pose_landmarker_full.task"
CONFIDENCE = 0.5
NUM_POSES = 4
DETECTION_VIEWS = "full+overlapping-horizontal-halves-v1"
LANDMARK_NAMES = (
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner",
    "right_eye", "right_eye_outer", "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist",
    "right_wrist", "left_pinky", "right_pinky", "left_index", "right_index",
    "left_thumb", "right_thumb", "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel", "left_foot_index", "right_foot_index",
)
CONNECTIONS = ((0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
               (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
               (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
               (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
               (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32))


@dataclass(frozen=True, slots=True)
class Joint:
    id: int
    name: str
    x: float
    y: float
    z: float
    visibility: float | None
    presence: float | None
    world: tuple[float, float, float] | None = None
    in_source_view: bool = True

    @property
    def reliable(self):
        return (self.in_source_view and all(math.isfinite(v) for v in (self.x, self.y, self.z))
                and 0 <= self.x <= 1 and 0 <= self.y <= 1
                and self.visibility is not None and CONFIDENCE <= self.visibility <= 1
                and self.presence is not None and CONFIDENCE <= self.presence <= 1)


@dataclass(frozen=True, slots=True)
class PersonPose:
    joints: tuple[Joint, ...]
    source_view: str = "full"

    @property
    def bbox(self):
        points = [joint for joint in self.joints if joint.reliable]
        if len(points) < 4:
            return None
        return (min(p.x for p in points), min(p.y for p in points),
                max(p.x for p in points), max(p.y for p in points))

    @property
    def trackable(self):
        # 몸통 네 관절을 모두 확인할 수 있어야 신원 연속성 비교를 허용한다.
        return len(self.joints) == 33 and all(self.joints[i].reliable for i in (11, 12, 23, 24))


@dataclass(frozen=True, slots=True)
class FrameResult:
    frame_index: int
    timestamp_ms: int
    poses: tuple[PersonPose, ...]
    detection_success: bool
    status: str
    timestamp_source: str = "frame_index/fps"


def convert_result(result, width, x_offset, crop_width, source_view):
    poses = []
    for pose_index, landmarks in enumerate(result.pose_landmarks):
        if len(landmarks) != 33:
            continue
        world = result.pose_world_landmarks[pose_index] if pose_index < len(result.pose_world_landmarks) else ()
        joints = tuple(Joint(i, LANDMARK_NAMES[i], (x_offset + p.x * crop_width) / width,
                             p.y, p.z * crop_width / width, p.visibility, p.presence,
                             (world[i].x, world[i].y, world[i].z) if i < len(world) else None,
                             0 <= p.x <= 1 and 0 <= p.y <= 1)
                       for i, p in enumerate(landmarks))
        poses.append(PersonPose(joints, source_view))
    return poses


def merge_poses(poses, limit):
    def quality(pose):
        return (pose.trackable, sum(p.reliable for p in pose.joints), pose.source_view == "full")

    def duplicate(a, b):
        common = [i for i in (11, 12, 23, 24) if a.joints[i].reliable and b.joints[i].reliable]
        if len(common) < 3:
            return False
        scale = max(math.hypot(p.joints[11].x - p.joints[23].x,
                               p.joints[11].y - p.joints[23].y) for p in (a, b))
        delta = sum(math.hypot(a.joints[i].x - b.joints[i].x,
                               a.joints[i].y - b.joints[i].y) for i in common) / len(common)
        return delta < max(0.01, scale * 0.3)

    merged = []
    for pose in sorted(poses, key=quality, reverse=True):
        if not any(duplicate(pose, accepted) for accepted in merged):
            merged.append(pose)
        if len(merged) == limit:
            break
    return tuple(merged)


class PoseDetector:
    def __init__(self, model_path=MODEL_PATH, num_poses=NUM_POSES):
        self.landmarker = None
        self.landmarkers = []
        self.num_poses = num_poses
        self.last_timestamp = -1
        log.info("AI 모델 로딩 시작: %s, num_poses=%s", model_path, num_poses)
        try:
            if not Path(model_path).is_file():
                raise FileNotFoundError(f"AI 모델이 없습니다: {model_path}. models/README.md를 확인하세요.")
            import mediapipe as mp
            self.mp = mp
            options = mp.tasks.vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_poses=num_poses,
                min_pose_detection_confidence=CONFIDENCE,
                min_pose_presence_confidence=CONFIDENCE,
                min_tracking_confidence=CONFIDENCE,
                output_segmentation_masks=False,
            )
            for _ in range(3):
                self.landmarkers.append(mp.tasks.vision.PoseLandmarker.create_from_options(options))
            self.landmarker = self.landmarkers[0]
            log.info("AI 모델 로딩 성공: MediaPipe %s, views=%s", mp.__version__, DETECTION_VIEWS)
        except Exception:
            log.exception("AI 모델 로딩 실패")
            try:
                self.close()
            except Exception:
                log.exception("모델 로딩 실패 후 자원 해제 오류")
            raise

    def detect(self, rgb, frame_index, timestamp_ms):
        if timestamp_ms <= self.last_timestamp:
            raise ValueError("VIDEO 모드 타임스탬프는 단조 증가해야 합니다.")
        import numpy as np
        width = rgb.shape[1]
        views = (("full", 0, width), ("left", 0, max(1, round(width * 0.6))),
                 ("right", min(width - 1, round(width * 0.4)), width))
        poses = []
        for landmarker, (name, start, end) in zip(self.landmarkers, views):
            image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb[:, start:end]))
            result = landmarker.detect_for_video(image, timestamp_ms)
            poses.extend(convert_result(result, width, start, end - start, name))
        self.last_timestamp = timestamp_ms
        poses = merge_poses(poses, self.num_poses)
        success = any(p.trackable for p in poses)
        status = "detected" if success else "low_confidence" if poses else "not_detected"
        return FrameResult(frame_index, timestamp_ms, tuple(poses), success, status)

    def close(self):
        errors = []
        for landmarker in self.landmarkers:
            try:
                landmarker.close()
            except Exception as exc:
                log.exception("VIDEO 모델 인스턴스 종료 실패")
                errors.append(exc)
        self.landmarkers.clear()
        self.landmarker = None
        if errors:
            raise RuntimeError("AI 모델 인스턴스 종료 실패") from errors[0]
