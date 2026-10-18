# 로컬 코칭 엔진의 관측값, 품질, 규칙 결과와 스윙 보고서 데이터 계약.
# 관측/파생/잠정 해석/판단 불가를 분리하고 프레임·원본 PTS·출처 버전을 보존한다.
# 사용자 검토는 생성 결과와 분리하여 재분석 후에도 메모를 유지하되 재확인을 요구한다.
# 확률, 실력 점수, 표준 자세 또는 임팩트 판정을 표현하지 않는다. Qt 의존성 없음.
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from app.analysis.motion_models import PHASE_LABELS, fingerprint

ENGINE_VERSION = "local-observation-1.0"
MEASUREMENT_VERSION = "joint_angles-pts-1"
LEVELS = ("MEASURED", "DERIVED", "INFERRED", "UNDETERMINED")
REVIEW_STATES = ("unreviewed", "confirmed", "deferred", "error")


def feature_label(name):
    phase, _, metric = name.partition(".")
    exact = {"duration": "지속 시간", "valid_ratio": "유효 관측 비율", "wrist_path": "손목 상대 경로 길이",
             "wrist_peak_time": "손목 속도 피크 시각", "elbow_peak_time": "팔꿈치 각속도 피크 시각",
             "shoulder_peak_time": "어깨 각속도 피크 시각", "peak_time_difference": "손목−팔꿈치 피크 시간 차이",
             "next_ready_time": "다음 준비 자세까지 시간"}
    label = exact.get(metric)
    if label is None:
        for prefix, description in (("wrist_speed", "손목 상대 속도"), ("elbow_rate", "팔꿈치 각속도"),
                ("shoulder_rate", "어깨 각속도"), ("trunk_rate", "몸통 각속도"), ("elbow", "팔꿈치 각도"),
                ("shoulder", "어깨 각도"), ("trunk", "몸통 기울기"), ("left_knee", "왼쪽 무릎 내각"),
                ("right_knee", "오른쪽 무릎 내각"), ("wrist_height", "손목 상대 높이")):
            if metric.startswith(prefix + "_"):
                suffix = metric[len(prefix) + 1:]
                label = description + " " + {"start": "시작값", "change": "끝−시작 변화", "range": "변화폭",
                                                "min": "최소", "max": "최대", "peak": "절댓값 피크"}.get(suffix, suffix)
                break
    return ("스윙 전체" if phase == "SWING" else PHASE_LABELS.get(phase, phase) + " 후보") + " / " + (label or metric)


def now():
    return datetime.now(timezone.utc).isoformat()


def event_key(event):
    return fingerprint(asdict(event))


@dataclass(frozen=True)
class Observation:
    name: str
    value: float | None
    unit: str
    valid: bool
    reason: str
    level: str
    start_frame: int
    end_frame: int
    peak_frame: int | None = None


@dataclass(frozen=True)
class CoachQuality:
    state: str
    detection: str
    valid_frames: int
    total_frames: int
    valid_ratio: float
    phase: str
    applicability: str
    user_confirmed: bool
    reasons: tuple[str, ...]
    integrity_ok: bool


@dataclass(frozen=True)
class CoachMessage:
    result_id: str
    video_id: str
    athlete_key: str
    player_id: int
    track_segment: str
    event_id: str
    phases: tuple[str, ...]
    rule_id: str
    rule_version: str
    level: str
    status: str
    measurements: tuple[Observation, ...]
    start_frame: int
    end_frame: int
    start_pts: float | None
    end_pts: float | None
    observation: str
    interpretation: str
    suggestion: str
    quality: CoachQuality
    deferred_reasons: tuple[str, ...]
    graph_metrics: tuple[str, ...]
    generated_at: str
    source: str = "automatic"
    rule_name: str = ""


@dataclass(frozen=True)
class CoachReport:
    video_identity: tuple
    video_id: str
    context_id: str
    event_id: str
    event_fingerprint: str
    input_key: str
    rules_key: str
    rules_version: str
    engine_version: str
    measurement_version: str
    motion_version: str
    generated_at: str
    quality: CoachQuality
    messages: tuple[CoachMessage, ...]
    features: dict[str, Observation]
    phase_status: dict[str, str]
    comparison: dict
    event_snapshot: dict


@dataclass(frozen=True)
class CoachReview:
    status: str = "unreviewed"
    note: str = ""
    error_reason: str = ""
    hidden: bool = False
    reviewed_input_key: str = ""
    updated_at: str = ""
    history: tuple[dict, ...] = field(default_factory=tuple)


def review_key(event_id, rule_id):
    return event_id + ":" + rule_id
