# 규칙 기반 동작 분석의 설정, 이벤트/단계/관측 구간 및 편집 문서 데이터 계약.
# 실제 영상 PTS/프레임, 선택 선수의 문맥 키, 근거/불확실성과 원본/수정 결과를
# UI와 독립적으로 보존한다. 구질/임팩트/자세 품질을 확정하는 모델이 아니다.
# 경계·순서·신원·중복 유효성 검사를 자동 결과와 수동/JSON 입력에 공통 적용한다.
# 선택적 provenance는 관절 실행/캐시 지문을 저장하며 구버전 JSON은 빈 출처로 읽는다.
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math

ALGORITHM_VERSION = "overhead-rules-1.0"
KINDS = ("OVERHEAD_CANDIDATE", "OTHER_SWING", "UNKNOWN", "NO_SWING")
PHASES = ("READY", "BACKSWING", "FORWARD_SWING", "FOLLOW_THROUGH", "RECOVERY")
PHASE_LABELS = {"READY": "준비", "BACKSWING": "백스윙", "FORWARD_SWING": "전방 스윙",
                "FOLLOW_THROUGH": "팔로스루", "RECOVERY": "복귀", "UNKNOWN_PHASE": "단계 보류"}


@dataclass(frozen=True, slots=True)
class MotionConfig:
    sensitivity: float = 1.0
    min_duration: float = 0.18
    merge_gap: float = 0.12
    smoothing: str = "median"
    window: int = 3
    max_time_gap: float = 0.15
    speed_floor: float = 1.4  # 몸통 길이/초. 라켓 속도가 아니다.
    min_excursion: float = 0.35
    context_seconds: float = 0.30
    min_valid_ratio: float = 0.7

    def __post_init__(self):
        for name in ("sensitivity", "min_duration", "merge_gap", "max_time_gap", "speed_floor",
                     "min_excursion", "context_seconds", "min_valid_ratio"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f"유효하지 않은 동작 설정: {name}")
        if not (0.3 <= self.sensitivity <= 3 and 0.05 <= self.min_duration <= 3 and 0 <= self.merge_gap <= 0.5
                and 0.03 <= self.max_time_gap <= 1 and 0.1 <= self.speed_floor <= 20
                and 0.05 <= self.min_excursion <= 3 and 0 <= self.context_seconds <= 1
                and 0.5 <= self.min_valid_ratio <= 1):
            raise ValueError("동작 설정이 허용 범위를 벗어났습니다.")
        if self.smoothing not in ("median", "none") or type(self.window) is not int or self.window not in range(1, 16, 2):
            raise ValueError("평활화는 median/none, 창 크기는 1~15 사이의 홀수여야 합니다.")


@dataclass(frozen=True, slots=True)
class PhaseSpan:
    phase: str
    start_frame: int
    end_frame: int
    start_seconds: float
    end_seconds: float  # 마지막 프레임의 표시 종료 (exclusive time)
    start_pts: float | None
    end_pts: float | None  # 마지막 프레임 자체의 원본 PTS
    evidence: str


@dataclass(frozen=True, slots=True)
class MotionEvent:
    event_id: str
    video_id: str
    athlete_key: str
    player_id: int
    track_segment: str
    racket_hand: str
    kind: str
    start_frame: int
    end_frame: int
    start_seconds: float
    end_seconds: float
    start_pts: float | None
    end_pts: float | None
    duration: float
    phases: tuple[PhaseSpan, ...]
    peak_frame: int | None
    peak_seconds: float | None
    peak_pts: float | None
    peak_angles: dict
    indicators: dict
    details: dict
    valid_ratio: float | None
    evidence: tuple[str, ...]
    uncertainty: tuple[str, ...]
    source: str = "automatic"
    manually_modified: bool = False
    user_label: str = ""
    clipped: bool = False
    algorithm_version: str = ALGORITHM_VERSION


@dataclass(frozen=True, slots=True)
class MotionInterval:
    start_frame: int
    end_frame: int
    kind: str
    reason: str


@dataclass(frozen=True, slots=True)
class MotionResult:
    video_identity: tuple
    video_id: str
    athlete_key: str
    context_id: str
    player_id: int
    racket_hand: str
    scope: tuple[int, int]
    config: MotionConfig
    events: tuple[MotionEvent, ...]
    coverage: tuple[MotionInterval, ...]
    analysis_key: str
    source_anchor: dict
    algorithm_version: str = ALGORITHM_VERSION
    provenance: dict = field(default_factory=dict)


@dataclass(frozen=True)
class MotionDocument:
    automatic: MotionResult
    events: tuple[MotionEvent, ...]
    operations: tuple[dict, ...] = field(default_factory=tuple)

    @property
    def modified(self):
        return self.events != self.automatic.events


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()[:24]


def make_context(identity, anchor_frame, anchor_pose, hand, player_id):
    if hand not in ("left", "right"):
        raise ValueError("라켓 손을 먼저 선택하세요.")
    anchor = {"frame": anchor_frame, "torso": [[round(anchor_pose.joints[i].x, 3), round(anchor_pose.joints[i].y, 3)]
                                               for i in (11, 12, 23, 24)]}
    video_id = fingerprint(identity)
    athlete_key = fingerprint((video_id, anchor))
    return {"video_id": video_id, "athlete_key": athlete_key,
            "context_id": fingerprint((athlete_key, hand, ALGORITHM_VERSION)), "player_id": player_id, "source_anchor": anchor}


def phase_span(name, start, end, timeline, evidence):
    return PhaseSpan(name, start, end, timeline.seconds[start], timeline.frame_end(end),
                     timeline.raw_pts_seconds[start], timeline.raw_pts_seconds[end], evidence)


def normalize_phases(spans, start, end, timeline):
    result, cursor, last_rank = [], start, -1
    for span in sorted(spans, key=lambda p: p.start_frame):
        if span.phase not in (*PHASES, "UNKNOWN_PHASE"):
            raise ValueError("알 수 없는 동작 단계입니다.")
        if type(span.start_frame) is not int or type(span.end_frame) is not int or not start <= span.start_frame <= span.end_frame <= end:
            raise ValueError("단계 경계는 이벤트 프레임 안에 있어야 합니다.")
        if span.start_frame < cursor:
            raise ValueError("동작 단계가 서로 겹칩니다.")
        if span.phase != "UNKNOWN_PHASE":
            rank = PHASES.index(span.phase)
            if rank <= last_rank:
                raise ValueError("동작 단계의 순서 또는 중복을 확인하세요.")
            last_rank = rank
        if span.start_frame > cursor:
            result.append(phase_span("UNKNOWN_PHASE", cursor, span.start_frame - 1, timeline, "경계 근거 부족"))
        result.append(phase_span(span.phase, span.start_frame, span.end_frame, timeline, span.evidence))
        cursor = span.end_frame + 1
    if cursor <= end:
        result.append(phase_span("UNKNOWN_PHASE", cursor, end, timeline, "경계 근거 부족"))
    return tuple(result)


def validate_event(event, timeline, others=()):
    if (type(event.start_frame) is not int or type(event.end_frame) is not int
            or not 0 <= event.start_frame <= event.end_frame < len(timeline.seconds)):
        raise ValueError("이벤트 시작/종료 프레임 범위가 올바르지 않습니다.")
    if event.kind not in KINDS or event.racket_hand not in ("left", "right") or event.source not in ("automatic", "manual"):
        raise ValueError("동작 라벨/라켓 손/데이터 출처가 올바르지 않습니다.")
    if len(event.user_label) > 80 or any(ord(c) < 32 for c in event.user_label):
        raise ValueError("사용자 라벨은 줄바꿈 없이 80자 이하로 입력하세요.")
    if not event.event_id or not event.athlete_key or not event.track_segment:
        raise ValueError("이벤트/선수/추적 구간 식별자가 없습니다.")
    if type(event.player_id) is not int or event.algorithm_version != ALGORITHM_VERSION:
        raise ValueError("선택 세션 또는 알고리즘 버전이 올바르지 않습니다.")
    if event.valid_ratio is not None and (not math.isfinite(event.valid_ratio) or not 0 <= event.valid_ratio <= 1):
        raise ValueError("유효 데이터 비율이 올바르지 않습니다.")
    if (not math.isclose(event.start_seconds, timeline.seconds[event.start_frame], abs_tol=1e-7)
            or not math.isclose(event.end_seconds, timeline.frame_end(event.end_frame), abs_tol=1e-7)
            or not math.isclose(event.duration, event.end_seconds - event.start_seconds, abs_tol=1e-7) or event.duration <= 0):
        raise ValueError("이벤트 시간과 원본 프레임 시간축이 다릅니다.")
    if event.start_pts != timeline.raw_pts_seconds[event.start_frame] or event.end_pts != timeline.raw_pts_seconds[event.end_frame]:
        raise ValueError("이벤트 원본 PTS가 영상과 다릅니다.")
    normalized = normalize_phases(event.phases, event.start_frame, event.end_frame, timeline)
    if normalized != event.phases:
        raise ValueError("단계의 시간/연속성 또는 PTS를 확인하세요.")
    if event.peak_frame is not None and not event.start_frame <= event.peak_frame <= event.end_frame:
        raise ValueError("움직임 피크는 이벤트 내부에 있어야 합니다.")
    if event.peak_frame is not None:
        if event.peak_seconds != timeline.seconds[event.peak_frame] or event.peak_pts != timeline.raw_pts_seconds[event.peak_frame]:
            raise ValueError("피크 프레임과 PTS가 다릅니다.")
    elif event.peak_seconds is not None or event.peak_pts is not None:
        raise ValueError("피크 프레임 없이 피크 시간을 지정할 수 없습니다.")
    for other in others:
        if other.event_id != event.event_id and not (event.end_frame < other.start_frame or event.start_frame > other.end_frame):
            raise ValueError("다른 스윙 이벤트와 프레임 범위가 겹칩니다.")


def result_to_dict(result):
    return asdict(result)
