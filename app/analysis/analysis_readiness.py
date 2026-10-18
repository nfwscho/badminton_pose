# 분석 준비 상태와 저장 선수 연결을 검사하는 Qt 독립 진단 모듈.
# 원인 코드/한글 이유/해결 동작을 분리하고 실행 가능성과 기술 평가 품질을 구분한다.
# 저장 선수는 영상·선택 기준·현재 추적 연속성을 검증한 뒤에도 사용자 확인을 요구한다.
# 연결 갱신은 수동 경계/단계/라벨/이벤트 ID를 보존하고 현재 캐시로 수치만 다시 계산한다.
from dataclasses import dataclass, replace
from app.analysis.coach_models import now
from app.analysis.motion_models import MotionDocument, validate_event


@dataclass(frozen=True)
class DiagnosticItem:
    code: str
    title: str
    status: str
    reason: str
    action: str
    action_label: str
    blocking: bool = False


@dataclass(frozen=True)
class Readiness:
    checks: tuple[DiagnosticItem, ...]
    steps: tuple[tuple[str, str], ...]
    cache_state: str
    can_run: bool
    can_run_all: bool
    summary: str


def diagnose(state):
    checks = []
    def check(ok, code, title, reason, action, label, status="사용자 확인 필요", blocking=True, success="현재 입력과 일치합니다."):
        checks.append(DiagnosticItem(code, title, "완료" if ok else status, success if ok else reason,
                                     "" if ok else action, "" if ok else label, blocking and not ok))
    check(state["video"], "VIDEO_MISSING", "영상 불러오기", "분석할 MP4가 없습니다.", "open", "영상 불러오기", "분석 불가")
    check(state["timeline"], "TIMELINE_PENDING", "원본 시간축", "원본 PTS 확인 중이거나 시간축을 만들지 못했습니다.",
          "open" if state.get("timeline_error") else "details", "영상 다시 불러오기" if state.get("timeline_error") else "시간축 상태 보기",
          "오류" if state.get("timeline_error") else "진행 중" if state["video"] else "분석 불가")
    check(state["pose"], "POSE_MISSING", "관절 분석", state.get("pose_reason", "현재 영상의 관절 분석이 필요합니다."),
          "pose", "관절 분석 시작", "진행 중" if state.get("pose_busy") else "재분석 필요",
          success=f"{state.get('pose_count', 0)}프레임 관절 캐시가 있습니다.")
    check(state["asset_valid"], "POSE_SOURCE_CHANGED", "영상·관절 캐시 일치", state.get("asset_reason", "원본 일치 재검증이 필요합니다."),
          state.get("asset_action", "check"), "영상 다시 불러오기" if state.get("asset_action") == "open" else "관절 분석 갱신" if state.get("asset_action") == "pose" else "캐시 재검증", "재분석 필요")
    check(state["selected"], "PLAYER_MISSING", "분석 선수 선택", "분석할 선수가 선택되지 않았습니다. 영상의 선수를 직접 지정하세요.", "player", "선수 선택")
    check(state["measurements"], "MEASUREMENTS_PENDING", "선수 추적·관절 측정", "선택 선수의 추적/측정 완료를 기다리거나 선수를 다시 선택하세요.",
          "player", "선수 다시 선택", "진행 중" if state.get("measurement_busy") else "분석 불가",
          success=f"측정 가능 {state.get('measured_count', 0)}프레임. 스윙별 품질은 별도 검사합니다.")
    check(state["hand"], "HAND_MISSING", "라켓 손 설정", "오른손잡이/왼손잡이를 선택하세요.", "hand", "라켓 손 선택")
    check(state["events"], "SWINGS_MISSING", "스윙 검출", "스윙 후보가 없습니다. 검출 범위와 선수 추적 상태를 확인하세요.", "detect", "스윙 검출", "진행 중" if state.get("motion_busy") else "분석 불가")
    cannot_link = state.get("context_mismatch", False) or state.get("link_rejected", False)
    check(state["linked"], "PLAYER_LINK_REQUIRED", "저장 선수 연결", state.get("link_reason", "저장 스윙과 현재 선수의 연결이 확인되지 않았습니다."),
          "detect" if cannot_link else "link", "현재 선수로 새 스윙 검출" if cannot_link else "선수 연결 확인", "재검증 필요")
    changes = state.get("changes", ())
    blocking_changes = [reason for code, reason in changes if code != "scope"]
    check(not blocking_changes, "MOTION_INPUT_CHANGED", "스윙 입력·설정", "; ".join(blocking_changes), "refresh", "영향받는 분석 갱신", "재계산 필요",
          success="다음 검출의 A/B 범위만 변경됨 — 기존 스윙과 코칭 캐시는 재사용 가능" if changes else "현재 스윙 설정과 일치합니다.")
    check(state["event_valid"], "SWING_NOT_SELECTED", "코칭 대상 스윙", state.get("event_reason", "목록에서 코칭할 스윙을 선택하세요."), "swing", "스윙 목록 보기")
    check(state["reviewed"], "SWING_REVIEW_REQUIRED", "스윙·단계 검토", "후보 영상을 보고 경계·단계를 직접 검토하세요. 미검토 상태에서도 보류 사유 분석은 가능합니다.",
          "review", "해당 스윙 보기·검토", blocking=False)
    check(state["rules"], "RULES_INVALID", "코칭 규칙 설정", "규칙 파일을 읽을 수 없거나 설정이 유효하지 않습니다.", "rules", "규칙 설정 불러오기", "오류")
    check(state.get("storage_ready", True), "COACH_STORE_ERROR", "코칭 기록 호환성", "코칭 기록을 현재 영상에 연결하지 못했습니다. 저장 파일 오류를 확인하고 다시 검사하세요.",
          "check", "저장 상태 다시 확인", "오류")
    if state.get("coach_error"):
        checks.append(DiagnosticItem("COACHING_ERROR", "최근 코칭 실행", "오류",
                                    "최근 코칭 실행을 완료하지 못했습니다. 원인: " + state["coach_error"], "coach", "입력 확인 후 코칭 다시 실행", False))
    if state.get("motion_error"):
        checks.append(DiagnosticItem("MOTION_ERROR", "최근 스윙 검출", "오류",
                                    "최근 스윙 검출을 완료하지 못했습니다. 원인: " + state["motion_error"], "detect", "스윙 검출 다시 실행", False))
    quality = state.get("quality")
    check(quality is not None and quality.state == "MEASUREMENT_ONLY", "QUALITY_DEFERRED", "코칭 근거 품질", "; ".join(quality.reasons) if quality else
          "스윙별 유효 관절·단계·규칙을 아직 검사하지 않았습니다. 분석 상태 확인을 누르세요.",
          "details", "부족한 데이터·규칙 보기", "분석 불가" if quality else "재검증 필요", blocking=False,
          success="관찰 규칙 적용 가능. 검증된 표준 기술 평가는 제공하지 않습니다.")
    check(not state.get("busy"), "ANALYSIS_BUSY", "백그라운드 작업", "분석 작업이 진행 중입니다. 완료 후 자동 갱신합니다.", "details", "진행 상태 보기", "진행 중")
    all_blocked = any(item.blocking for item in checks if item.code != "SWING_NOT_SELECTED")
    blocked = [item for item in checks if item.blocking]
    can_run, can_run_all = not blocked, not all_blocked
    report_current = state.get("report_current", False)
    steps = (("영상", checks[0].status), ("관절", "오류" if state.get("pose_error") else checks[2].status), ("선수", checks[4].status),
             ("스윙 검출", "오류" if state.get("motion_error") else checks[7].status if not state["events"] else "재분석 필요" if blocking_changes else "사용자 확인 필요" if not state["linked"] else "완료"),
             ("스윙 검토", "완료" if state["reviewed"] else "사용자 확인 필요"),
             ("코칭", "진행 중" if state.get("coach_busy") else "오류" if state.get("coach_error") else "완료" if report_current else "재분석 필요" if state.get("has_report") else "사용자 확인 필요"),
             ("결과·저장", "완료" if report_current and state.get("saved") else "오류" if state.get("save_error") else "사용자 확인 필요"))
    cache_state = "현재 결과와 호환되지 않음" if not state["asset_valid"] or (state.get("context_mismatch", False)) else "재검증 필요" if not state["linked"] else "재계산 필요" if blocking_changes or (state.get("has_report") and not report_current) else "재사용 가능"
    summary = ("코칭 분석 실행 가능" + (" — 관찰/평가 보류 결과를 제공합니다." if not state["reviewed"] or not quality or quality.state != "MEASUREMENT_ONLY" else " — 측정 관찰만 가능")) if can_run else blocked[0].reason
    if report_current:
        summary = "코칭 결과 생성 완료" + (" · 평가 보류: 근거/단계 부족" if quality and quality.state == "COACHING_DEFERRED" else " · 측정 관찰")
    return Readiness(tuple(checks), steps, cache_state, can_run, can_run_all, summary)


def validate_link(document, frames, selections, measurements, timeline, context):
    """후보 순번 대신 동일 영상/기준 몸통/실제 프레임별 추적을 검사한다. 승인 자체는 UI 책임."""
    reasons, mapping = [], {}
    source = document.automatic
    if (tuple(timeline.identity) != source.video_identity or context is None or
            context["context_id"] != source.context_id or context["source_anchor"] != source.source_anchor):
        return ("영상·라켓 손·선택 기준 프레임/몸통 위치가 다릅니다. 원래 기준 프레임에서 선수를 다시 선택하거나 새로 검출하세요.",), {}
    for event in (*source.events, *document.events):
        try:
            validate_event(event, timeline)
        except ValueError as exc:
            reasons.append(str(exc))
            continue
        if event.player_id != source.player_id or event.athlete_key != source.athlete_key:
            reasons.append("저장 이벤트에 다른 선수 식별 정보가 포함되어 있습니다.")
        segments = set()
        for i in range(event.start_frame, event.end_frame + 1):
            s, m, frame = selections.get(i), measurements.get(i), frames.get(i)
            if (s is None or m is None or frame is None or s.player_id != context["player_id"] or m.player_id != context["player_id"]
                    or s.state not in ("selected", "tracked") or m.tracking_state not in ("selected", "tracked")
                    or s.selected_pose_index is None or not 0 <= s.selected_pose_index < len(frame.poses)
                    or not frame.poses[s.selected_pose_index].trackable or s.frame_index != i or m.frame_index != i
                    or abs(m.timestamp_seconds - timeline.seconds[i]) > 1e-6 or not m.track_segment
                    or timeline.sources[i] != "pts" or (i > event.start_frame and
                    not 1e-6 < timeline.seconds[i] - timeline.seconds[i - 1] <= source.config.max_time_gap)):
                reasons.append(f"프레임 {i}: 현재 선택 선수의 연속 추적을 확인할 수 없습니다. 다른 선수로 보충하지 않습니다.")
                break
            segments.add(m.track_segment)
        if len(segments) != 1:
            reasons.append(f"스윙 {event.start_frame}–{event.end_frame}: 하나의 연속 추적 구간이 아닙니다.")
        elif not reasons:
            mapping[event.event_id] = next(iter(segments))
    return tuple(dict.fromkeys(reasons)), mapping


def rebind_document(document, features, measurements, timeline, context, mapping, provenance):
    from app.analysis.motion_detector import manual_event
    source = document.automatic
    temporary = MotionDocument(replace(source, player_id=context["player_id"], events=()), ())
    def refresh(event):
        updated = manual_event(temporary, timeline, event.start_frame, event.end_frame, event.kind,
                               event.phases, event.user_label, event, features, measurements)
        return replace(updated, player_id=context["player_id"], track_segment=mapping[event.event_id],
                       source=event.source, manually_modified=event.manually_modified, evidence=event.evidence,
                       uncertainty=event.uncertainty, clipped=event.clipped)
    baseline = tuple(refresh(event) for event in source.events)
    current = tuple(refresh(event) for event in document.events)
    operation = {"action": "confirm_player_link", "at": now(), "previous_player_id": source.player_id,
                 "player_id": context["player_id"], "previous_provenance": source.provenance}
    from app.analysis.motion_models import fingerprint
    return MotionDocument(replace(source, player_id=context["player_id"], events=baseline, provenance=provenance,
                                  analysis_key=fingerprint((source.analysis_key, provenance))),
                          current, (*document.operations, operation))
