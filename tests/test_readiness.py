# 통합 분석 진단의 원인/해결 동작과 레거시 저장 선수 연결의 안전성을 검사한다.
# 다른 영상/선수/구간·추적 실패를 거부하고 수동 경계/단계/메모 식별자를 보존한다.
# 실행 가능 여부와 스윙 검토·기술 품질 판정을 독립적으로 검증한다. Qt 의존성 없음.
from dataclasses import replace
import unittest
from test_motion import synthetic_case, analyze
from app.analysis.analysis_readiness import diagnose, rebind_document, validate_link
from app.analysis.motion_models import MotionDocument


def complete_state():
    return dict(video=True, timeline=True, pose=True, asset_valid=True, selected=True,
                measurements=True, hand=True, events=True, linked=True, event_valid=True,
                reviewed=False, rules=True, busy=False, quality=None)


class ReadinessTests(unittest.TestCase):
    def test_missing_player_and_swing_are_distinct(self):
        state = complete_state()
        state.update(selected=False, event_valid=False)
        report = diagnose(state)
        self.assertFalse(report.can_run)
        self.assertTrue(any(item.code == "PLAYER_MISSING" and item.action == "player" and item.blocking for item in report.checks))
        self.assertTrue(any(item.code == "SWING_NOT_SELECTED" and item.action == "swing" and item.blocking for item in report.checks))

    def test_review_and_quality_not_implicitly_approved(self):
        report = diagnose(complete_state())
        self.assertTrue(report.can_run)
        self.assertEqual(dict(report.steps)["스윙 검토"], "사용자 확인 필요")
        self.assertIn("평가 보류", report.summary)
        self.assertTrue(any(i.code == "QUALITY_DEFERRED" and i.status != "완료" for i in report.checks))

    def test_scope_does_not_invalidate_existing_event(self):
        state = complete_state()
        state["changes"] = (("scope", "다음 검출 범위 변경"),)
        report = diagnose(state)
        self.assertTrue(report.can_run)
        self.assertEqual(report.cache_state, "재사용 가능")
        state["changes"] += (("hand", "라켓 손 변경"),)
        report = diagnose(state)
        self.assertFalse(report.can_run)
        self.assertEqual(report.cache_state, "재계산 필요")

    def test_no_selected_event_all_events_can_run(self):
        state = complete_state()
        state["event_valid"] = False
        report = diagnose(state)
        self.assertFalse(report.can_run)
        self.assertTrue(report.can_run_all)

    def test_legacy_link_checks_context_not_pose_order(self):
        case = synthetic_case()
        result, features = analyze(case)
        doc = MotionDocument(result, result.events)
        reasons, mapping = validate_link(doc, *case[:3], case[3], case[4])
        self.assertFalse(reasons)
        self.assertEqual(mapping[doc.events[0].event_id], "7:0")
        wrong = {**case[4], "context_id": "different_anchor"}
        self.assertTrue(validate_link(doc, *case[:3], case[3], wrong)[0])
        self.assertTrue(validate_link(doc, *case[:3], replace(case[3], identity=("foreign", 1, 1)), case[4])[0])

    def test_relink_updates_session_but_keeps_user_boundaries(self):
        case = synthetic_case()
        result, _ = analyze(case)
        event = replace(result.events[0], user_label="사용자 라벨", manually_modified=True, source="manual")
        doc = MotionDocument(result, (event,))
        for i, selected in list(case[1].items()):
            case[1][i] = replace(selected, player_id=19)
            case[2][i] = replace(case[2][i], player_id=19, track_segment="19:0")
        context = {**case[4], "player_id": 19}
        _, features = analyze(case)
        reasons, mapping = validate_link(doc, *case[:3], case[3], context)
        self.assertFalse(reasons)
        updated = rebind_document(doc, features, case[2], case[3], context, mapping, {"pose_run_id": "new"})
        current = updated.events[0]
        self.assertEqual(current.player_id, 19)
        self.assertEqual(current.track_segment, "19:0")
        self.assertEqual((current.event_id, current.start_frame, current.end_frame, current.phases, current.user_label),
                         (event.event_id, event.start_frame, event.end_frame, event.phases, event.user_label))
        self.assertTrue(current.manually_modified)
        self.assertEqual(doc.events[0].player_id, 7)

    def test_tracking_gap_or_foreign_player_cannot_link(self):
        for change in ("gap", "foreign"):
            case = synthetic_case()
            result, _ = analyze(case)
            case[1][72] = replace(case[1][72], state="lost" if change == "gap" else "tracked", player_id=99 if change == "foreign" else 7)
            self.assertTrue(validate_link(MotionDocument(result, result.events), *case[:3], case[3], case[4])[0])

    def test_failed_stage_is_not_shown_as_completed(self):
        state = complete_state()
        state.update(pose_error=True, coach_error="규칙 실행 실패", motion_error="파일 읽기 실패")
        result = diagnose(state)
        self.assertEqual(dict(result.steps)["관절"], "오류")
        self.assertEqual(dict(result.steps)["코칭"], "오류")
        self.assertEqual(dict(result.steps)["스윙 검출"], "오류")
        self.assertTrue(any(i.code == "COACHING_ERROR" for i in result.checks))


if __name__ == "__main__":
    unittest.main()
