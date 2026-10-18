# 5.1 통합 상태/다음 단계/선수 연결/빈 패널/해상도와 정상 코칭 도달 경로 검증.
# 실제 임시 MP4와 기존 합성 캐시를 사용하며 실제 모델 정상 흐름은 별도 native smoke로 검증한다.
# 사용자 승인 취소, 다른 선수 거부, 수동 자료/메모 보호, 캐시 재사용과 원본 변경도 확인한다.
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from dataclasses import replace
from unittest.mock import patch
import unittest
from PySide6.QtCore import QPoint, QTimer
import test_motion_ui as helpers
from app.analysis.coach_models import event_key


class ReadinessUiTests(unittest.TestCase):
    setUpClass = classmethod(helpers.MotionUiTests.setUpClass.__func__)
    wait_for = helpers.MotionUiTests.wait_for
    run_motion = helpers.MotionUiTests.run_motion

    def setUp(self):
        helpers.MotionUiTests.setUp(self)

    def tearDown(self):
        helpers.MotionUiTests.tearDown(self)

    def codes(self):
        self.w.workflow.refresh(force=True)
        return {i.code for i in self.w.workflow.readiness.checks if i.blocking}

    def test_loaded_pose_and_missing_selection_next_action(self):
        self.assertNotIn("POSE_MISSING", self.codes())
        self.w.clear_selection()
        self.assertIn("PLAYER_MISSING", self.codes())
        self.assertTrue(self.w.workflow.conditions_button.isEnabled())

    def test_normal_flow_motion_review_ready_coaching_results(self):
        self.run_motion()
        self.assertIsNotNone(self.w.coach.selected_event())
        self.w.workflow.refresh(force=True)
        self.assertTrue(self.w.workflow.readiness.can_run)
        self.assertEqual(self.w.workflow.next_action()[0], "review")
        self.assertTrue(self.w.coach.analyze_button.isEnabled())
        with patch("app.core.pose_detector.PoseDetector.detect", side_effect=AssertionError("재추론 금지")):
            self.w.workflow.review_button.click()
            self.wait_for(lambda: not self.w.workflow.workers)
            self.assertTrue(self.w.workflow.state()["reviewed"])
            self.assertEqual(self.w.workflow.next_action()[0], "coach")
            self.w.workflow.advance()
            self.wait_for(lambda: not self.w.coach.workers)
        self.assertIsNotNone(self.w.coach.panel.report)
        self.assertEqual(dict(self.w.workflow.readiness.steps)["코칭"], "완료")

    def test_no_swing_selected_has_reason_and_always_available_check(self):
        self.run_motion()
        self.w.motion.panel.selected_id = None
        self.w.motion.refresh()
        self.assertFalse(self.w.coach.analyze_button.isEnabled())
        self.assertIn("SWING_NOT_SELECTED", self.codes())
        self.assertTrue(self.w.workflow.conditions_button.isEnabled())
        self.w.workflow.perform("swing")
        self.assertTrue(self.w.coach.analyze_button.isEnabled())

    def test_ab_selection_reuses_valid_coaching_cache(self):
        self.run_motion()
        self.w.coach.run()
        self.wait_for(lambda: not self.w.coach.workers)
        before = self.w.coach.panel.report
        self.w.motion.use_event_range()
        self.assertTrue(self.w.motion.stale)  # 기존 상태 API는 유지한다.
        self.assertTrue(self.w.coach.analyze_button.isEnabled())
        self.assertNotIn("재검출 필요", self.w.motion.panel.status_label.text())
        self.assertFalse(self.w.coach.is_stale(before))
        with patch("app.analysis.coach_engine.extract_features", side_effect=AssertionError("유효 캐시 재계산 금지")):
            self.w.coach.run()
            self.wait_for(lambda: not self.w.coach.workers)
        self.assertIs(self.w.coach.panel.report, before)

    def test_legacy_historical_link_requires_explicit_approval(self):
        self.run_motion()
        motion, flow = self.w.motion, self.w.workflow
        motion.load_document()
        self.assertTrue(motion.historical)
        self.assertFalse(self.w.coach.analyze_button.isEnabled())
        before = motion.document
        with patch.object(flow, "confirm_link", return_value=False) as confirmation:
            flow.start_check(True)
            self.wait_for(lambda: not flow.workers)
        confirmation.assert_called_once()
        self.assertIs(motion.document, before)
        self.assertTrue(motion.historical)
        with patch.object(flow, "confirm_link", return_value=True):
            flow.start_check(True)
            self.wait_for(lambda: not flow.workers)
        self.assertFalse(motion.historical)
        self.assertTrue(self.w.coach.analyze_button.isEnabled())
        self.assertFalse(flow.state()["reviewed"])

    def test_wrong_player_cannot_be_approved(self):
        self.run_motion()
        motion, flow = self.w.motion, self.w.workflow
        motion.load_document()
        self.w.cache.frames[0] = replace(self.w.cache.frames[0], poses=(replace(self.w.cache.frames[0].poses[0],
            joints=tuple(replace(j, x=j.x + 0.1) for j in self.w.cache.frames[0].poses[0].joints)),))
        with patch.object(flow, "confirm_link", return_value=True) as confirmation:
            flow.start_check(True)
            self.wait_for(lambda: not flow.workers)
        confirmation.assert_not_called()
        self.assertTrue(motion.historical)
        self.assertIn("연결 불가", flow.last_error)
        self.assertEqual(flow.next_action()[0], "detect")

    def test_manual_data_and_notes_preserved_during_link(self):
        self.run_motion()
        m, c, flow = self.w.motion, self.w.coach, self.w.workflow
        e = m.document.events[0]
        m.apply_manual(replace(e, user_label="보존할 경계", source="manual", manually_modified=True))
        c.run()
        self.wait_for(lambda: not c.workers)
        message = c.panel.report.messages[0]
        c.apply_review(message, "deferred", "유지할 메모")
        before = m.document.events[0]
        m.load_document()
        with patch.object(flow, "confirm_link", return_value=True):
            flow.start_check(True)
            self.wait_for(lambda: not flow.workers)
        after = m.document.events[0]
        self.assertEqual((after.event_id, after.start_frame, after.end_frame, after.phases, after.user_label),
                         (before.event_id, before.start_frame, before.end_frame, before.phases, before.user_label))
        self.assertTrue(any(review.note == "유지할 메모" for review in c.reviews.values()))

    def test_setting_changes_show_specific_cause_and_protect_manual(self):
        self.run_motion()
        m = self.w.motion
        e = m.document.events[0]
        m.apply_manual(replace(e, source="manual", manually_modified=True))
        m.config = replace(m.config, sensitivity=1.2)
        m.invalidate()
        self.assertIn("MOTION_INPUT_CHANGED", self.codes())
        self.assertIn("민감도", self.w.workflow.summary_label.text())
        with patch.object(m, "confirm_replace", return_value=False):
            self.w.workflow.perform("refresh")
        self.assertTrue(m.document.events[0].manually_modified)

    def test_background_diagnostic_response_rule_details_and_model_mismatch(self):
        self.run_motion()
        flow = self.w.workflow
        ticks = []
        timer = QTimer()
        timer.timeout.connect(lambda: ticks.append(1))
        timer.start(5)
        flow.inspect()
        self.wait_for(lambda: not flow.workers)
        timer.stop()
        self.assertTrue(ticks)
        self.assertTrue(flow.previews)
        self.assertTrue(any(flow.tree.topLevelItem(i).text(0).startswith("규칙:") for i in range(flow.tree.topLevelItemCount())))
        with patch("app.core.readiness_worker.cache_key", return_value=("changed",)):
            flow.start_check()
            self.wait_for(lambda: not flow.workers)
        self.assertFalse(flow.asset_valid)
        self.assertFalse(self.w.coach.available())
        self.assertEqual(next(i.action for i in flow.readiness.checks if i.code == "POSE_SOURCE_CHANGED"), "pose")

    def test_empty_panel_compact_and_resolution_access(self):
        c = self.w.coach
        c.panel_check.setChecked(True)
        self.assertTrue(c.panel.tabs.isHidden())
        self.assertLess(c.panel.minimumHeight(), 100)
        self.w.resize(1280, 720)
        self.w.show()
        self.app.processEvents()
        self.assertLessEqual(self.w.width(), 1280)
        self.assertLessEqual(self.w.height(), 720)
        self.assertTrue(self.w.workflow.conditions_button.isVisible())
        self.run_motion()
        c.run()
        self.wait_for(lambda: not c.workers)
        self.assertFalse(c.panel.tabs.isHidden())
        for width, height in ((1280, 720), (1920, 1080)):
            self.w.resize(width, height)
            self.app.processEvents()
            self.assertLessEqual(self.w.width(), width)
            self.assertLessEqual(self.w.height(), height)
            for button in (self.w.open_button, self.w.workflow.conditions_button, self.w.coach.analyze_button,
                           self.w.play, self.w.previous, self.w.next, self.w.slider):
                point = button.mapTo(self.w, QPoint(0, 0))
                self.assertGreaterEqual(point.x(), 0)
                self.assertGreaterEqual(point.y(), 0)
                self.assertLessEqual(point.x() + button.width(), width)
                self.assertLessEqual(point.y() + button.height(), height)
        sizes = self.w.content_splitter.sizes()
        sizes[-1] = 300
        self.w.content_splitter.setSizes(sizes)
        saved = self.w.content_splitter.sizes()[-1]
        c.panel_check.setChecked(False)
        c.panel_check.setChecked(True)
        self.assertEqual(self.w.content_splitter.sizes()[-1], saved)


if __name__ == "__main__":
    unittest.main()
