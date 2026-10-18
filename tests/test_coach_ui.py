# 실제 임시 MP4와 합성 관절 캐시를 사용하는 코칭 Qt 통합 검증.
# 기존 스윙 UI 준비 절차를 재사용하고 비동기/근거 PTS/A-B 격리/수동 무효화,
# 검토 적용·취소·숨김·재로딩·규칙 변경·취소/오류 후 복구를 검사한다.
# 합성 관찰 메시지 검증이며 실제 배드민턴 기술 지도 정확도를 뜻하지 않는다.
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from dataclasses import replace
import json
from pathlib import Path
import time
import unittest
from unittest.mock import patch

from PySide6.QtCore import QTimer, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QDialog, QMessageBox
import test_motion_ui as ui_helpers
from test_motion import synthetic_case
from app.analysis.coach_models import CoachReview, review_key
from app.analysis.coach_rules import DEFAULT_RULES
from app.ui.coach_panel import ReviewDialog


class CoachUiTests(unittest.TestCase):
    setUpClass = classmethod(ui_helpers.MotionUiTests.setUpClass.__func__)
    wait_for = ui_helpers.MotionUiTests.wait_for
    run_motion = ui_helpers.MotionUiTests.run_motion

    def setUp(self):
        ui_helpers.MotionUiTests.setUp(self)
        self.coach_errors = []
        self.w.coach.show_error = self.coach_errors.append
        self.run_motion()
        self.w.motion.select_event(self.w.motion.document.events[0].event_id)

    def tearDown(self):
        ui_helpers.MotionUiTests.tearDown(self)

    def run_coach(self, all_events=False):
        self.w.coach.run(all_events)
        self.wait_for(lambda: not self.w.coach.workers)
        self.assertFalse(self.coach_errors)
        event = self.w.motion.panel.selected_event()
        self.assertIn(event.event_id, self.w.coach.reports)
        return self.w.coach.reports[event.event_id]

    def test_async_no_inference_default_three_and_observation(self):
        from app.core.coach_worker import analyze_coaching
        ticks = []
        timer = QTimer()
        timer.timeout.connect(lambda: ticks.append(1))
        timer.start(5)
        def delayed(*args, **kwargs):
            time.sleep(0.06)
            return analyze_coaching(*args, **kwargs)
        try:
            with patch("app.core.coach_worker.analyze_coaching", side_effect=delayed), patch("app.core.pose_detector.PoseDetector.detect", side_effect=AssertionError("재추론 금지")):
                report = self.run_coach()
        finally:
            timer.stop()
        self.assertGreater(len(ticks), 3)
        self.assertEqual(report.quality.state, "COACHING_DEFERRED")
        self.assertLessEqual(self.w.coach.panel.list.count(), 3)
        self.assertTrue(all(self.w.coach.panel.list.item(i).data(Qt.ItemDataRole.UserRole).rule_id == "quality_notice"
                            or any(o.valid for o in self.w.coach.panel.list.item(i).data(Qt.ItemDataRole.UserRole).measurements)
                            for i in range(self.w.coach.panel.list.count())))
        self.w.coach.panel.more_check.setChecked(True)
        self.assertEqual(self.w.coach.panel.list.count(), 7)

    def test_confirmation_changes_quality_without_standard_claim(self):
        first = self.run_coach()
        c = self.w.coach
        c.panel.confirm_check.setChecked(True)
        self.assertTrue(c.is_stale(first))
        report = self.run_coach()
        self.assertEqual(report.quality.state, "MEASUREMENT_ONLY")
        self.assertIn("측정 관찰만 가능", c.panel.status.text())
        self.assertFalse(any(m.status == "COACHING_AVAILABLE" for m in report.messages))

    def test_evidence_seek_graph_pts_ab_and_replay(self):
        report = self.run_coach()
        c = self.w.coach
        message = next(m for m in report.messages if m.rule_id == "forward_extension")
        self.w.range_a, self.w.range_b = 0, 250
        self.w.refresh_measurements()
        before = self.w.measurement_panel.bounds
        c.show_evidence(message)
        self.wait_for(lambda: self.w.index == message.start_frame)
        self.assertEqual(self.w.measurement_panel.display_bounds, (message.start_frame, message.end_frame))
        self.assertEqual(self.w.measurement_panel.bounds, before)
        self.assertEqual((self.w.range_a, self.w.range_b), (0, 250))
        self.assertEqual(len(c.markers), 2)
        self.assertEqual(float(self.w.measurement_panel.cursor.value()), self.w.timeline.seconds[message.start_frame])
        with patch.object(self.w.player, "request") as request:
            c.panel.loop_check.setChecked(True)
            c.play_evidence()
            self.assertTrue(any(call.args[0] == "play_range" and call.args[1][2] for call in request.call_args_list))

    def test_message_mouse_click_selects_evidence(self):
        self.run_coach()
        c = self.w.coach
        self.w.resize(1300, 1000)
        self.w.show()
        self.app.processEvents()
        item = c.panel.list.item(1)
        message = item.data(Qt.ItemDataRole.UserRole)
        c.panel.list.scrollToItem(item)
        self.app.processEvents()
        QTest.mouseClick(c.panel.list.viewport(), Qt.MouseButton.LeftButton, pos=c.panel.list.visualItemRect(item).center())
        self.wait_for(lambda: self.w.index == message.start_frame)
        self.assertEqual(c.panel.selected_message().result_id, message.result_id)
        self.assertEqual(len(c.markers), 2)

    def test_event_edit_stales_report_preserves_review_after_reanalysis(self):
        first = self.run_coach()
        c = self.w.coach
        message = first.messages[0]
        c.apply_review(message, "confirmed", "보존할 사용자 메모")
        event = self.w.motion.document.events[0]
        self.w.motion.apply_manual(replace(event, user_label="변경", source="manual", manually_modified=True))
        self.assertTrue(c.is_stale(first))
        c.show_evidence(message)
        self.assertFalse(c.markers)
        second = self.run_coach()
        review = c.reviews[review_key(message.event_id, message.rule_id)]
        self.assertEqual(review.note, "보존할 사용자 메모")
        self.assertNotEqual(review.reviewed_input_key, second.input_key)
        self.assertTrue(any(row["action"] == "reanalyze" for row in c.history))

    def test_review_dialog_validation_cancel_hide_and_restore(self):
        report = self.run_coach()
        c = self.w.coach
        message = report.messages[0]
        dialog = ReviewDialog(CoachReview(), self.w)
        dialog.note.setPlainText("취소됨")
        dialog.reject()
        self.assertEqual(dialog.result(), QDialog.DialogCode.Rejected)
        self.assertFalse(c.reviews)
        dialog.state.setCurrentIndex(dialog.state.findData("error"))
        dialog.apply()
        self.assertIn("이유", dialog.error.text())
        dialog.reason.setText("관절 가림")
        dialog.apply()
        self.assertEqual(dialog.result(), QDialog.DialogCode.Accepted)
        c.apply_review(message, "error", "메모", "관절 가림", True)
        c.panel.more_check.setChecked(True)
        self.assertEqual(c.panel.list.count(), 6)
        c.panel.hidden_check.setChecked(True)
        self.assertEqual(c.panel.list.count(), 7)
        c.apply_review(message, "deferred", "메모", hidden=False)
        c.panel.hidden_check.setChecked(False)
        self.assertEqual(c.panel.list.count(), 7)

    def test_save_reload_export_historical_and_settings(self):
        report = self.run_coach()
        c = self.w.coach
        message = report.messages[0]
        c.apply_review(message, "confirmed", "재로딩 메모")
        loaded, reviews, confirmations, history = c.store.load(self.w.timeline)
        self.assertEqual(loaded[report.event_id].input_key, report.input_key)
        self.assertEqual(reviews[review_key(message.event_id, message.rule_id)].note, "재로딩 메모")
        c.export(self.root / "coach.json")
        c.export(self.root / "coach.csv")
        self.assertTrue((self.root / "coach.csv").read_bytes().startswith(b"\xef\xbb\xbf"))
        c.live.clear()
        c.refresh()
        self.assertIn("재계산 필요", c.panel.status.text())
        c.panel_check.setChecked(True)
        self.assertTrue(c.prepare_close())
        self.assertTrue(self.w.settings.value("coach/showPanel", False, type=bool))
        self.assertTrue(self.w.motion.document.events)

    def test_rule_change_same_version_invalidation_and_bad_rules_recovery(self):
        report = self.run_coach()
        c = self.w.coach
        path = self.root / "rules.json"
        data = json.loads(DEFAULT_RULES.read_text(encoding="utf-8"))
        data["rules"][0]["suggestion"] += " 추가 촬영 확인."
        path.write_text(json.dumps(data), encoding="utf-8")
        c.rules_path = path
        c.check_rules()
        self.assertTrue(c.is_stale(report))
        newer = self.run_coach()
        self.assertNotEqual(report.input_key, newer.input_key)
        path.write_text("{}", encoding="utf-8")
        c.check_rules()
        self.assertIsNone(c.rules)
        self.assertTrue(c.is_stale(newer))
        self.w.player.request("seek", 10)
        self.wait_for(lambda: self.w.index == 10)

    def test_source_change_and_cancel_discard_late_result(self):
        from app.core.coach_worker import analyze_coaching
        c = self.w.coach
        def delayed(*args, **kwargs):
            time.sleep(0.08)
            return analyze_coaching(*args, **kwargs)
        with patch("app.core.coach_worker.analyze_coaching", side_effect=delayed):
            c.run()
            self.w.motion.hand_combo.setCurrentIndex(self.w.motion.hand_combo.findData("left"))
            self.wait_for(lambda: not c.workers)
        self.assertFalse(c.reports)
        self.assertFalse(c.analyze_button.isEnabled())

    def test_error_and_cancel_keep_player_and_previous_report(self):
        first = self.run_coach()
        c = self.w.coach
        with patch("app.core.coach_worker.analyze_coaching", side_effect=RuntimeError("test coaching error")):
            c.run()
            self.wait_for(lambda: not c.workers)
        self.assertIn("test coaching error", self.coach_errors)
        self.assertIs(c.reports[first.event_id], first)
        self.coach_errors.clear()
        from app.core.coach_worker import analyze_coaching
        def delayed(*args, **kwargs):
            time.sleep(0.08)
            return analyze_coaching(*args, **kwargs)
        with patch("app.core.coach_worker.analyze_coaching", side_effect=delayed):
            c.run()
            c.cancel_button.click()
            self.wait_for(lambda: not c.workers)
        self.assertIs(c.reports[first.event_id], first)
        self.w.player.request("seek", 12)
        self.wait_for(lambda: self.w.index == 12)

    def test_reset_archives_results_keeps_notes(self):
        report = self.run_coach()
        c = self.w.coach
        message = report.messages[0]
        c.apply_review(message, "deferred", "초기화 후 보존")
        with patch("app.ui.coach_controller.QMessageBox.question", return_value=QMessageBox.StandardButton.No):
            c.reset_results()
        self.assertTrue(c.reports)
        with patch("app.ui.coach_controller.QMessageBox.question", return_value=QMessageBox.StandardButton.Yes):
            c.reset_results()
        self.assertFalse(c.reports)
        self.assertEqual(c.reviews[review_key(message.event_id, message.rule_id)].note, "초기화 후 보존")
        self.assertTrue(any(item["action"] == "reset" for item in c.history))

    def test_all_swings_compare_and_switch_selected(self):
        frames, selections, measurements, _, _, _ = synthetic_case((1.0, 3.4))
        self.w.cache.frames, self.w.cache.selections, self.w.cache.measurements = frames, selections, measurements
        self.w.measurement_revision += 1
        self.run_motion()
        c = self.w.coach
        for event in self.w.motion.document.events:
            self.w.motion.select_event(event.event_id)
            c.panel.confirm_check.setChecked(True)
        c.panel.compare_check.setChecked(True)
        c.panel.group_edit.setText("동일 기술 테스트")
        self.run_coach(True)
        self.assertEqual(len(c.reports), 2)
        self.assertTrue(all(not r.comparison["reasons"] for r in c.reports.values()))
        event = self.w.motion.document.events[0]
        self.w.motion.select_event(event.event_id)
        self.assertEqual(c.panel.report.event_id, event.event_id)
