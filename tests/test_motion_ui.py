# 임시 실제 MP4와 정답 포함 합성 관절 캐시로 스윙 UI 통합/회귀 동작을 검사한다.
# 백그라운드 응답, 기존 A/B 보존, 실제 타임라인 마우스 탐색, 수동 적용/취소,
# 재분석 확인, JSON 재로딩과 선택 구간 재생/반복을 검증한다. 영상 인식 정확도
# 검증과 구분하며 포즈 추론을 호출하지 않는다.
# 코칭 준비 검사와도 연결할 수 있도록 임시 영상의 실제 원본/모델 캐시 서명을 설정한다.
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from dataclasses import replace
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QTimer, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from test_motion import synthetic_case
from app.analysis.motion_detector import manual_event
from app.core.frame_processor import cache_key
from app.ui.main_window import MainWindow
from app.ui.motion_panel import MotionEditDialog


class MotionUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root / "swing.mp4"
        writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), 50, (200, 160))
        self.assertTrue(writer.isOpened())
        for _ in range(300):
            writer.write(np.zeros((160, 200, 3), dtype=np.uint8))
        writer.release()
        self.w = MainWindow(self.root / "settings.ini")
        self.errors = []
        self.w.motion.show_error = self.errors.append
        self.w.player.request("open", str(self.path))
        self.wait_for(lambda: self.w.timeline is not None and self.w.original is not None)
        frames, selections, measurements, _, _, _ = synthetic_case()
        self.w.cache.frames, self.w.cache.selections, self.w.cache.measurements = frames, selections, measurements
        self.w.cache.anchor, self.w.cache.state = (0, 0), "complete"
        self.w.cache.key = cache_key(self.path)
        self.w.refresh_measurements()
        self.w.motion.hand_combo.setCurrentIndex(self.w.motion.hand_combo.findData("right"))
        self.w.render_frame()

    def tearDown(self):
        self.w.close()
        self.app.processEvents()
        self.assertFalse(self.w.player.isRunning())
        self.temp.cleanup()

    def wait_for(self, condition, seconds=10):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.app.processEvents()
            if condition():
                return
            time.sleep(0.005)
        self.fail("Qt 동작 완료 시간 초과")

    def run_motion(self):
        self.w.motion.run_analysis()
        self.wait_for(lambda: not self.w.motion.workers)
        self.assertIsNotNone(self.w.motion.document)
        self.assertFalse(self.errors)

    def test_async_response_no_inference_and_event_focus_preserves_ab(self):
        from app.core.motion_worker import extract_features
        ticks = []
        timer = QTimer()
        timer.timeout.connect(lambda: ticks.append(1))
        timer.start(5)
        def delayed(*args, **kwargs):
            time.sleep(0.06)
            return extract_features(*args, **kwargs)
        try:
            self.w.range_a, self.w.range_b = 30, 200
            self.w.refresh_measurements()
            with patch("app.core.motion_worker.extract_features", side_effect=delayed), patch("app.core.pose_detector.PoseDetector.detect", side_effect=AssertionError("추론 재실행 금지")):
                self.run_motion()
        finally:
            timer.stop()
        self.assertGreater(len(ticks), 3)
        event = self.w.motion.document.events[0]
        before = (self.w.range_a, self.w.range_b, self.w.measurement_panel.bounds)
        self.w.motion.select_event(event.event_id)
        self.wait_for(lambda: self.w.index == event.start_frame)
        self.assertEqual((self.w.range_a, self.w.range_b, self.w.measurement_panel.bounds), before)
        self.assertEqual(self.w.measurement_panel.display_bounds, (event.start_frame, event.end_frame))
        self.assertTrue(self.w.motion.graph_markers)

    def test_timeline_mouse_seek_cursor_and_phase_toggle(self):
        self.run_motion()
        motion = self.w.motion
        event = motion.document.events[0]
        motion.select_event(event.event_id)
        self.w.panel_check.setChecked(True)
        motion.panel.tabs.setCurrentIndex(1)
        self.w.resize(1200, 950)
        self.w.show()
        self.app.processEvents()
        motion.reset_view()
        self.app.processEvents()
        np.testing.assert_allclose(motion.panel.timeline_plot.viewRange()[0], self.w.measurement_panel.plot.viewRange()[0], atol=1e-6)
        target = event.peak_seconds
        plot = motion.panel.timeline_plot
        point = plot.mapFromScene(plot.getViewBox().mapViewToScene(QPointF(target, 0.5)))
        QTest.mouseClick(plot.viewport(), Qt.MouseButton.LeftButton, pos=point)
        self.wait_for(lambda: abs(self.w.timeline.seconds[self.w.index] - target) <= 0.03)
        self.assertAlmostEqual(motion.panel.cursor.value(), self.w.timeline.seconds[self.w.index])
        self.assertAlmostEqual(self.w.measurement_panel.cursor.value(), motion.panel.cursor.value())
        markers = list(motion.graph_markers)
        self.w.next.click()
        self.wait_for(lambda: self.w.index > event.start_frame)
        self.assertEqual(motion.graph_markers, markers)
        motion.phase_check.setChecked(False)
        self.assertFalse(motion.graph_markers)
        motion.phase_check.setChecked(True)
        self.assertTrue(motion.graph_markers)

    def test_manual_apply_cancel_delete_restore_and_reanalysis_confirmation(self):
        self.run_motion()
        motion = self.w.motion
        original = motion.document
        event = original.events[0]
        factory = lambda a, b, kind, spans, label: manual_event(motion.document, self.w.timeline, a, b, kind, spans, label,
                                                               event, motion.features, self.w.cache.measurements)
        cancelled = MotionEditDialog(self.w.timeline, factory, event, parent=self.w)
        cancelled.user_label.setText("취소될 수정")
        cancelled.reject()
        self.assertEqual(motion.document, original)
        dialog = MotionEditDialog(self.w.timeline, factory, event, parent=self.w)
        dialog.user_label.setText("사용자 검토")
        dialog.apply()
        self.assertIsNotNone(dialog.result_event)
        motion.apply_manual(dialog.result_event)
        self.assertTrue(motion.document.modified)
        self.assertEqual(motion.document.automatic, original.automatic)
        with patch.object(motion, "confirm_replace", return_value=False) as confirm:
            motion.run_analysis()
            confirm.assert_called_once()
        self.assertFalse(motion.workers)
        self.assertTrue(motion.document.modified)
        motion.delete_selected()
        self.assertFalse(motion.document.events)
        with patch("app.ui.motion_controller.QMessageBox.question", return_value=QMessageBox.StandardButton.No):
            motion.restore_automatic()
        self.assertFalse(motion.document.events)
        with patch("app.ui.motion_controller.QMessageBox.question", return_value=QMessageBox.StandardButton.Yes):
            motion.restore_automatic()
        self.assertEqual(motion.document.events, original.events)
        self.assertFalse(motion.document.modified)

    def test_manual_json_reload_is_historical_and_settings_remain(self):
        self.run_motion()
        motion = self.w.motion
        event = motion.document.events[0]
        edited = manual_event(motion.document, self.w.timeline, event.start_frame, event.end_frame, "UNKNOWN", (), "저장한 라벨",
                              event, motion.features, self.w.cache.measurements)
        motion.apply_manual(edited)
        motion.panel.loop_check.setChecked(True)
        self.w.close()
        self.app.processEvents()
        self.w = MainWindow(self.root / "settings.ini")
        self.w.motion.show_error = self.errors.append
        self.w.player.request("open", str(self.path))
        self.wait_for(lambda: self.w.motion.document is not None)
        self.assertEqual(self.w.motion.document.events[0].user_label, "저장한 라벨")
        self.assertTrue(self.w.motion.historical)
        self.assertIsNone(self.w.motion.features)
        self.assertFalse(self.w.cache.measurements)
        self.assertEqual(self.w.motion.hand_combo.currentData(), "right")
        self.assertTrue(self.w.motion.panel.loop_check.isChecked())

    def test_hand_change_cancels_in_flight_result(self):
        from app.core.motion_worker import extract_features
        def delayed(*args, **kwargs):
            time.sleep(0.08)
            return extract_features(*args, **kwargs)
        with patch("app.core.motion_worker.extract_features", side_effect=delayed):
            self.w.motion.run_analysis()
            self.w.motion.hand_combo.setCurrentIndex(self.w.motion.hand_combo.findData("left"))
            self.wait_for(lambda: not self.w.motion.workers)
        self.assertIsNone(self.w.motion.document)
        self.run_motion()
        self.assertFalse(self.w.motion.document.events)
        self.assertIn("스윙 후보 없음", self.w.motion.panel.status_label.text())

    def test_cancel_button_preserves_previous_document(self):
        self.run_motion()
        previous = self.w.motion.document
        from app.core.motion_worker import extract_features
        def delayed(*args, **kwargs):
            time.sleep(0.08)
            return extract_features(*args, **kwargs)
        with patch("app.core.motion_worker.extract_features", side_effect=delayed):
            self.w.motion.run_analysis()
            self.w.motion.cancel_button.click()
            self.wait_for(lambda: not self.w.motion.workers)
        self.assertEqual(self.w.motion.document, previous)
        self.assertIn("취소", self.w.motion.panel.status_label.text())

    def test_selected_range_play_loop_and_manual_seek(self):
        self.run_motion()
        motion = self.w.motion
        short = manual_event(motion.document, self.w.timeline, 2, 5, "UNKNOWN", (), "짧은 수동 구간",
                             features=motion.features, measurements=self.w.cache.measurements)
        motion.apply_manual(short, "add")
        seen = []
        self.w.player.frame_ready.connect(lambda image, index: seen.append(index))
        motion.panel.loop_check.setChecked(False)
        motion.play_selected()
        self.wait_for(lambda: 5 in seen and self.w.play.text() == "재생")
        self.assertTrue(all(2 <= i <= 5 for i in seen))
        seen.clear()
        motion.panel.loop_check.setChecked(True)
        motion.play_selected()
        self.wait_for(lambda: seen.count(2) >= 3)
        self.assertTrue(all(2 <= i <= 5 for i in seen))
        self.w.player.request("seek", 20)
        self.wait_for(lambda: self.w.index == 20 and self.w.play.text() == "재생")
        time.sleep(0.08)
        self.app.processEvents()
        self.assertEqual(self.w.index, 20)

    def test_worker_error_keeps_existing_player_and_motion_document(self):
        self.run_motion()
        previous = self.w.motion.document
        with patch("app.core.motion_worker.extract_features", side_effect=RuntimeError("motion test error")):
            self.w.motion.run_analysis()
            self.wait_for(lambda: not self.w.motion.workers)
        self.assertTrue(self.errors)
        self.assertEqual(self.w.motion.document, previous)
        self.w.next.click()
        self.wait_for(lambda: self.w.index == 1)


if __name__ == "__main__":
    unittest.main()
