# 관절 측정의 수학적 기준, 신뢰도/신원/결측, 구간 통계 및 CSV 원시값을 검증한다.
# 합성 VFR MP4를 실제 인코딩해 PTS/FPS 차이를 확인하고 Qt 그래프/영상 탐색,
# 측정 캐시 갱신/초기화, 오버레이 및 설정 저장의 통합 동작도 검사한다.
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import csv
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import av
import cv2
import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest

from app.analysis.joint_angles import (METRICS, angle_degrees, calculate_measurements, export_csv,
                                       measure_pose, measurement_bounds, summarize)
from app.core.frame_processor import AnalysisWorker
from app.core.player_tracker import FrameSelection, track_selection
from app.core.pose_detector import FrameResult, Joint, LANDMARK_NAMES, PersonPose
from app.core.video_timeline import VideoTimeline, read_timeline, video_identity
from app.ui.main_window import MainWindow
from app.ui.pose_overlay import draw_angle_overlay


def pose(width=1000, height=500, shift=0):
    points = {11: (200 + shift, 100), 13: (200 + shift, 200), 15: (200 + shift, 300),
              12: (400 + shift, 100), 14: (400 + shift, 200), 16: (500 + shift, 200),
              23: (200 + shift, 300), 25: (200 + shift, 350), 27: (200 + shift, 450),
              24: (400 + shift, 300), 26: (400 + shift, 350), 28: (500 + shift, 350)}
    return PersonPose(tuple(Joint(i, name, points.get(i, (300 + shift, 100))[0] / width,
                                  points.get(i, (300 + shift, 100))[1] / height, 0, 1, 1)
                            for i, name in enumerate(LANDMARK_NAMES)))


def result(index, poses):
    return FrameResult(index, index * 50, tuple(poses), bool(poses), "detected" if poses else "not_detected")


def timeline(count=4):
    return VideoTimeline(tuple(i * 0.05 for i in range(count)), tuple(i * 0.05 for i in range(count)),
                         ("pts",) * count, count * 0.05, 1000, 500, ("sample.mp4", 100, 1000))


def make_vfr(path):
    stamps = (0, 40, 140, 180, 380, 420, 620, 660)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=25)
        stream.width, stream.height = 160, 90
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, 1000)
        stream.codec_context.time_base = Fraction(1, 1000)
        for index, stamp in enumerate(stamps):
            frame = av.VideoFrame.from_ndarray(np.full((90, 160, 3), index * 30, dtype=np.uint8), format="rgb24")
            frame.pts, frame.time_base = stamp, Fraction(1, 1000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return stamps


class AngleMathTests(unittest.TestCase):
    def test_straight_right_angle_and_degenerate(self):
        self.assertAlmostEqual(angle_degrees((0, 0), (1, 0), (2, 0)), 180)
        self.assertAlmostEqual(angle_degrees((0, 0), (1, 0), (1, 1)), 90)
        self.assertIsNone(angle_degrees((1, 1), (1, 1), (2, 2)))
        self.assertIsNone(angle_degrees((1e-9, 0), (0, 0), (0, 1)))
        self.assertIsNone(angle_degrees((np.nan, 0), (0, 0), (1, 1)))

    def test_left_right_aspect_ratio_and_y_axis(self):
        p = pose()
        before = p.joints
        values = measure_pose(p, 1000, 500)
        self.assertAlmostEqual(values["left_elbow"].raw, 180)
        self.assertAlmostEqual(values["right_elbow"].raw, 90)
        self.assertAlmostEqual(values["left_knee"].raw, 180)
        self.assertAlmostEqual(values["right_knee"].raw, 90)
        self.assertAlmostEqual(values["trunk_tilt"].raw, 0)
        self.assertAlmostEqual(values["left_wrist_height"].raw, -1)
        joints = list(p.joints)
        joints[15] = replace(joints[15], y=0)
        self.assertGreater(measure_pose(PersonPose(tuple(joints)), 1000, 500)["left_wrist_height"].raw, 0)
        self.assertEqual(p.joints, before)
        # 이미지에서 직각인 대각선 두 벡터는 정규화 좌표를 그대로 쓰면 왜곡된다.
        joints = list(p.joints)
        for i, xy in ((11, (100, 100)), (13, (200, 200)), (15, (300, 100))):
            joints[i] = replace(joints[i], x=xy[0] / 1000, y=xy[1] / 500)
        self.assertAlmostEqual(measure_pose(PersonPose(tuple(joints)), 1000, 500)["left_elbow"].raw, 90)

    def test_missing_low_confidence_and_zero_torso(self):
        joints = list(pose().joints)
        joints[13] = replace(joints[13], visibility=0.2)
        values = measure_pose(PersonPose(tuple(joints)), 1000, 500)
        self.assertFalse(values["left_elbow"].valid)
        self.assertIsNone(values["left_elbow"].raw)
        self.assertTrue(values["right_elbow"].valid)
        for hip, shoulder in ((23, 11), (24, 12)):
            joints[hip] = replace(joints[hip], x=joints[shoulder].x, y=joints[shoulder].y)
        values = measure_pose(PersonPose(tuple(joints)), 1000, 500)
        self.assertFalse(values["trunk_tilt"].valid)
        self.assertFalse(values["left_wrist_height"].valid)

    def test_only_selected_identity_and_no_auto_recovery(self):
        a, b = pose(), pose(shift=500)
        other_joints = list(b.joints)
        other_joints[15] = replace(other_joints[15], x=0.8, y=0.4)
        b = PersonPose(tuple(other_joints))
        frames = {0: result(0, (a, b)), 1: result(1, (b, a)), 2: result(2, ()), 3: result(3, (a, b))}
        selections = track_selection(frames, 0, 0, player_id=42)
        measurements = calculate_measurements(frames, selections, timeline(), 1000, 500)
        self.assertEqual(selections[1].selected_pose_index, 1)
        self.assertEqual(measurements[1].player_id, 42)
        self.assertEqual(measurements[1].values["left_elbow"].raw, 180)
        self.assertEqual(measure_pose(b, 1000, 500)["left_elbow"].raw, 90)
        self.assertEqual(selections[3].stop_frame, 2)
        self.assertEqual(selections[3].stop_reason, "not_detected")
        self.assertFalse(measurements[2].analyzable)
        self.assertFalse(measurements[3].analyzable)
        self.assertTrue(all(v.raw is None for v in measurements[3].values.values()))

    def test_statistics_range_missing_and_insufficient(self):
        frames = {i: result(i, (pose(),)) for i in range(4)}
        selections = {i: FrameSelection(i, 0, "tracked", 7) for i in (0, 1, 3)}
        measurements = calculate_measurements(frames, selections, timeline(), 1000, 500)
        stats = summarize(measurements, timeline(), (0, 3))
        self.assertEqual(stats["usable_frames"], 3)
        self.assertEqual(stats["ratio"], 0.75)
        self.assertEqual(stats["metrics"]["left_elbow"]["min"], 180)
        self.assertEqual(stats["metrics"]["left_elbow"]["range"], 0)  # 실제로 변화가 없을 때의 0만 허용.
        single = summarize(measurements, timeline(), (3, 3))
        self.assertIsNone(single["metrics"]["left_elbow"]["min"])
        self.assertEqual(measurement_bounds(measurements, 4, 1, 2), (1, 2))
        self.assertEqual(measurement_bounds(measurements, 4), (0, 3))

    def test_smoothing_never_fills_missing_or_crosses_session(self):
        frames = {i: result(i, (pose(),)) for i in range(4)}
        selections = {0: FrameSelection(0, 0, "selected", 1), 2: FrameSelection(2, 0, "selected", 2),
                      3: FrameSelection(3, 0, "tracked", 2)}
        measurements = calculate_measurements(frames, selections, timeline(), 1000, 500)
        self.assertIsNone(measurements[1].values["left_elbow"].display)
        self.assertNotEqual(measurements[0].track_segment, measurements[2].track_segment)
        self.assertEqual(measurements[0].values["left_elbow"].raw, 180)

    def test_csv_identity_time_validity_and_raw_values(self):
        tl = timeline()
        frames = {i: result(i, (pose(),)) for i in range(4)}
        selections = {i: FrameSelection(i, 0, "tracked", 99) for i in (0, 1, 3)}
        measurements = calculate_measurements(frames, selections, tl, 1000, 500)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "측정.csv"
            export_csv(path, measurements, tl, (1, 3))
            with path.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["video_path"], "sample.mp4")
        self.assertEqual(float(rows[0]["timestamp_seconds"]), tl.seconds[1])
        self.assertEqual(float(rows[0]["left_elbow"]), measurements[1].values["left_elbow"].raw)
        self.assertEqual(rows[0]["player_session_id"], "99")
        self.assertEqual(rows[1]["left_elbow"], "")
        self.assertEqual(rows[1]["left_elbow_valid"], "0")


class VfrUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root / "vfr.mp4"
        self.stamps = make_vfr(self.path)
        self.window = MainWindow(self.root / "settings.ini")
        self.window.player.request("open", str(self.path))
        self.wait_for(lambda: self.window.original is not None and self.window.timeline is not None)

    def tearDown(self):
        self.window.close()
        self.app.processEvents()
        self.assertFalse(self.window.player.isRunning())
        self.temp.cleanup()

    def wait_for(self, condition):
        end = time.monotonic() + 10
        while time.monotonic() < end:
            self.app.processEvents()
            if condition():
                return
            time.sleep(0.005)
        self.fail("Qt 작업 시간 초과")

    def install_measurements(self):
        cache = self.window.cache
        cache.frames = {i: result(i, (pose(),)) for i in range(8)}
        cache.selections = {i: FrameSelection(i, 0, "tracked", 5) for i in range(8) if i != 3}
        cache.anchor = (0, 0)
        self.window.start_measurements()
        self.wait_for(lambda: bool(cache.measurements) and not self.window.measurement_pending)

    def test_vfr_pts_player_and_graph_click_sync_without_replot(self):
        tl = self.window.timeline
        self.assertEqual(tl.sources, ("pts",) * 8)
        np.testing.assert_allclose(tl.seconds, np.asarray(self.stamps) / 1000, atol=1e-6)
        self.assertNotAlmostEqual(tl.seconds[4], 4 / self.window.fps, places=2)
        self.install_measurements()
        panel = self.window.measurement_panel
        self.window.panel_check.setChecked(True)
        self.window.resize(1100, 850)
        self.window.show()
        self.app.processEvents()
        panel.reset_view()
        for index in (7, 1, 6, 0):
            self.window.player.request("seek", index)
            self.wait_for(lambda: self.window.index == index)
            self.assertAlmostEqual(self.window.original.toImage().pixelColor(0, 0).red(), index * 30, delta=4)
        self.app.processEvents()
        reloads = panel.reload_count
        position = panel.plot.mapFromScene(panel.plot.getViewBox().mapViewToScene(QPointF(0.38, 90)))
        QTest.mouseClick(panel.plot.viewport(), Qt.MouseButton.LeftButton, pos=position)
        self.wait_for(lambda: self.window.index == 4)
        self.assertIn("00:00:00.380", self.window.time_label.text())
        self.assertAlmostEqual(self.window.original.toImage().pixelColor(0, 0).red(), 4 * 30, delta=4)
        self.assertAlmostEqual(panel.cursor.value(), 0.38)
        self.assertEqual(panel.reload_count, reloads)
        _, values = panel.curves["left_elbow"].getData()
        self.assertTrue(np.isnan(values[3]))
        self.assertFalse(panel.curves["left_elbow"].opts["connect"][2])
        panel.checks["left_elbow"].setChecked(False)
        self.assertFalse(panel.curves["left_elbow"].isVisible())
        panel.reset_view()

    def test_ab_stats_csv_and_video_change(self):
        self.install_measurements()
        self.window.player.request("seek", 2)
        self.wait_for(lambda: self.window.index == 2)
        self.window.set_range("a")
        self.window.player.request("seek", 5)
        self.wait_for(lambda: self.window.index == 5)
        self.window.set_range("b")
        panel = self.window.measurement_panel
        self.assertEqual(panel.bounds, (2, 5))
        self.assertIn("3/4", panel.range_label.text())
        path = self.root / "export.csv"
        self.window.export_measurements(path)
        self.wait_for(lambda: not self.window.export_workers)
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        current = self.window.cache.measurements[5].values["right_elbow"].raw
        self.assertAlmostEqual(float(rows[-1]["right_elbow"]), current)
        self.assertEqual(panel.table.item(1, 1).text(), f"{current:.1f}")
        self.window.player.request("open", str(self.path))
        self.wait_for(lambda: self.window.cache.state == "empty" and not self.window.cache.frames)
        self.assertFalse(self.window.cache.measurements)
        self.assertIsNone(self.window.range_a)
        self.assertIsNone(self.window.range_b)

    def test_angle_overlay_missing_and_settings(self):
        self.install_measurements()
        original = QPixmap(1000, 500)
        original.fill(QColor("black"))
        measured = self.window.cache.measurements[0]
        self.assertNotEqual(draw_angle_overlay(original, pose(), measured).toImage(), original.toImage())
        self.assertEqual(draw_angle_overlay(original, pose(), self.window.cache.measurements[3]).toImage(), original.toImage())
        self.window.angle_check.setChecked(True)
        self.window.panel_check.setChecked(True)
        self.window.measurement_panel.smooth_check.setChecked(True)
        self.window.close()
        self.app.processEvents()
        self.window = MainWindow(self.root / "settings.ini")
        self.assertTrue(self.window.angle_check.isChecked())
        self.assertTrue(self.window.panel_check.isChecked())
        self.assertTrue(self.window.measurement_panel.smooth_check.isChecked())

    def test_reselection_replaces_session_measurements_and_keeps_range(self):
        self.install_measurements()
        self.window.range_a, self.window.range_b = 2, 5
        old = self.window.cache.measurements[0]
        self.window.video.selecting = True
        self.window.select_person(0.3, 0.5)
        self.wait_for(lambda: not self.window.tracking_pending and not self.window.measurement_pending and bool(self.window.cache.measurements))
        self.assertNotEqual(self.window.cache.measurements[0].player_id, old.player_id)
        self.assertIsNot(self.window.cache.measurements[0], old)
        self.assertTrue(self.window.cache.measurements[3].analyzable)
        self.assertEqual((self.window.range_a, self.window.range_b), (2, 5))

    def test_ai_uses_pts_but_does_not_require_new_inference_for_measurements(self):
        calls = []

        class Detector:
            def __init__(self, path):
                pass

            def detect(self, rgb, index, timestamp):
                calls.append(timestamp)
                return result(index, (pose(),))

            def close(self):
                pass

        worker = AnalysisWorker(self.path, detector_factory=Detector)
        worker.timeline = self.window.timeline
        frames, errors = [], []
        worker.batch_ready.connect(lambda batch: frames.extend(batch))
        worker.error.connect(errors.append)
        with patch("app.core.frame_processor.cache_key", return_value=("test",)):
            worker.start()
            self.wait_for(lambda: not worker.isRunning())
            self.app.processEvents()
        self.assertFalse(errors)
        self.assertEqual(calls, list(self.stamps))
        self.assertEqual([r.timestamp_seconds for r in frames], list(self.window.timeline.seconds))
        self.install_measurements()
        self.assertEqual(len(calls), 8)


if __name__ == "__main__":
    unittest.main()
