# 합성 33관절과 실제 임시 MP4로 신뢰도 필터, 선택/추적 중단, 동기화,
# UI 응답성/오버레이/캐시 격리/취소/오류를 검증한다. 추적 단위 테스트의
# 합성 데이터는 실제 다인 영상에서의 신원 정확도를 보장하는 근거가 아니다.
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from dataclasses import replace
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QTimer, Qt
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from app.core.frame_processor import AnalysisWorker
from app.core.player_tracker import pick_person, track_selection
from app.core.pose_detector import (FrameResult, Joint, LANDMARK_NAMES, PersonPose,
                                   PoseDetector, convert_result, merge_poses)
from app.ui.main_window import MainWindow
from app.ui.pose_overlay import VideoLabel, draw_overlay, image_rect


def person(cx, confidence=1.0):
    points = [(cx, 0.25)] * 33
    for a, b, y in ((11, 12, 0.35), (13, 14, 0.45), (15, 16, 0.55),
                    (23, 24, 0.55), (25, 26, 0.7), (27, 28, 0.85)):
        points[a], points[b] = (cx - 0.04, y), (cx + 0.04, y)
    return PersonPose(tuple(Joint(i, name, *points[i], 0.0, confidence, confidence,
                                  (0.0, 0.0, 0.0)) for i, name in enumerate(LANDMARK_NAMES)))


def frame(i, poses):
    return FrameResult(i, i * 50, tuple(poses), any(p.trackable for p in poses),
                       "detected" if poses else "not_detected")


class TrackingTests(unittest.TestCase):
    def test_crop_coordinates_and_duplicate_suppression(self):
        source = SimpleNamespace(x=0.5, y=0.4, z=0.2, visibility=1.0, presence=1.0)
        result = SimpleNamespace(pose_landmarks=[[source] * 33], pose_world_landmarks=[])
        converted = convert_result(result, 1000, 400, 600, "right")[0]
        self.assertAlmostEqual(converted.joints[0].x, 0.7)
        self.assertAlmostEqual(converted.joints[0].y, 0.4)
        self.assertAlmostEqual(converted.joints[0].z, 0.12)
        self.assertEqual(converted.source_view, "right")
        source.x = -0.1
        converted = convert_result(result, 1000, 400, 600, "right")[0]
        self.assertFalse(converted.joints[0].reliable)
        self.assertGreater(converted.joints[0].x, 0)
        a, b, duplicate = person(0.3), person(0.75), person(0.305)
        merged = merge_poses((a, b, duplicate), 4)
        self.assertEqual(len(merged), 2)
        self.assertEqual(len(merge_poses((a, b), 1)), 1)

    def test_confidence_and_invalid_coordinates(self):
        p = person(0.3)
        self.assertEqual(len(p.joints), 33)
        self.assertTrue(p.trackable)
        self.assertFalse(person(0.3, 0.49).trackable)
        for value in (-1, 2, float("nan")):
            self.assertFalse(replace(p.joints[11], x=value).reliable)
        self.assertFalse(replace(p.joints[11], presence=None).reliable)
        self.assertFalse(replace(p.joints[11], visibility=float("inf")).reliable)

    def test_pick_and_ambiguous_click(self):
        self.assertEqual(pick_person((person(0.3), person(0.7)), 0.3, 0.5), 0)
        self.assertIsNone(pick_person((person(0.3), person(0.32)), 0.31, 0.5))
        self.assertIsNone(pick_person((person(0.3),), 0.9, 0.5))

    def test_order_changes_are_not_identity_changes(self):
        frames = {0: frame(0, (person(0.3), person(0.75))),
                  1: frame(1, (person(0.74), person(0.31))),
                  2: frame(2, (person(0.32), person(0.73)))}
        result = track_selection(frames, 1, 1)
        self.assertEqual([result[i].selected_pose_index for i in range(3)], [0, 1, 0])

    def test_lost_never_reuses_old_pose_or_auto_reacquires(self):
        frames = {0: frame(0, (person(0.3),)), 1: frame(1, ()),
                  2: frame(2, (person(0.3),))}
        result = track_selection(frames, 0, 0)
        self.assertEqual(result[1].state, "lost")
        self.assertIsNone(result[1].selected_pose_index)
        self.assertEqual(result[2].state, "reselect_required")
        self.assertIsNone(result[2].selected_pose_index)

    def test_crossing_and_abrupt_replacement_require_reselection(self):
        frames = {0: frame(0, (person(0.3), person(0.7))),
                  1: frame(1, (person(0.31), person(0.34))),
                  2: frame(2, (person(0.32), person(0.7)))}
        result = track_selection(frames, 0, 0)
        self.assertEqual(result[1].state, "ambiguous")
        self.assertIsNone(result[2].selected_pose_index)
        result = track_selection({0: frames[0], 1: frame(1, (person(0.7),))}, 0, 0)
        self.assertEqual(result[1].state, "lost")


class OverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_overlay_filters_and_does_not_mutate_original(self):
        original = QPixmap(400, 200)
        original.fill(QColor("black"))
        before = original.toImage()
        annotated = draw_overlay(original, frame(0, (person(0.3),)))
        self.assertEqual(original.toImage(), before)
        self.assertNotEqual(annotated.toImage(), before)
        self.assertEqual(draw_overlay(original, frame(0, (person(0.3, 0.1),))).toImage(), before)
        self.assertNotEqual(draw_overlay(original, frame(0, (person(0.3),)), 0).toImage(), annotated.toImage())

    def test_resize_click_mapping_and_letterbox(self):
        label = VideoLabel("")
        label.source_size = (1920, 1080)
        label.selecting = True
        clicked = []
        label.image_clicked.connect(lambda x, y: clicked.append((x, y)))
        for width, height in ((800, 800), (1000, 500)):
            label.resize(width, height)
            rect = image_rect(width, height, 1920, 1080)
            self.assertAlmostEqual(rect.width() / rect.height(), 1920 / 1080)
            point = QPoint(round(rect.x() + rect.width() * 0.3), round(rect.y() + rect.height() * 0.6))
            QTest.mouseClick(label, Qt.MouseButton.LeftButton, pos=point)
            self.assertAlmostEqual(clicked[-1][0], 0.3, delta=0.002)
            self.assertAlmostEqual(clicked[-1][1], 0.6, delta=0.002)
        count = len(clicked)
        # QTest의 QPoint(0, 0)은 기본값으로 해석되어 중앙을 클릭하므로 피한다.
        QTest.mouseClick(label, Qt.MouseButton.LeftButton, pos=QPoint(1, 1))
        self.assertEqual(len(clicked), count)
        label.close()


class FakeDetector:
    def __init__(self, path):
        self.closed = False

    def detect(self, rgb, index, timestamp):
        time.sleep(0.01)
        return replace(frame(index, (person(0.3), person(0.75))), timestamp_ms=timestamp)

    def close(self):
        self.closed = True


class AnalysisUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root / "테스트.mp4"
        writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), 20, (160, 90))
        self.assertTrue(writer.isOpened())
        for _ in range(40):
            writer.write(np.zeros((90, 160, 3), np.uint8))
        writer.release()
        self.key_patch = patch("app.core.frame_processor.cache_key", return_value=("test",))
        self.ui_key_patch = patch("app.ui.main_window.cache_key", return_value=("test",))
        self.worker_patch = patch("app.ui.main_window.AnalysisWorker",
                                  side_effect=lambda path, parent: AnalysisWorker(path, parent=parent, detector_factory=FakeDetector))
        for patcher in (self.key_patch, self.ui_key_patch, self.worker_patch):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.window = MainWindow(self.root / "settings.ini")
        self.errors = []
        self.window.on_ai_error = self.errors.append
        self.window.player.request("open", str(self.path))
        self.wait_for(lambda: self.window.original is not None)

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

    def analyze(self):
        self.window.start_analysis()
        self.wait_for(lambda: self.window.analysis_worker is None)

    def test_analysis_sync_response_selection_toggle(self):
        ticks = []
        timer = QTimer()
        timer.timeout.connect(lambda: ticks.append(1))
        timer.start(5)
        try:
            self.window.start_analysis()
            self.window.next.click()
            self.wait_for(lambda: self.window.index == 1)
            self.assertIsNotNone(self.window.analysis_worker)
            self.wait_for(lambda: self.window.analysis_worker is None)
        finally:
            timer.stop()
        self.assertGreater(len(ticks), 10)
        self.assertEqual(self.window.cache.state, "complete")
        self.assertEqual(self.window.progress.value(), 100)
        self.assertEqual(len(self.window.cache.frames), 40)
        self.window.slider.setValue(20)
        self.wait_for(lambda: self.window.index == 20)
        self.assertEqual(self.window.cache.frames[20].timestamp_ms, 1000)
        self.assertIn("1000ms", self.window.ai_status.text())
        self.window.begin_selection()
        self.window.select_person(0.3, 0.5)
        self.wait_for(lambda: not self.window.tracking_pending)
        self.assertEqual(self.window.cache.selections[20].selected_pose_index, 0)
        self.assertEqual(self.window.cache.selections[39].selected_pose_index, 0)
        self.window.overlay_check.setChecked(False)
        expected = self.window.original.scaled(self.window.video.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                               Qt.TransformationMode.SmoothTransformation).toImage()
        self.assertEqual(self.window.video.pixmap().toImage(), expected)
        self.window.clear_selection()
        self.assertFalse(self.window.cache.selections)
        self.assertFalse(self.errors)

    def test_cancel_partial_and_reanalyze(self):
        self.window.start_analysis()
        self.wait_for(lambda: len(self.window.cache.frames) >= 16)
        self.window.cancel_analysis()
        self.wait_for(lambda: self.window.analysis_worker is None)
        self.assertEqual(self.window.cache.state, "cancelled")
        self.assertLess(len(self.window.cache.frames), 40)
        self.assertTrue(self.window.play.isEnabled())
        self.analyze()
        self.assertEqual(len(self.window.cache.frames), 40)
        self.assertEqual(self.window.cache.state, "complete")

    def test_new_video_discards_in_flight_results(self):
        self.window.start_analysis()
        self.wait_for(lambda: len(self.window.cache.frames) >= 16)
        self.window.player.request("open", str(self.path))
        self.wait_for(lambda: self.window.cache.state == "empty")
        self.wait_for(lambda: self.window.analysis_worker is None)
        self.assertFalse(self.window.cache.frames)
        self.assertEqual(self.window.progress.value(), 0)

    def test_analysis_error_preserves_player(self):
        with patch.object(FakeDetector, "detect", side_effect=RuntimeError("inference failed")):
            self.analyze()
        self.assertEqual(self.window.cache.state, "error")
        self.assertTrue(self.errors)
        self.window.next.click()
        self.wait_for(lambda: self.window.index == 1)

    def test_changed_cache_is_invalidated(self):
        self.analyze()
        with patch("app.ui.main_window.cache_key", return_value=("changed",)):
            self.window.begin_selection()
        self.assertEqual(self.window.cache.state, "stale")
        self.assertFalse(self.window.cache.frames)
        self.assertTrue(self.errors)
        self.assertTrue(self.window.play.isEnabled())

    def test_missing_model_and_non_increasing_timestamp(self):
        with self.assertRaises(FileNotFoundError):
            PoseDetector(self.root / "missing.task")
        detector = PoseDetector.__new__(PoseDetector)
        detector.last_timestamp = 100
        with self.assertRaises(ValueError):
            detector.detect(None, 1, 100)


if __name__ == "__main__":
    unittest.main()
