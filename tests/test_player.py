# 실제 임시 MP4를 생성하여 작업 스레드의 열기/프레임 이동/재생/오류 처리를
# 검증하고, 오프스크린 Qt 화면에서 컨트롤 및 INI 창 상태 복원을 확인한다.
# 표준 unittest만 사용하며 임시 파일은 테스트 종료 시 정리한다.
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication
from app.ui.main_window import MainWindow, format_time


class PlayerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root / "테스트 영상.mp4"
        writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), 20, (96, 64))
        self.assertTrue(writer.isOpened())
        for i in range(20):
            writer.write(np.full((64, 96, 3), i * 10, np.uint8))
        writer.release()
        self.window = MainWindow(self.root / "settings.ini")
        self.errors = []
        self.window.player.error.disconnect(self.window.on_error)
        self.window.player.error.connect(self.errors.append)

    def tearDown(self):
        self.window.close()
        self.app.processEvents()
        self.assertFalse(self.window.player.isRunning())
        self.temp.cleanup()

    def wait_for(self, condition):
        end = time.monotonic() + 5
        while time.monotonic() < end:
            self.app.processEvents()
            if condition():
                return
            time.sleep(0.01)
        self.fail("Qt 작업 완료 시간 초과")

    def load(self):
        self.window.player.request("open", str(self.path))
        self.wait_for(lambda: self.window.original is not None)

    def test_open_step_seek_and_boundaries(self):
        self.assertFalse(self.window.play.isEnabled())
        self.load()
        self.assertEqual(self.window.count, 20)
        self.assertAlmostEqual(self.window.fps, 20)
        self.window.next.click()
        self.wait_for(lambda: self.window.index == 1)
        self.window.previous.click()
        self.wait_for(lambda: self.window.index == 0)
        self.window.slider.setValue(12)
        self.wait_for(lambda: self.window.index == 12)
        self.window.player.request("seek", 999)
        self.wait_for(lambda: self.window.index == 19)
        self.window.player.request("seek", -99)
        self.wait_for(lambda: self.window.index == 0)
        self.assertIn("00:00:01.000", self.window.time_label.text())
        self.assertFalse(self.errors)

    def test_play_pause_end_and_restart(self):
        self.load()
        self.window.play.click()
        self.wait_for(lambda: self.window.index >= 2)
        self.window.play.click()
        self.wait_for(lambda: self.window.play.text() == "재생")
        index = self.window.index
        until = time.monotonic() + 0.15
        while time.monotonic() < until:
            self.app.processEvents()
            time.sleep(0.01)
        self.assertEqual(self.window.index, index)
        self.window.player.request("seek", 18)
        self.wait_for(lambda: self.window.index == 18)
        self.window.play.click()
        self.wait_for(lambda: self.window.index == 19 and self.window.play.text() == "재생")
        self.window.play.click()
        self.wait_for(lambda: self.window.index < 19)

    def test_bad_file_preserves_loaded_video(self):
        self.load()
        bad = self.root / "broken.mp4"
        bad.write_bytes(b"not a video")
        self.window.player.request("open", str(bad))
        self.wait_for(lambda: bool(self.errors))
        self.window.next.click()
        self.wait_for(lambda: self.window.index == 1)

    def test_settings_restore(self):
        # 오프스크린 가상 화면(800px) 안에서 Qt의 자동 화면 보정을 피한다.
        self.window.resize(700, 500)
        self.load()
        self.window.close()
        self.window = MainWindow(self.root / "settings.ini")
        self.assertEqual(self.window.width(), 700)
        self.assertEqual(self.window.height(), 500)
        self.assertEqual(self.window.settings.value("lastDirectory"), str(self.root))

    def test_time_format(self):
        self.assertEqual(format_time(3661.234), "01:01:01.234")
        self.assertEqual(format_time(0), "00:00:00.000")


if __name__ == "__main__":
    unittest.main()
