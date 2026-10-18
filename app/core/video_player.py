# 동영상 디코더와 재생 작업 스레드. MP4 열기, 순차 재생, 프레임 탐색을
# 단일 작업 스레드에서 처리하고 영상/메타데이터/오류를 Qt 신호로 전달한다.
# GUI 스레드에서는 OpenCV를 호출하지 않으며 종료 시 캡처 자원을 해제한다.
# 선수 클릭 선택을 위한 명시적 일시정지 명령도 제공한다. AI 분석 캡처/결과와는
# 독립적으로 동작하며 기존 frame_ready의 프레임 번호를 오버레이 동기화에 쓴다.
import logging
import math
import queue
import time
from pathlib import Path

import cv2
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

log = logging.getLogger(__name__)


class VideoPlayer(QThread):
    opened = Signal(str, int, float)
    frame_ready = Signal(QImage, int)
    playing_changed = Signal(bool)
    error = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.commands = queue.Queue()

    def request(self, command, value=None):
        self.commands.put((command, value))

    def run(self):
        cap = None
        playing = False
        index, count, fps = -1, 0, 30.0
        deadline = 0.0

        def pause():
            nonlocal playing
            playing = False
            self.playing_changed.emit(False)

        def read_frame(target=None):
            nonlocal index
            if target is not None:
                if not cap.set(cv2.CAP_PROP_POS_FRAMES, target):
                    raise RuntimeError("동영상 프레임 탐색에 실패했습니다.")
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("동영상 프레임을 읽을 수 없습니다.")
            index = target if target is not None else index + 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = rgb.shape
            self.frame_ready.emit(QImage(rgb.data, w, h, rgb.strides[0],
                                        QImage.Format.Format_RGB888).copy(), index)

        try:
            while True:
                try:
                    command, value = self.commands.get(timeout=max(0, deadline-time.monotonic()) if playing else None)
                except queue.Empty:
                    command, value = "tick", None
                if command == "quit":
                    break
                try:
                    if command == "open":
                        pause()
                        candidate = cv2.VideoCapture(str(value))
                        try:
                            new_count = candidate.get(cv2.CAP_PROP_FRAME_COUNT)
                            new_fps = candidate.get(cv2.CAP_PROP_FPS)
                            ok, _ = candidate.read()
                            if not candidate.isOpened() or not ok or not math.isfinite(new_count) or new_count < 1:
                                raise RuntimeError("재생 가능한 MP4 파일인지 확인해 주세요.")
                            if not math.isfinite(new_fps) or new_fps <= 0:
                                raise RuntimeError("동영상의 FPS 정보를 확인할 수 없습니다.")
                        except Exception:
                            candidate.release()
                            raise
                        if cap is not None:
                            cap.release()
                        cap, count, fps = candidate, int(new_count), new_fps
                        index = -1
                        self.opened.emit(str(Path(value).resolve()), count, fps)
                        read_frame(0)
                        log.info("동영상 열기: %s, frames=%s, fps=%s", value, count, fps)
                    elif cap is not None:
                        if command == "pause":
                            pause()
                            log.info("선수 선택을 위한 일시정지")
                        elif command == "play":
                            if index >= count - 1:
                                read_frame(0)
                            playing = not playing
                            deadline = time.monotonic() + 1 / fps
                            self.playing_changed.emit(playing)
                            log.info("재생 상태: %s", playing)
                        elif command in ("seek", "step"):
                            pause()
                            target = int(value) if command == "seek" else index + int(value)
                            read_frame(max(0, min(count - 1, target)))
                            log.info("프레임 이동: %s", index)
                        elif command == "tick":
                            if index >= count - 1:
                                pause()
                                log.info("동영상 재생 완료")
                            else:
                                read_frame()
                                deadline = max(deadline + 1 / fps, time.monotonic())
                except Exception as exc:
                    pause()
                    log.exception("동영상 처리 오류: command=%s", command)
                    self.error.emit(str(exc))
        finally:
            if cap is not None:
                cap.release()
            log.info("동영상 작업 스레드 종료")
