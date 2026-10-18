# 플레이어 메인 화면. MP4 선택, 재생/일시정지, 이전/다음 프레임, 위치 슬라이더와
# 시간 표시를 제공한다. 창 크기/위치/상태 및 마지막 폴더를 INI에 저장한다.
# 영상 디코딩은 VideoPlayer에 위임하며 오류를 대화상자와 로그로 알린다.
import logging

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (QFileDialog, QHBoxLayout, QLabel, QMainWindow,
                             QMessageBox, QPushButton, QSizePolicy, QSlider, QVBoxLayout, QWidget)

from app.core.video_player import VideoPlayer

log = logging.getLogger(__name__)


def format_time(seconds):
    milliseconds = max(0, round(seconds * 1000))
    seconds, ms = divmod(milliseconds, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{ms:03}"


class MainWindow(QMainWindow):
    def __init__(self, settings_path):
        super().__init__()
        self.settings = QSettings(str(settings_path), QSettings.Format.IniFormat)
        self.setWindowTitle("Badminton AI Coach — 동영상 플레이어")
        self.resize(1100, 760)
        self.setMinimumSize(640, 420)
        self.fps, self.count, self.index = 30.0, 0, 0
        self.original = None
        container = QWidget()
        self.setCentralWidget(container)
        layout = QVBoxLayout(container)
        self.open_button = QPushButton("MP4 동영상 불러오기")
        layout.addWidget(self.open_button)
        self.video = QLabel("MP4 동영상을 불러오세요")
        self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video.setMinimumSize(320, 180)
        self.video.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.video.setStyleSheet("background: #181818; color: #dddddd;")
        layout.addWidget(self.video, 1)
        row = QHBoxLayout()
        self.previous = QPushButton("◀ 이전 프레임")
        self.play = QPushButton("재생")
        self.next = QPushButton("다음 프레임 ▶")
        self.time_label = QLabel()
        for widget in (self.previous, self.play, self.next, self.time_label):
            row.addWidget(widget)
        layout.addLayout(row)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        layout.addWidget(self.slider)
        self.set_controls(False)
        self.update_time()
        self.player = VideoPlayer(self)
        self.player.opened.connect(self.on_opened)
        self.player.frame_ready.connect(self.on_frame)
        self.player.playing_changed.connect(lambda playing: self.play.setText("일시정지" if playing else "재생"))
        self.player.error.connect(self.on_error)
        self.open_button.clicked.connect(self.choose_file)
        self.play.clicked.connect(lambda: self.player.request("play"))
        self.previous.clicked.connect(lambda: self.player.request("step", -1))
        self.next.clicked.connect(lambda: self.player.request("step", 1))
        self.slider.sliderMoved.connect(lambda value: self.update_time(value))
        self.slider.valueChanged.connect(self.seek)
        self.slider.sliderReleased.connect(lambda: self.player.request("seek", self.slider.value()))
        for key, button in (("Space", self.play), ("Left", self.previous), ("Right", self.next), ("Ctrl+O", self.open_button)):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(button.click)
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        state = self.settings.value("windowState")
        if state:
            self.restoreState(state)
        self.statusBar().showMessage("준비")
        self.player.start()

    def set_controls(self, enabled):
        for widget in (self.previous, self.play, self.next, self.slider):
            widget.setEnabled(enabled)

    def choose_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "MP4 동영상 열기", self.settings.value("lastDirectory", ""), "MP4 동영상 (*.mp4)")
        if path:
            self.statusBar().showMessage("동영상 불러오는 중…")
            self.player.request("open", path)

    def on_opened(self, path, count, fps):
        from pathlib import Path
        self.count, self.fps = count, fps
        self.slider.setRange(0, count - 1)
        self.set_controls(True)
        self.settings.setValue("lastDirectory", str(Path(path).parent))
        self.statusBar().showMessage(path)

    def on_frame(self, image, index):
        self.index = index
        self.original = QPixmap.fromImage(image)
        self.render_frame()
        if not self.slider.isSliderDown():
            self.slider.blockSignals(True)
            self.slider.setValue(index)
            self.slider.blockSignals(False)
            self.update_time()

    def render_frame(self):
        if self.original is not None:
            self.video.setPixmap(self.original.scaled(self.video.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "original"):
            self.render_frame()

    def seek(self, value):
        if not self.slider.isSliderDown():
            self.player.request("seek", value)

    def update_time(self, index=None):
        position = self.index if index is None else index
        self.time_label.setText(f"{format_time(position / self.fps)} / {format_time(self.count / self.fps)}")

    def on_error(self, message):
        self.statusBar().showMessage("동영상 처리 오류")
        QMessageBox.warning(self, "동영상 오류", message)

    def closeEvent(self, event):
        self.player.request("quit")
        if not self.player.wait(5000):
            log.warning("디코더 종료 대기 중")
            self.statusBar().showMessage("디코더 종료 중입니다. 잠시 후 다시 닫아 주세요.")
            event.ignore()
            return
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())
        self.settings.sync()
        if self.settings.status() != QSettings.Status.NoError:
            log.error("UI 설정 저장 실패: %s", self.settings.status())
        log.info("프로그램 종료 및 UI 상태 저장")
        event.accept()
