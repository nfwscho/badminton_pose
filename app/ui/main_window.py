# 플레이어 메인 화면. MP4 선택, 재생/일시정지, 이전/다음 프레임, 위치 슬라이더와
# 시간 표시를 제공한다. 창 크기/위치/상태 및 마지막 폴더를 INI에 저장한다.
# 영상 디코딩은 VideoPlayer에 위임하며 오류를 대화상자와 로그로 알린다.
# 별도 분석/추적 스레드의 프레임 캐시를 동기화하고 선수 클릭 선택, 신뢰도
# 필터 골격 표시 및 취소/재분석을 제공한다. 원본 QPixmap은 항상 보존한다.
# PTS 시간축, 선택 세션별 관절 수치/각도 오버레이, 접이식 그래프/통계 패널,
# A/B 구간 및 CSV를 추가한다. 기존 추론 캐시와 재생 명령을 재사용한다.
import logging
from pathlib import Path

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (QCheckBox, QFileDialog, QHBoxLayout, QLabel, QMainWindow,
                             QMessageBox, QProgressBar, QPushButton, QSizePolicy, QSlider, QSplitter, QVBoxLayout, QWidget)

from app.core.video_player import VideoPlayer
from app.core.frame_processor import AnalysisCache, AnalysisWorker, CsvExportWorker, MeasurementWorker, cache_key
from app.core.player_tracker import FrameSelection, TrackingWorker, pick_person
from app.core.video_timeline import TimelineWorker, video_identity
from app.ui.pose_overlay import VideoLabel, draw_angle_overlay, draw_overlay
from app.ui.measurement_panel import MeasurementPanel

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
        self.video_path = None
        self.cache = AnalysisCache("")
        self.analysis_worker = None
        self.tracking_workers = []
        self.selection_revision = 0
        self.tracking_pending = False
        self.timeline = None
        self.video_revision = 0
        self.measurement_revision = 0
        self.timeline_workers, self.measurement_workers, self.export_workers = [], [], []
        self.measurement_pending = False
        self.measurement_usable_count = 0
        self.range_a, self.range_b = None, None
        container = QWidget()
        self.setCentralWidget(container)
        layout = QVBoxLayout(container)
        self.open_button = QPushButton("MP4 동영상 불러오기")
        layout.addWidget(self.open_button)
        ai_row = QHBoxLayout()
        self.analyze_button = QPushButton("AI 자세 분석")
        self.cancel_button = QPushButton("분석 취소")
        self.select_button = QPushButton("선수 선택")
        self.clear_button = QPushButton("선수 선택 해제")
        self.overlay_check = QCheckBox("골격 표시 ON/OFF")
        self.overlay_check.setChecked(self.settings.value("showSkeleton", True, type=bool))
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setMaximumWidth(160)
        for widget in (self.analyze_button, self.cancel_button, self.select_button,
                       self.clear_button, self.overlay_check, self.progress):
            ai_row.addWidget(widget)
        layout.addLayout(ai_row)
        extra_row = QHBoxLayout()
        self.angle_check = QCheckBox("관절 각도 표시 ON/OFF")
        self.angle_check.setChecked(self.settings.value("showAngles", False, type=bool))
        self.panel_check = QCheckBox("관절 측정·그래프 패널")
        self.panel_check.setChecked(self.settings.value("showMeasurements", False, type=bool))
        self.timeline_status = QLabel("시간축: 영상 없음")
        for widget in (self.angle_check, self.panel_check, self.timeline_status):
            extra_row.addWidget(widget)
        extra_row.addStretch()
        layout.addLayout(extra_row)
        self.ai_status = QLabel("관절: 미분석 | 캐시: 없음")
        self.track_status = QLabel("선수: 미선택")
        for label in (self.ai_status, self.track_status):
            label.setWordWrap(True)
            layout.addWidget(label)
        self.video = VideoLabel("MP4 동영상을 불러오세요")
        self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video.setMinimumSize(320, 180)
        self.video.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.video.setStyleSheet("background: #181818; color: #dddddd;")
        self.measurement_panel = MeasurementPanel(self)
        self.measurement_panel.setVisible(self.panel_check.isChecked())
        self.content_splitter = QSplitter(Qt.Orientation.Vertical)
        self.content_splitter.addWidget(self.video)
        self.content_splitter.addWidget(self.measurement_panel)
        self.content_splitter.setCollapsible(0, False)
        self.content_splitter.setSizes([500, 240])
        layout.addWidget(self.content_splitter, 1)
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
        self.update_ai_controls()
        self.update_time()
        self.player = VideoPlayer(self)
        self.player.opened.connect(self.on_opened)
        self.player.frame_ready.connect(self.on_frame)
        self.player.playing_changed.connect(lambda playing: self.play.setText("일시정지" if playing else "재생"))
        self.player.error.connect(self.on_error)
        self.analyze_button.clicked.connect(self.start_analysis)
        self.cancel_button.clicked.connect(self.cancel_analysis)
        self.select_button.clicked.connect(self.begin_selection)
        self.clear_button.clicked.connect(self.clear_selection)
        self.overlay_check.toggled.connect(self.render_frame)
        self.angle_check.toggled.connect(self.render_frame)
        self.panel_check.toggled.connect(self.measurement_panel.setVisible)
        self.measurement_panel.smooth_check.toggled.connect(self.render_frame)
        self.measurement_panel.a_button.clicked.connect(lambda: self.set_range("a"))
        self.measurement_panel.b_button.clicked.connect(lambda: self.set_range("b"))
        self.measurement_panel.clear_range_button.clicked.connect(self.clear_range)
        self.measurement_panel.csv_button.clicked.connect(self.choose_csv)
        self.measurement_panel.time_clicked.connect(self.seek_time)
        self.video.image_clicked.connect(self.select_person)
        self.video.resized.connect(self.render_frame)
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
        split = self.settings.value("measurementSplitter")
        if split:
            self.content_splitter.restoreState(split)
        self.measurement_panel.smooth_check.setChecked(self.settings.value("smoothMeasurements", False, type=bool))
        for key, check in self.measurement_panel.checks.items():
            check.setChecked(self.settings.value("metricVisible/" + key, check.isChecked(), type=bool))
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
        self.cancel_analysis()
        self.clear_selection()
        self.video_path = path
        self.cache = AnalysisCache(path)
        self.video_revision += 1
        self.timeline = None
        self.range_a = self.range_b = None
        self.refresh_measurements()
        for worker in self.timeline_workers:
            worker.requestInterruption()
        token = self.video_revision
        worker = TimelineWorker(path, self)
        self.timeline_workers.append(worker)
        worker.ready.connect(lambda timeline: self.accept_timeline(token, timeline))
        worker.error.connect(lambda message: self.timeline_error(token, message))
        worker.finished.connect(lambda: self.background_finished(worker, self.timeline_workers))
        self.timeline_status.setText("시간축: 원본 PTS 확인 중")
        worker.start()
        self.progress.setValue(0)
        self.count, self.fps = count, fps
        self.slider.blockSignals(True)
        self.slider.setRange(0, count - 1)
        self.slider.blockSignals(False)
        self.set_controls(True)
        self.settings.setValue("lastDirectory", str(Path(path).parent))
        self.statusBar().showMessage(path)

    def on_frame(self, image, index):
        self.index = index
        self.original = QPixmap.fromImage(image)
        self.video.source_size = (image.width(), image.height())
        self.render_frame()
        if not self.slider.isSliderDown():
            self.slider.blockSignals(True)
            self.slider.setValue(index)
            self.slider.blockSignals(False)
            self.update_time()

    def render_frame(self):
        if self.original is not None:
            frame = self.cache.frames.get(self.index)
            selection = self.cache.selections.get(self.index)
            selected_index = selection.selected_pose_index if selection else None
            rendered = (draw_overlay(self.original, frame, selected_index)
                        if frame is not None and self.overlay_check.isChecked() else self.original)
            measurement = self.cache.measurements.get(self.index)
            if (self.angle_check.isChecked() and measurement and measurement.analyzable and frame
                    and selected_index is not None and measurement.player_id == selection.player_id):
                rendered = draw_angle_overlay(rendered, frame.poses[selected_index], measurement,
                                              self.measurement_panel.smooth_check.isChecked())
            self.video.setPixmap(rendered.scaled(self.video.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.update_ai_controls()
        self.measurement_panel.update_current(self.index)

    def update_ai_controls(self):
        busy = self.analysis_worker is not None
        self.analyze_button.setEnabled(self.video_path is not None and not busy)
        self.analyze_button.setText("AI 재분석" if self.cache.frames else "AI 자세 분석")
        self.cancel_button.setEnabled(busy and not self.analysis_worker.isInterruptionRequested())
        frame = self.cache.frames.get(self.index)
        self.select_button.setEnabled(frame is not None and any(p.trackable for p in frame.poses)
                                      and not busy and not self.tracking_pending)
        self.clear_button.setEnabled(self.cache.anchor is not None or self.video.selecting)
        states = {"empty": "없음", "running": "분석 중", "complete": "완료",
                  "cancelled": "취소·부분 결과", "error": "오류·부분 결과", "stale": "무효·재분석 필요"}
        detection = {"detected": "검출", "not_detected": "미검출", "low_confidence": "신뢰도 부족"}
        info = (f"{detection[frame.status]} 유효 {sum(p.trackable for p in frame.poses)}명 (후보 {len(frame.poses)}명) / {frame.timestamp_ms}ms"
                if frame else "이 프레임 결과 없음")
        self.ai_status.setText(f"관절: {info} | 캐시: {states[self.cache.state]} ({len(self.cache.frames)}/{self.count} 프레임)")
        selection = self.cache.selections.get(self.index)
        tracking = {"selected": "선택 프레임", "tracked": "연속 추적 중",
                    "lost": "추적 실패 — 재선택 필요", "ambiguous": "대상 혼동 위험 — 재선택 필요",
                    "reselect_required": "추적 중단 — 재선택 필요", "not_analyzed": "미분석 — 재선택 필요"}
        if self.video.selecting:
            text = "영상에서 선수 클릭 (겹치지 않고 몸통 관절이 보이는 프레임)"
        elif self.tracking_pending:
            text = "선택 선수 추적 계산 중"
        elif self.cache.anchor is None:
            text = "미선택"
        elif selection:
            text = tracking[selection.state]
        else:
            text = "이 프레임 추적 결과 없음"
        self.track_status.setText("선수: " + text)
        if selection and selection.stop_frame is not None:
            reasons = {"not_detected": "관절 미검출", "low_confidence": "몸통 신뢰도 부족", "identity_ambiguous": "후보 혼동",
                       "motion_or_identity_gate": "이동/신원 연속성 기준 초과", "not_analyzed": "미분석"}
            self.track_status.setText(self.track_status.text() + f" (최초 중단 {selection.stop_frame}번: {reasons.get(selection.stop_reason, selection.stop_reason)})")
        if self.cache.anchor is not None:
            self.track_status.setText(self.track_status.text() + (" | 관절 수치 계산 중" if self.measurement_pending else f" | 측정 가능 {self.measurement_usable_count}프레임"))
        self.measurement_panel.csv_button.setEnabled(bool(self.cache.measurements) and not self.measurement_pending and not self.export_workers)

    def start_analysis(self):
        if self.video_path is None or self.analysis_worker is not None:
            return
        self.clear_selection()
        cache = AnalysisCache(self.video_path, state="running")
        self.cache = cache
        self.progress.setValue(0)
        worker = AnalysisWorker(self.video_path, parent=self)
        worker.timeline = self.timeline
        self.analysis_worker = worker
        worker.batch_ready.connect(lambda batch: self.accept_batch(cache, batch))
        worker.progress.connect(lambda value: self.progress.setValue(value) if self.cache is cache else None)
        worker.completed.connect(lambda state, key, elapsed: self.analysis_completed(cache, state, key, elapsed))
        worker.error.connect(lambda message: self.on_ai_error(message) if self.cache is cache else None)
        worker.finished.connect(lambda: self.analysis_finished(worker))
        worker.start()
        self.render_frame()

    def accept_batch(self, cache, batch):
        if self.cache is not cache:
            return
        for frame in batch:
            cache.frames[frame.frame_index] = frame
        self.render_frame()

    def analysis_completed(self, cache, state, key, elapsed):
        if self.cache is not cache:
            return
        cache.state, cache.key = state, key
        self.statusBar().showMessage(f"AI 분석 {state} / {elapsed:.1f}초")
        self.render_frame()

    def analysis_finished(self, worker):
        if self.analysis_worker is worker:
            self.analysis_worker = None
        worker.deleteLater()
        self.update_ai_controls()

    def cancel_analysis(self):
        if self.analysis_worker is not None:
            if not self.analysis_worker.isInterruptionRequested():
                log.info("AI 분석 취소 요청")
            self.analysis_worker.requestInterruption()

    def begin_selection(self):
        if not self.select_button.isEnabled():
            return
        # 분석 후 파일이 외부에서 바뀐 캐시를 선수 선택에 사용하지 않는다.
        try:
            if self.cache.key != cache_key(self.video_path):
                self.cache.frames.clear()
                self.cache.state = "stale"
                self.clear_selection()
                self.progress.setValue(0)
                raise RuntimeError("영상 또는 모델이 변경되었습니다. AI 재분석이 필요합니다.")
        except Exception as exc:
            self.on_ai_error(str(exc))
            return
        self.player.request("pause")
        self.video.selecting = True
        self.video.setCursor(Qt.CursorShape.CrossCursor)
        self.update_ai_controls()

    def select_person(self, x, y):
        if not self.video.selecting:
            return
        frame = self.cache.frames.get(self.index)
        if frame is None:
            self.statusBar().showMessage("현재 프레임은 분석되지 않았습니다.")
            return
        pose_index = pick_person(frame.poses, x, y)
        if pose_index is None:
            self.statusBar().showMessage("선수를 확정할 수 없습니다. 겹치지 않는 선수의 몸통을 클릭하세요.")
            log.info("선수 선택 보류: frame=%s, x=%.3f, y=%.3f", self.index, x, y)
            return
        self.clear_selection()
        cache, revision = self.cache, self.selection_revision
        cache.anchor = (self.index, pose_index)
        cache.selections[self.index] = FrameSelection(self.index, pose_index, "selected", revision)
        log.info("선수 선택: frame=%s, pose_index=%s, x=%.3f, y=%.3f", self.index, pose_index, x, y)
        self.tracking_pending = True
        worker = TrackingWorker(cache.frames, self.index, pose_index, self)
        worker.player_id = revision
        self.tracking_workers.append(worker)
        worker.ready.connect(lambda selections: self.accept_tracking(cache, revision, selections))
        worker.error.connect(lambda message: self.on_ai_error(message) if revision == self.selection_revision else None)
        worker.finished.connect(lambda: self.tracking_finished(worker, revision))
        worker.start()
        self.render_frame()

    def accept_tracking(self, cache, revision, selections):
        if self.cache is cache and revision == self.selection_revision:
            cache.selections = selections
            self.tracking_pending = False
            self.render_frame()
            self.start_measurements()

    def tracking_finished(self, worker, revision):
        if worker in self.tracking_workers:
            self.tracking_workers.remove(worker)
        if revision == self.selection_revision:
            self.tracking_pending = False
            self.update_ai_controls()
        worker.deleteLater()

    def clear_selection(self):
        if self.cache.anchor is not None or self.video.selecting:
            log.info("선수 선택 해제")
        self.selection_revision += 1
        for worker in self.tracking_workers:
            worker.requestInterruption()
        self.tracking_pending = False
        self.cache.anchor = None
        self.cache.selections.clear()
        self.cache.measurements.clear()
        self.measurement_revision += 1
        self.measurement_pending = False
        self.measurement_usable_count = 0
        for worker in self.measurement_workers:
            worker.requestInterruption()
        self.refresh_measurements()
        self.video.selecting = False
        self.video.unsetCursor()
        self.render_frame()

    def on_ai_error(self, message):
        log.error("AI 작업 오류 안내: %s", message)
        self.statusBar().showMessage("AI 오류 — 동영상 플레이어는 계속 사용할 수 있습니다.")
        QMessageBox.warning(self, "AI 분석 오류", message)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "original"):
            self.render_frame()

    def seek(self, value):
        if not self.slider.isSliderDown():
            self.player.request("seek", value)

    def update_time(self, index=None):
        position = self.index if index is None else index
        seconds = self.timeline.seconds[position] if self.timeline and position < len(self.timeline.seconds) else position / self.fps
        duration = self.timeline.duration if self.timeline else self.count / self.fps
        self.time_label.setText(f"{format_time(seconds)} / {format_time(duration)}")

    def accept_timeline(self, token, timeline):
        if self.video_revision != token:
            return
        if len(timeline.seconds) != self.count:
            self.timeline_error(token, "시간축과 재생 프레임 수가 달라 수치 분석을 보류합니다.")
            return
        self.timeline = timeline
        fallback = timeline.sources.count("fps_fallback")
        self.timeline_status.setText(f"시간축: 원본 PTS (FPS 대체 {fallback}프레임)" if fallback else "시간축: 원본 PTS")
        self.player.request("timeline", (self.video_path, timeline))
        self.update_time()
        self.refresh_measurements()
        if self.cache.selections:
            self.start_measurements()

    def timeline_error(self, token, message):
        if self.video_revision == token:
            self.timeline_status.setText("시간축 오류 — 재생 시간은 FPS 추정, 수치 분석 보류")
            log.error("원본 시간축 오류: %s", message)

    def background_finished(self, worker, collection):
        if worker in collection:
            collection.remove(worker)
        worker.deleteLater()
        self.update_ai_controls()

    def start_measurements(self):
        if self.timeline is None or not self.cache.selections or self.original is None or self.tracking_pending:
            return
        for old in self.measurement_workers:
            old.requestInterruption()
        cache, revision = self.cache, self.selection_revision
        self.measurement_revision += 1
        calculation = self.measurement_revision
        self.measurement_pending = True
        worker = MeasurementWorker(cache.frames, cache.selections, self.timeline,
                                   self.original.width(), self.original.height(), self)
        self.measurement_workers.append(worker)
        worker.ready.connect(lambda measurements: self.accept_measurements(cache, revision, calculation, measurements))
        worker.error.connect(lambda message: self.measurement_error(cache, revision, calculation, message))
        worker.finished.connect(lambda: self.background_finished(worker, self.measurement_workers))
        worker.start()

    def accept_measurements(self, cache, revision, calculation, measurements):
        if self.cache is cache and revision == self.selection_revision and calculation == self.measurement_revision:
            cache.measurements = measurements
            self.measurement_pending = False
            self.measurement_usable_count = sum(m.analyzable for m in measurements.values())
            self.refresh_measurements()
            self.render_frame()

    def measurement_error(self, cache, revision, calculation, message):
        if self.cache is cache and revision == self.selection_revision and calculation == self.measurement_revision:
            self.measurement_pending = False
            self.on_ai_error(message)

    def refresh_measurements(self):
        self.measurement_panel.set_data(self.cache.measurements, self.timeline, self.range_a, self.range_b)

    def seek_time(self, seconds):
        if self.timeline:
            self.player.request("seek", self.timeline.nearest_frame(seconds))

    def set_range(self, endpoint):
        if not self.timeline:
            self.statusBar().showMessage("원본 시간축 확인 후 구간을 지정하세요.")
            return
        if (endpoint == "a" and self.range_b is not None and self.index > self.range_b) or (endpoint == "b" and self.range_a is not None and self.index < self.range_a):
            self.statusBar().showMessage("구간 시작 A는 종료 B보다 늦을 수 없습니다.")
            return
        if endpoint == "a":
            self.range_a = self.index
        else:
            self.range_b = self.index
        log.info("측정 구간 지정: A=%s, B=%s", self.range_a, self.range_b)
        self.refresh_measurements()
        self.measurement_panel.reset_view()

    def clear_range(self):
        self.range_a = self.range_b = None
        log.info("측정 구간 초기화")
        self.refresh_measurements()
        self.measurement_panel.reset_view()

    def choose_csv(self):
        if not self.cache.measurements or self.timeline is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "관절 측정 CSV 저장", "joint_measurements.csv", "CSV (*.csv)")
        if path:
            self.export_measurements(path)

    def export_measurements(self, path):
        if self.timeline is None or self.export_workers:
            return
        try:
            if video_identity(self.video_path) != self.timeline.identity:
                raise RuntimeError("영상 파일이 변경되어 CSV 저장을 보류합니다. 영상을 다시 열고 분석하세요.")
        except Exception as exc:
            self.on_ai_error(str(exc))
            return
        worker = CsvExportWorker(path, self.cache.measurements, self.timeline, self.measurement_panel.bounds, self)
        self.export_workers.append(worker)
        worker.saved.connect(lambda saved: self.statusBar().showMessage("CSV 저장 완료: " + saved))
        worker.error.connect(self.on_ai_error)
        worker.finished.connect(lambda: self.background_finished(worker, self.export_workers))
        worker.start()
        self.update_ai_controls()

    def on_error(self, message):
        self.statusBar().showMessage("동영상 처리 오류")
        QMessageBox.warning(self, "동영상 오류", message)

    def closeEvent(self, event):
        self.cancel_analysis()
        for worker in self.timeline_workers + self.measurement_workers:
            worker.requestInterruption()
        for worker in self.tracking_workers:
            worker.requestInterruption()
        workers = (([self.analysis_worker] if self.analysis_worker is not None else []) + self.tracking_workers
                   + self.timeline_workers + self.measurement_workers + self.export_workers)
        for worker in workers:
            if not worker.wait(5000):
                log.warning("AI 작업 종료 대기 중")
                self.statusBar().showMessage("AI 작업 종료 중입니다. 잠시 후 다시 닫아 주세요.")
                event.ignore()
                return
        self.player.request("quit")
        if not self.player.wait(5000):
            log.warning("디코더 종료 대기 중")
            self.statusBar().showMessage("디코더 종료 중입니다. 잠시 후 다시 닫아 주세요.")
            event.ignore()
            return
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())
        self.settings.setValue("showSkeleton", self.overlay_check.isChecked())
        self.settings.setValue("showAngles", self.angle_check.isChecked())
        self.settings.setValue("showMeasurements", self.panel_check.isChecked())
        self.settings.setValue("measurementSplitter", self.content_splitter.saveState())
        self.settings.setValue("smoothMeasurements", self.measurement_panel.smooth_check.isChecked())
        for key, check in self.measurement_panel.checks.items():
            self.settings.setValue("metricVisible/" + key, check.isChecked())
        self.settings.sync()
        if self.settings.status() != QSettings.Status.NoError:
            log.error("UI 설정 저장 실패: %s", self.settings.status())
        log.info("프로그램 종료 및 UI 상태 저장")
        event.accept()
