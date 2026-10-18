# 기존 MainWindow에 스윙 검출·목록·타임라인·수동 검토·JSON 저장을 연결한다.
# 독립 스레드/입력 세대 검사를 사용해 새 선수·영상·A/B·설정에 오래된 결과가
# 적용되지 않게 한다. 자동 결과와 수동 문서를 분리하고 재분석 덮어쓰기 전에
# 명시적으로 확인한다. 저장 기록은 현재 선수 캐시와 확인되기 전 과거 관측으로 표시한다.
from bisect import bisect_right
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import logging

import pyqtgraph as pg
from PySide6.QtCore import QObject, Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QHBoxLayout, QInputDialog, QLabel,
                              QMessageBox, QProgressBar, QPushButton, QVBoxLayout, QWidget)

from app.analysis.motion_detector import manual_event
from app.analysis.motion_models import MotionConfig, MotionDocument, PHASE_LABELS, make_context
from app.analysis.motion_store import MotionStore
from app.core.motion_worker import MotionWorker
from app.core.video_timeline import video_identity
from app.ui.motion_panel import MotionEditDialog, MotionPanel, MotionSettingsDialog, PHASE_COLORS

log = logging.getLogger(__name__)


class MotionController(QObject):
    def __init__(self, window, settings_path, main_layout):
        super().__init__(window)
        from pathlib import Path
        self.w = window
        previous_splitter_sizes = window.content_splitter.sizes()
        self.store = MotionStore(Path(settings_path).parent / "motion_events")
        self.document, self.features = None, None
        self.historical, self.stale, self.unsaved = False, False, False
        self.workers, self.graph_markers = [], []
        self.epoch, self.events, self.starts = 0, [], []
        self.pending_documents = []
        try:
            config = json.loads(window.settings.value("motion/config", "{}"))
            self.config = MotionConfig(**config)
        except (ValueError, TypeError):
            log.exception("스윙 설정 읽기 실패 — 기본 설정 사용")
            self.config = MotionConfig()
        self.controls = QWidget()
        vertical = QVBoxLayout(self.controls)
        vertical.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self.hand_combo = QComboBox()
        self.hand_combo.addItem("라켓 손 선택", None)
        self.hand_combo.addItem("오른손잡이", "right")
        self.hand_combo.addItem("왼손잡이", "left")
        self.hand_combo.setCurrentIndex(max(0, self.hand_combo.findData(window.settings.value("motion/hand", ""))))
        self.detect_button, self.cancel_button, self.settings_button = (QPushButton(t) for t in ("스윙 자동 검출", "스윙 검출 취소", "스윙 설정"))
        self.phase_check = QCheckBox("동작 단계 표시 ON/OFF")
        self.phase_check.setChecked(window.settings.value("motion/showPhases", True, type=bool))
        self.panel_check = QCheckBox("스윙 목록·타임라인")
        self.panel_check.setChecked(window.settings.value("motion/showPanel", False, type=bool))
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setMaximumWidth(100)
        for widget in (self.hand_combo, self.detect_button, self.cancel_button, self.settings_button, self.progress):
            row.addWidget(widget)
        vertical.addLayout(row)
        self.phase_label = QLabel("동작 단계: 미분석 (구질·임팩트 판정 없음)")
        self.phase_label.setWordWrap(True)
        self.phase_label.setTextFormat(Qt.TextFormat.PlainText)
        display_row = QHBoxLayout()
        for widget in (self.phase_check, self.panel_check):
            display_row.addWidget(widget)
        display_row.addWidget(self.phase_label, 1)
        vertical.addLayout(display_row)
        main_layout.insertWidget(3, self.controls)
        self.panel = MotionPanel(window)
        window.content_splitter.addWidget(self.panel)
        self.panel.setVisible(self.panel_check.isChecked())
        self.panel.timeline_plot.setXLink(window.measurement_panel.plot)
        self.panel.loop_check.setChecked(window.settings.value("motion/loop", False, type=bool))
        self.detect_button.clicked.connect(self.run_analysis)
        self.cancel_button.clicked.connect(self.cancel_requested)
        self.settings_button.clicked.connect(self.open_settings)
        self.hand_combo.currentIndexChanged.connect(self.settings_changed)
        self.phase_check.toggled.connect(self.update_phase_visibility)
        self.panel_check.toggled.connect(self.show_panel)
        self.panel.event_selected.connect(self.select_event)
        self.panel.time_clicked.connect(window.seek_time)
        self.panel.play_button.clicked.connect(self.play_selected)
        self.panel.loop_check.toggled.connect(lambda checked: window.player.request("range_loop", checked))
        self.panel.ab_button.clicked.connect(self.use_event_range)
        self.panel.edit_button.clicked.connect(lambda: self.edit(False))
        self.panel.add_button.clicked.connect(lambda: self.edit(True))
        self.panel.delete_button.clicked.connect(self.delete_selected)
        self.panel.restore_button.clicked.connect(self.restore_automatic)
        self.panel.save_button.clicked.connect(self.save_document)
        self.panel.load_button.clicked.connect(lambda: self.load_document(choose=True))
        self.panel.reset_button.clicked.connect(self.reset_view)
        self.update_phase_visibility()
        splitter_state = window.settings.value("motion/splitter")
        if splitter_state:
            window.content_splitter.restoreState(splitter_state)
        elif not self.panel_check.isChecked():
            window.content_splitter.setSizes([*previous_splitter_sizes, 0])
        else:
            window.content_splitter.setSizes([600, 180, 180])
        self.update_controls()

    def show_panel(self, visible):
        self.panel.setVisible(visible)
        if visible:
            # 새 패널의 기본 sizeHint가 영상 영역을 대부분 차지하지 않도록 한다.
            self.w.content_splitter.setSizes([600, 180, 180])

    def current_context(self):
        if not self.w.timeline or not self.w.cache.anchor or not self.w.cache.measurements or self.hand_combo.currentData() is None:
            return None
        anchor_frame, pose_index = self.w.cache.anchor
        frame = self.w.cache.frames.get(anchor_frame)
        selected = self.w.cache.selections.get(anchor_frame)
        if frame is None or selected is None:
            return None
        return make_context(self.w.timeline.identity, anchor_frame, frame.poses[pose_index],
                            self.hand_combo.currentData(), selected.player_id)

    def scope(self):
        return (self.w.range_a if self.w.range_a is not None else 0,
                self.w.range_b if self.w.range_b is not None else self.w.count - 1)

    def update_controls(self):
        busy = bool(self.workers)
        available = (self.w.timeline is not None and bool(self.w.cache.measurements)
                     and not self.w.measurement_pending and self.w.analysis_worker is None and not self.w.tracking_pending)
        self.detect_button.setEnabled(available and self.hand_combo.currentData() is not None and not busy)
        self.detect_button.setToolTip("필요: 관절 분석 → 선수 선택 → 수치 계산 완료 → 라켓 손 선택. 포즈 추론을 다시 실행하지 않습니다.")
        self.cancel_button.setEnabled(busy)
        event = self.panel.selected_event()
        for button in (self.panel.play_button, self.panel.ab_button, self.panel.edit_button, self.panel.delete_button):
            button.setEnabled(event is not None and not busy)
        for button in (self.panel.add_button, self.panel.save_button):
            button.setEnabled(self.document is not None and not busy)
        self.panel.restore_button.setEnabled(self.document is not None and self.document.modified and not busy)
        self.panel.load_button.setEnabled(self.w.timeline is not None and not busy)

    def update_phase_visibility(self):
        show = self.phase_check.isChecked()
        self.phase_label.setVisible(show)
        self.panel.tabs.setTabVisible(1, show)
        self.draw_graph_phases()

    def settings_changed(self):
        self.invalidate()
        self.w.settings.setValue("motion/hand", self.hand_combo.currentData() or "")
        if self.current_context() is not None:
            self.load_document()

    def open_settings(self):
        dialog = MotionSettingsDialog(self.config, self.w)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.config = dialog.result_config
            self.w.settings.setValue("motion/config", json.dumps(asdict(self.config)))
            self.invalidate()
            log.info("스윙 분석 설정 변경: %s", asdict(self.config))

    def invalidate(self):
        self.epoch += 1
        self.cancel()
        if self.document:
            self.stale = True
            self.refresh()
        self.update_controls()

    def source_cleared(self):
        if self.unsaved and not self.save_document() and self.document is not None:
            self.pending_documents.append((self.document, self.w.timeline))
        self.epoch += 1
        self.cancel()
        self.document, self.features = None, None
        self.w.player.request("clear_range")
        self.historical, self.stale = False, False
        self.unsaved = False
        self.events, self.starts = [], []
        self.panel.selected_id = None
        self.progress.setValue(0)
        self.refresh()

    def timeline_ready(self):
        if not self.document:
            self.load_document()

    def measurements_ready(self):
        self.epoch += 1
        self.cancel()
        self.features = None
        context = self.current_context()
        if context:
            try:
                document = self.store.load(self.w.timeline, context["context_id"])
                self.document, self.historical, self.stale = document, document is not None, False
            except Exception as exc:
                self.show_error("스윙 저장 결과 읽기: " + str(exc))
        self.refresh()

    def confirm_replace(self):
        return QMessageBox.question(self.w, "수동 수정 결과 교체 확인",
                                    "수동 수정한 이벤트가 있습니다. 새 자동 결과로 교체하시겠습니까?\n기존 문서는 JSON 이력에 보존됩니다.",
                                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                    QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes

    def run_analysis(self):
        context = self.current_context()
        if context is None or self.workers:
            return
        try:
            if video_identity(self.w.video_path) != self.w.timeline.identity:
                raise ValueError("영상이 변경되었습니다. 다시 열고 분석하세요.")
            previous = self.document if self.document and self.document.automatic.context_id == context["context_id"] else self.store.load(self.w.timeline, context["context_id"])
            if previous and previous.modified and not self.confirm_replace():
                log.info("스윙 자동 재분석 보류 — 사용자 수동 수정 유지")
                return
        except Exception as exc:
            self.show_error(str(exc))
            return
        self.epoch += 1
        token = self.epoch
        worker = MotionWorker(self.w.cache.frames, self.w.cache.selections, self.w.cache.measurements,
                              self.w.timeline, self.w.original.width(), self.w.original.height(), self.hand_combo.currentData(),
                              self.config, context, self.scope(), self)
        self.workers.append(worker)
        worker.progress.connect(lambda value: self.progress.setValue(value) if token == self.epoch else None)
        worker.ready.connect(lambda result, features: self.accept_result(token, result, features))
        worker.error.connect(lambda message: self.show_error(message) if token == self.epoch else None)
        worker.finished.connect(lambda: self.finished(worker))
        self.progress.setValue(0)
        self.panel.status_label.setText("스윙 분석 중 — 기존 관절/측정 캐시 사용")
        self.panel_check.setChecked(True)
        worker.start()
        self.update_controls()

    def accept_result(self, token, result, features):
        if token != self.epoch:
            return
        self.document, self.features = MotionDocument(result, result.events), features
        self.historical, self.stale = False, False
        self.panel.selected_id = None
        self.unsaved = True
        self.save_document()
        self.refresh()
        self.reset_view()

    def finished(self, worker):
        if worker in self.workers:
            self.workers.remove(worker)
        worker.deleteLater()
        self.update_controls()

    def cancel(self):
        for worker in self.workers:
            worker.requestInterruption()

    def cancel_requested(self):
        self.epoch += 1
        self.cancel()
        self.panel.status_label.setText("스윙 검출 취소 요청 — 이전 결과는 유지됩니다.")
        log.info("스윙 검출 취소 요청")

    def refresh(self):
        self.panel.set_document(self.document, self.w.timeline, self.historical, self.stale)
        self.events = sorted(self.document.events, key=lambda event: event.start_frame) if self.document else []
        self.starts = [event.start_frame for event in self.events]
        self.draw_graph_phases()
        self.on_frame(self.w.index)
        self.update_controls()

    def on_frame(self, index):
        if self.w.timeline and index < len(self.w.timeline.seconds):
            self.panel.cursor.setValue(self.w.timeline.seconds[index])
        if not self.phase_check.isChecked():
            return
        pos = bisect_right(self.starts, index) - 1
        event = self.events[pos] if pos >= 0 and self.events[pos].end_frame >= index else None
        if event:
            phase = next((p for p in event.phases if p.start_frame <= index <= p.end_frame), None)
            label = PHASE_LABELS[phase.phase] if phase else "단계 보류"
            status = "저장 기록, 현재 선수 연결 미확인" if self.historical else "기존 결과·현재 설정으로 재검출 필요" if self.stale else "규칙 후보·수동 검토 필요"
            self.phase_label.setText(f"동작 단계: {label} / {event.kind} — {status}")
        elif self.document:
            interval = next((c for c in self.document.automatic.coverage if c.start_frame <= index <= c.end_frame), None)
            self.phase_label.setText("동작 단계: " + ("관측 불가 / UNKNOWN" if interval and interval.kind == "UNKNOWN" else "스윙 후보 외 구간"))
        else:
            self.phase_label.setText("동작 단계: 미분석 (구질·임팩트 판정 없음)")

    def select_event(self, event_id):
        self.panel.selected_id = event_id
        event = self.panel.selected_event()
        if event:
            self.w.player.request("seek", event.start_frame)
            self.w.measurement_panel.focus_event((event.start_frame, event.end_frame))
            self.w.measurement_panel.plot.setXRange(event.start_seconds, event.end_seconds, padding=0.03)
            self.draw_graph_phases()
            self.update_controls()

    def draw_graph_phases(self):
        plot = self.w.measurement_panel.plot
        for marker in self.graph_markers:
            plot.removeItem(marker)
        self.graph_markers.clear()
        event = self.panel.selected_event()
        if event and self.phase_check.isChecked():
            for phase in event.phases:
                color = pg.mkColor(PHASE_COLORS[phase.phase])
                color.setAlpha(25)
                region = pg.LinearRegionItem((phase.start_seconds, phase.end_seconds), movable=False,
                                             brush=color, pen=pg.mkPen(PHASE_COLORS[phase.phase], style=Qt.PenStyle.DashLine))
                region.setZValue(-20)
                region.setToolTip(PHASE_LABELS[phase.phase] + " (후보)")
                plot.addItem(region)
                self.graph_markers.append(region)

    def reset_view(self):
        if self.w.timeline:
            self.w.measurement_panel.focus_event(None)
            self.w.measurement_panel.plot.setXRange(0, self.w.timeline.duration, padding=0.01)
            self.panel.timeline_plot.setYRange(0, 1, padding=0)

    def play_selected(self):
        event = self.panel.selected_event()
        if event:
            self.w.player.request("play_range", (event.start_frame, event.end_frame, self.panel.loop_check.isChecked()))

    def use_event_range(self):
        event = self.panel.selected_event()
        if event:
            self.w.range_a, self.w.range_b = event.start_frame, event.end_frame
            self.w.refresh_measurements()
            self.invalidate()
            log.info("선택 스윙을 A/B에 적용: %s", event.event_id)

    def edit(self, adding=False):
        if self.document is None:
            return
        existing = None if adding else self.panel.selected_event()
        if not adding and existing is None:
            return
        a = self.w.range_a if self.w.range_a is not None else self.w.index
        b = self.w.range_b if self.w.range_b is not None else min(self.w.count - 1, a + 30)
        factory = lambda start, end, kind, spans, label: manual_event(self.document, self.w.timeline, start, end, kind,
                                                                   spans, label, existing, self.features,
                                                                   self.w.cache.measurements if self.features is not None else None)
        dialog = MotionEditDialog(self.w.timeline, factory, existing, (a, b), self.w)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.apply_manual(dialog.result_event, "add" if adding else "edit")

    def apply_manual(self, event, action="edit"):
        from app.analysis.motion_models import validate_event
        validate_event(event, self.w.timeline, self.document.events)
        events = [other for other in self.document.events if other.event_id != event.event_id] + [event]
        self.change_events(tuple(sorted(events, key=lambda item: item.start_frame)), action, event.event_id)
        self.panel.selected_id = event.event_id
        self.refresh()

    def change_events(self, events, action, event_id):
        self.w.player.request("pause")
        operation = {"action": action, "event_id": event_id, "at": datetime.now(timezone.utc).isoformat()}
        self.document = replace(self.document, events=events, operations=(*self.document.operations, operation))
        self.unsaved = True
        log.info("스윙 수동 수정: event=%s, action=%s", event_id, action)
        self.save_document()
        self.refresh()

    def delete_selected(self):
        event = self.panel.selected_event()
        if event:
            self.panel.selected_id = None
            self.change_events(tuple(e for e in self.document.events if e.event_id != event.event_id), "delete", event.event_id)

    def restore_automatic(self):
        if not self.document or not self.document.modified:
            return
        if QMessageBox.question(self.w, "전체 자동 결과 복원", "모든 수동 변경을 원본 자동 결과로 복원하시겠습니까?\n수정 전 문서는 JSON 이력에 남습니다.",
                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
            self.change_events(self.document.automatic.events, "restore_automatic", "all")

    def save_document(self):
        if self.document is None or self.w.timeline is None:
            return True
        try:
            path = self.store.save(self.document, self.w.timeline)
            self.unsaved = False
            log.info("스윙 문서 저장: %s, manual=%s", path, self.document.modified)
            return True
        except Exception as exc:
            self.show_error("스윙 저장 실패: " + str(exc))
            return False

    def load_document(self, choose=False):
        if self.w.timeline is None:
            return
        try:
            if self.unsaved and not self.save_document():
                return
            context = self.current_context()
            context_id = context["context_id"] if context else None
            if choose:
                documents = self.store.available(self.w.timeline)
                if len(documents) > 1:
                    labels = [f"{d.automatic.racket_hand} / 선택 프레임 {d.automatic.source_anchor['frame']} / 범위 {d.automatic.scope} / 수동 {d.modified} / {d.automatic.context_id[:8]}"
                              for d in documents]
                    label, accepted = QInputDialog.getItem(self.w, "저장된 선수/라켓 손 문맥", "불러올 기록 (과거 관측으로 표시)", labels, 0, False)
                    if not accepted:
                        return
                    context_id = documents[labels.index(label)].automatic.context_id
                elif documents:
                    context_id = documents[0].automatic.context_id
            document = self.store.load(self.w.timeline, context_id)
            if document:
                self.document, self.features = document, None
                self.historical, self.stale, self.unsaved = True, False, False
                if self.hand_combo.currentData() is None:
                    self.hand_combo.blockSignals(True)
                    self.hand_combo.setCurrentIndex(self.hand_combo.findData(document.automatic.racket_hand))
                    self.hand_combo.blockSignals(False)
                log.info("스윙 문서 불러오기: context=%s, manual=%s (현재 선수와 자동 결합하지 않음)", document.automatic.context_id, document.modified)
                self.refresh()
            elif choose:
                self.w.statusBar().showMessage("이 영상/선수 문맥의 저장된 스윙 결과가 없습니다.")
        except Exception as exc:
            self.show_error("스윙 결과 읽기 실패: " + str(exc))

    def show_error(self, message):
        log.error("스윙 기능 오류: %s", message)
        self.w.statusBar().showMessage(message)
        QMessageBox.warning(self.w, "스윙 분석 오류", message)

    def prepare_close(self):
        self.epoch += 1
        self.cancel()
        for worker in self.workers:
            if not worker.wait(5000):
                self.w.statusBar().showMessage("스윙 작업 종료 중입니다. 잠시 후 다시 닫아 주세요.")
                return False
        if self.unsaved and not self.save_document():
            return False
        try:
            for document, timeline in self.pending_documents:
                self.store.save(document, timeline)
            self.pending_documents.clear()
        except Exception as exc:
            self.show_error("이전 선수/영상의 미저장 수정 보존 실패: " + str(exc))
            return False
        self.w.settings.setValue("motion/hand", self.hand_combo.currentData() or "")
        self.w.settings.setValue("motion/config", json.dumps(asdict(self.config)))
        self.w.settings.setValue("motion/showPhases", self.phase_check.isChecked())
        self.w.settings.setValue("motion/showPanel", self.panel_check.isChecked())
        self.w.settings.setValue("motion/loop", self.panel.loop_check.isChecked())
        self.w.settings.setValue("motion/splitter", self.w.content_splitter.saveState())
        return True
