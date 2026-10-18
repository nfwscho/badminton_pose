# 기존 MainWindow/스윙 선택에 접이식 코칭 패널과 비동기 로컬 규칙 엔진을 연결한다.
# 영상/선수/손/측정/이벤트/규칙/사용자 확인 변경 시 결과를 오래된 것으로 표시한다.
# 근거 탐색·강조·반복은 기존 A/B를 바꾸지 않으며 같은 선수의 현재 근거만 연결한다.
# 저장된 결과/검토/메모/자동 이력을 분리 보존하고 실패 시 종료 전에 다시 저장한다.
# 5.1에서는 A/B 변경과 선수/설정 변경을 구분하고 원본 캐시 조건을 검사한다.
# 빈 패널을 간결하게 표시하며 사용자가 조절한 펼침 크기를 보존하고 통합 진단과 상태를 공유한다.
from dataclasses import asdict
import logging
from pathlib import Path

import pyqtgraph as pg
from PySide6.QtCore import QObject, QTimer
from PySide6.QtWidgets import (QCheckBox, QDialog, QFileDialog, QHBoxLayout, QLabel, QMessageBox,
                              QProgressBar, QPushButton, QWidget)
from app.analysis.coach_models import CoachReview, ENGINE_VERSION, event_key, now, review_key
from app.analysis.coach_rules import DEFAULT_RULES, load_rules
from app.analysis.coach_store import CoachStore, export_reports
from app.core.coach_worker import CoachWorker
from app.core.frame_processor import cache_key
from app.core.video_timeline import video_identity
from app.ui.coach_panel import CoachPanel, ReviewDialog

log = logging.getLogger(__name__)


class CoachController(QObject):
    def __init__(self, window, settings_path, layout):
        super().__init__(window)
        self.w = window
        self.store = CoachStore(Path(settings_path).parent / "coaching")
        self.reports, self.reviews, self.confirmations, self.history = {}, {}, {}, []
        self.live, self.workers, self.markers = {}, [], []
        self.evidence_range_active = False
        self.epoch, self.loaded_identity, self.dirty = 0, None, False
        self.signature = None
        self.last_error = ""
        self.last_panel_size = window.settings.value("coach/lastHeight", 280, type=int)
        self.display_has_report = None
        self.rules_path = Path(window.settings.value("coach/rulesPath", str(DEFAULT_RULES)))
        self.rules, self.rules_stat = None, None
        self.load_rules(quiet=True)
        self.controls = QWidget()
        row = QHBoxLayout(self.controls)
        row.setContentsMargins(0, 0, 0, 0)
        self.panel_check = QCheckBox("AI 코칭 패널 ON/OFF")
        self.panel_check.setChecked(window.settings.value("coach/showPanel", False, type=bool))
        self.analyze_button = QPushButton("AI 코칭 분석")
        self.cancel_button = QPushButton("코칭 분석 취소")
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(100)
        self.status = QLabel("로컬 관찰 규칙 V1 · 기술 점수 없음")
        for control in (self.panel_check, self.analyze_button, self.cancel_button, self.progress, self.status):
            row.addWidget(control)
        layout.insertWidget(4, self.controls)
        previous_sizes = window.content_splitter.sizes()
        self.panel = CoachPanel(window)
        window.content_splitter.addWidget(self.panel)
        self.panel.setVisible(self.panel_check.isChecked())
        split = window.settings.value("coach/splitter")
        if split:
            window.content_splitter.restoreState(split)
        else:
            window.content_splitter.setSizes([*previous_sizes, 0 if not self.panel_check.isChecked() else 200])
        self.panel_check.toggled.connect(self.show_panel)
        self.analyze_button.clicked.connect(lambda: self.run(False))
        self.panel.selected_button.clicked.connect(lambda: self.run(False))
        self.panel.all_button.clicked.connect(lambda: self.run(True))
        self.cancel_button.clicked.connect(self.cancel_requested)
        self.panel.message_selected.connect(self.show_evidence)
        self.panel.evidence_button.clicked.connect(self.play_evidence)
        self.panel.loop_check.toggled.connect(self.loop_changed)
        self.panel.confirm_check.toggled.connect(self.confirm_event)
        self.panel.compare_check.toggled.connect(self.options_changed)
        self.panel.group_edit.textChanged.connect(self.options_changed)
        self.panel.save_button.clicked.connect(self.choose_export)
        self.panel.reset_button.clicked.connect(self.reset_results)
        self.panel.review_button.clicked.connect(self.edit_review)
        self.panel.rules_button.clicked.connect(self.choose_rules)
        window.motion.changed.connect(self.source_changed)
        window.motion.selection_changed.connect(self.selected_event_changed)
        for button in (window.play, window.previous, window.next, window.motion.panel.play_button):
            button.clicked.connect(self.clear_replay_owner)
        window.player.playing_changed.connect(lambda playing: self.clear_replay_owner() if not playing else None)
        self.rule_timer = QTimer(self)
        self.rule_timer.setInterval(1000)
        self.rule_timer.timeout.connect(self.check_rules)
        self.rule_timer.start()
        self.source_changed()

    def show_panel(self, visible):
        if not visible and self.panel.isVisible() and self.panel.report:
            self.last_panel_size = self.w.content_splitter.sizes()[-1]
        self.panel.setVisible(visible)
        if visible:
            sizes = self.w.content_splitter.sizes()
            sizes[-1] = self.last_panel_size if self.panel.report else 75
            self.w.content_splitter.setSizes(sizes)

    def selected_event(self):
        return self.w.motion.panel.selected_event()

    def current_signature(self):
        m = self.w.motion
        document = m.document
        return (self.w.video_revision, self.w.selection_revision, self.w.measurement_revision,
                id(self.w.cache.measurements), id(self.w.cache.frames), id(self.w.cache.selections),
                document, m.historical, tuple(item for item in m.change_reasons() if item[0] != "scope"), m.hand_combo.currentData(), self.rules.key if self.rules else None,
                tuple(sorted(self.confirmations.items())), self.panel.compare_check.isChecked(), self.panel.group_edit.text())

    def available(self):
        m = self.w.motion
        context = m.current_context()
        key = self.w.cache.key
        source_valid = bool(self.w.timeline and key and tuple(key[:3]) == tuple(self.w.timeline.identity))
        workflow = getattr(self.w, "workflow", None)
        if workflow and workflow.workers:
            return False
        if workflow and workflow.asset_signature == workflow.asset_key() and workflow.asset_valid is False:
            source_valid = False
        return (self.rules is not None and m.document is not None and bool(m.document.events)
                and source_valid
                and self.w.timeline is not None and self.loaded_identity == tuple(self.w.timeline.identity)
                and m.current_ready() and context is not None
                and context["context_id"] == m.document.automatic.context_id
                and context["player_id"] == m.document.automatic.player_id
                and m.hand_combo.currentData() == m.document.automatic.racket_hand
                and not self.w.measurement_pending and not self.w.tracking_pending
                and self.w.analysis_worker is None and not m.workers)

    def source_changed(self):
        identity = tuple(self.w.timeline.identity) if self.w.timeline else None
        if identity != self.loaded_identity:
            # clear_selection은 영상 시간축 교체 전에 호출된다. 저장 성공 전에는 버리지 않는다.
            if self.loaded_identity is not None and self.dirty and not self.save():
                return
            self.reports, self.reviews, self.confirmations, self.history = {}, {}, {}, []
            self.live.clear()
            self.loaded_identity = identity
            self.last_error = ""
            self.loaded_timeline = self.w.timeline
            if identity:
                try:
                    self.reports, self.reviews, self.confirmations, self.history = self.store.load(self.w.timeline)
                    log.info("코칭 저장 결과 읽기: %s reports=%s (재검증 전 과거 기록)", identity[0], len(self.reports))
                except Exception as exc:
                    self.show_error("코칭 기록 읽기 실패: " + str(exc))
                    # 손상된 기존 파일을 빈 문서로 자동 덮어쓰지 않는다.
                    self.loaded_identity = None
        signature = self.current_signature()
        if self.signature != signature:
            self.epoch += 1
            self.cancel()
            self.signature = signature
            self.clear_markers()
            if self.evidence_range_active:
                self.w.player.request("clear_range")
                self.evidence_range_active = False
        self.refresh()

    def selected_event_changed(self, event_id):
        self.clear_replay_owner()
        self.clear_markers()
        self.refresh()

    def refresh(self):
        event = self.selected_event()
        report = self.reports.get(event.event_id) if event else None
        stale = self.is_stale(report)
        self.panel.confirm_check.blockSignals(True)
        self.panel.confirm_check.setChecked(bool(event and self.confirmations.get(event.event_id) == event_key(event)))
        self.panel.confirm_check.blockSignals(False)
        self.panel.confirm_check.setEnabled(event is not None)
        self.panel.set_report(report, self.reviews, stale)
        has_report = report is not None
        if self.display_has_report != has_report and self.panel_check.isChecked():
            sizes = self.w.content_splitter.sizes()
            if not has_report and self.display_has_report:
                self.last_panel_size = sizes[-1]
            sizes[-1] = max(240, self.last_panel_size) if has_report else 75
            self.w.content_splitter.setSizes(sizes)
        self.display_has_report = has_report
        self.update_controls()
        if hasattr(self.w, "workflow"):
            self.w.workflow.refresh()

    def is_stale(self, report):
        return (report is None or not self.available() or self.live.get(report.event_id) != self.current_signature()
                or report.rules_key != self.rules.key or report.engine_version != ENGINE_VERSION)

    def update_controls(self):
        busy, available = bool(self.workers), self.available()
        event = self.selected_event()
        valid = event is not None
        if event and self.w.timeline:
            from app.analysis.motion_models import validate_event
            try:
                validate_event(event, self.w.timeline, self.w.motion.document.events)
            except (ValueError, TypeError):
                valid = False
        self.analyze_button.setEnabled(available and valid and not busy)
        self.panel.selected_button.setEnabled(self.analyze_button.isEnabled())
        self.panel.all_button.setEnabled(available and not busy)
        self.cancel_button.setEnabled(busy)
        self.panel.save_button.setEnabled(bool(self.reports))
        self.panel.reset_button.setEnabled(bool(self.reports) and not busy)
        message = self.panel.selected_message()
        report = self.reports.get(message.event_id) if message else None
        usable = bool(message and not self.is_stale(report) and report.quality.integrity_ok)
        self.panel.evidence_button.setEnabled(usable)
        self.panel.review_button.setEnabled(message is not None)
        if not busy:
            self.status.setText("현재 스윙 코칭 실행 가능" if available and valid else "스윙을 선택하세요 · 실행 조건 확인" if available else "실행 조건 확인에서 원인·해결 방법 보기")

    def options_changed(self):
        self.source_changed()

    def confirm_event(self, confirmed):
        event = self.selected_event()
        if event is None:
            return
        self.confirmations[event.event_id] = event_key(event) if confirmed else ""
        self.history.append({"action": "event_confirmation", "event_id": event.event_id, "confirmed": confirmed, "at": now()})
        self.dirty = True
        log.info("코칭용 사용자 스윙/단계 확인: event=%s confirmed=%s", event.event_id, confirmed)
        self.save()
        self.source_changed()

    def load_rules(self, quiet=False):
        try:
            rules = load_rules(self.rules_path)
            stat = self.rules_path.stat()
            self.rules, self.rules_stat = rules, (stat.st_mtime_ns, stat.st_size)
            log.info("코칭 규칙 설정: path=%s version=%s key=%s", self.rules_path, rules.version, rules.key)
            return True
        except Exception as exc:
            self.rules = None
            try:
                stat = self.rules_path.stat()
                self.rules_stat = (stat.st_mtime_ns, stat.st_size)
            except OSError:
                self.rules_stat = None
            log.exception("코칭 규칙 로딩 실패")
            if not quiet:
                self.show_error("규칙 설정 오류: " + str(exc))
            return False

    def check_rules(self):
        try:
            stat = self.rules_path.stat()
            changed = (stat.st_mtime_ns, stat.st_size) != self.rules_stat
        except OSError:
            changed = self.rules is not None
        if changed:
            self.load_rules(quiet=True)
            self.source_changed()

    def choose_rules(self):
        path, _ = QFileDialog.getOpenFileName(self.w, "코칭 규칙 설정", str(self.rules_path), "JSON (*.json)")
        if path:
            self.rules_path = Path(path)
            self.load_rules()
            self.w.settings.setValue("coach/rulesPath", str(self.rules_path))
            self.source_changed()

    def run(self, all_events=False):
        self.check_rules()
        self.source_changed()
        if not self.available() or self.workers or (not all_events and self.selected_event() is None):
            return
        try:
            if video_identity(self.w.video_path) != self.w.timeline.identity:
                raise ValueError("영상이 변경되었습니다. 다시 분석하세요.")
            if cache_key(self.w.video_path) != self.w.cache.key:
                raise ValueError("영상·모델 또는 관절 분석 설정이 바뀌었습니다. 실행 조건 확인에서 관절 캐시를 갱신하세요.")
        except Exception as exc:
            self.live.clear()
            self.show_error(str(exc))
            if hasattr(self.w, "workflow"):
                flow = self.w.workflow
                flow.asset_signature, flow.asset_valid = flow.asset_key(), False
                flow.asset_reason, flow.asset_action = str(exc), "pose"
                try:
                    if video_identity(self.w.video_path) != self.w.timeline.identity:
                        flow.asset_action = "open"
                except OSError:
                    flow.asset_action = "open"
            self.refresh()
            return
        self.epoch += 1
        token, signature = self.epoch, self.current_signature()
        self.last_error = ""
        ids = None if all_events else {self.selected_event().event_id}
        worker = CoachWorker(self.w.motion.document, self.w.cache.frames, self.w.cache.selections,
                             self.w.cache.measurements, self.w.timeline, self.rules, self.confirmations,
                             ids, self.panel.compare_check.isChecked(), self.panel.group_edit.text(), self.reports, self)
        self.workers.append(worker)
        worker.progress.connect(lambda value: self.progress.setValue(value) if token == self.epoch else None)
        worker.ready.connect(lambda reports: self.accept(token, signature, reports))
        worker.error.connect(lambda message: self.show_error(message) if token == self.epoch else None)
        worker.finished.connect(lambda: self.finished(worker))
        self.progress.setValue(0)
        self.status.setText("로컬 코칭 분석 중 — 기존 캐시 사용")
        self.panel_check.setChecked(True)
        worker.start()
        self.update_controls()

    def accept(self, token, signature, reports):
        if token != self.epoch or signature != self.current_signature():
            return
        for report in reports:
            previous = self.reports.get(report.event_id)
            if previous and previous.input_key != report.input_key:
                self.history.append({"action": "reanalyze", "at": now(), "previous_report": asdict(previous)})
            self.reports[report.event_id] = report
            self.live[report.event_id] = signature
        self.dirty = True
        self.save()
        self.refresh()

    def finished(self, worker):
        if worker in self.workers:
            self.workers.remove(worker)
        worker.deleteLater()
        self.update_controls()
        if hasattr(self.w, "workflow"):
            self.w.workflow.refresh(force=True)

    def cancel(self):
        for worker in self.workers:
            worker.requestInterruption()

    def cancel_requested(self):
        self.epoch += 1
        self.cancel()
        log.info("코칭 분석 취소 요청")

    def clear_markers(self):
        for plot, marker in self.markers:
            plot.removeItem(marker)
        self.markers.clear()

    def evidence_current(self, message):
        report = self.reports.get(message.event_id)
        event = self.selected_event()
        if (event is None or event.event_id != message.event_id or self.is_stale(report)
                or not report.quality.integrity_ok or report.event_fingerprint != event_key(event)):
            return False
        try:
            if video_identity(self.w.video_path) != self.w.timeline.identity:
                return False
        except OSError:
            return False
        return all(not self.w.cache.measurements.get(i) or
                   (self.w.cache.measurements[i].player_id == message.player_id
                    and self.w.cache.measurements[i].track_segment == message.track_segment)
                   for i in range(message.start_frame, message.end_frame + 1))

    def show_evidence(self, message):
        self.update_controls()
        if not self.evidence_current(message):
            return
        # 현재 코칭은 이미 선택된 스윙에 속한다. 같은 스윙을 재선택하면
        # 마우스 이벤트 처리 도중 코칭 목록까지 다시 만들어 Qt 항목이 무효화된다.
        self.w.motion.panel.highlight_selected()
        self.clear_markers()
        a, b = message.start_frame, message.end_frame
        self.w.panel_check.setChecked(True)
        self.w.measurement_panel.focus_event((a, b))
        for metric in message.graph_metrics:
            if metric in self.w.measurement_panel.checks:
                self.w.measurement_panel.checks[metric].setChecked(True)
        start, end = self.w.timeline.seconds[a], self.w.timeline.frame_end(b)
        self.w.measurement_panel.plot.setXRange(start, end, padding=0.03)
        for plot in (self.w.measurement_panel.plot, self.w.motion.panel.timeline_plot):
            region = pg.LinearRegionItem((start, end), movable=False, brush=(255, 220, 60, 30), pen=pg.mkPen("#ffd43b"))
            region.setZValue(-5)
            region.setToolTip("코칭 근거 범위 — A/B 통계와 독립")
            plot.addItem(region)
            self.markers.append((plot, region))
        self.w.player.request("seek", a)

    def play_evidence(self):
        message = self.panel.selected_message()
        if message and self.evidence_current(message):
            self.show_evidence(message)
            self.w.player.request("play_range", (message.start_frame, message.end_frame, self.panel.loop_check.isChecked()))
            self.evidence_range_active = True

    def loop_changed(self, enabled):
        if self.evidence_range_active:
            self.w.player.request("range_loop", enabled)

    def clear_replay_owner(self):
        self.evidence_range_active = False

    def edit_review(self):
        message = self.panel.selected_message()
        if message is None:
            return
        review = self.reviews.get(review_key(message.event_id, message.rule_id), CoachReview())
        dialog = ReviewDialog(review, self.w)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.apply_review(message, dialog.state.currentData(), dialog.note.toPlainText(), dialog.reason.text(), dialog.hidden.isChecked())

    def apply_review(self, message, status, note="", error_reason="", hidden=False):
        from app.analysis.coach_models import REVIEW_STATES
        if status not in REVIEW_STATES or len(note) > 10000 or len(error_reason) > 2000 or (status == "error" and not error_reason.strip()):
            raise ValueError("사용자 검토 상태/메모/오류 이유가 올바르지 않습니다.")
        report = self.reports[message.event_id]
        key = review_key(message.event_id, message.rule_id)
        previous = self.reviews.get(key, CoachReview())
        history = (*previous.history, {name: value for name, value in asdict(previous).items() if name != "history"})
        self.reviews[key] = CoachReview(status, note, error_reason, hidden, report.input_key, now(), history)
        self.dirty = True
        log.info("코칭 사용자 검토 변경: event=%s rule=%s status=%s hidden=%s", message.event_id, message.rule_id, status, hidden)
        self.save()
        self.refresh()

    def save(self):
        if not self.dirty:
            return True
        if self.loaded_identity is None:
            self.show_error("기존 코칭 기록을 읽지 못해 덮어쓰기를 보류했습니다. 원본 JSON을 확인하세요.")
            return False
        try:
            path = self.store.save(self.loaded_timeline, self.reports, self.reviews, self.confirmations, self.history)
            self.dirty = False
            log.info("코칭 결과/검토 저장: %s", path)
            return True
        except Exception as exc:
            self.show_error("코칭 저장 실패: " + str(exc))
            return False

    def choose_export(self):
        path, _ = QFileDialog.getSaveFileName(self.w, "코칭 결과 저장/내보내기", "coaching_results.json", "JSON (*.json);;CSV (*.csv)")
        if path:
            try:
                self.export(path)
                self.w.statusBar().showMessage("코칭 내보내기 완료: " + path)
            except Exception as exc:
                self.show_error("코칭 내보내기 실패: " + str(exc))

    def export(self, path):
        stale = {key for key, report in self.reports.items() if self.is_stale(report)}
        export_reports(path, self.reports.values(), self.reviews, stale)

    def reset_results(self):
        if QMessageBox.question(self.w, "코칭 결과 초기화", "현재 영상의 코칭 결과를 초기화합니까?\n검토/메모와 이전 결과는 이력에 보존합니다.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
            return
        self.epoch += 1
        self.cancel()
        self.history.append({"action": "reset", "at": now(), "reports": {key: asdict(report) for key, report in self.reports.items()}})
        self.reports.clear()
        self.live.clear()
        self.clear_markers()
        self.dirty = True
        self.save()
        self.refresh()

    def show_error(self, message):
        self.last_error = message
        log.error("코칭 기능 오류: %s", message)
        self.w.statusBar().showMessage(message)

    def prepare_close(self):
        self.rule_timer.stop()
        self.epoch += 1
        self.cancel()
        for worker in self.workers:
            if not worker.wait(5000):
                self.w.statusBar().showMessage("코칭 작업 종료 중입니다. 잠시 후 다시 닫아 주세요.")
                self.rule_timer.start()
                return False
        if not self.save():
            self.rule_timer.start()
            return False
        self.w.settings.setValue("coach/showPanel", self.panel_check.isChecked())
        self.w.settings.setValue("coach/rulesPath", str(self.rules_path))
        self.w.settings.setValue("coach/splitter", self.w.content_splitter.saveState())
        if self.panel.report and self.panel.isVisible():
            self.last_panel_size = self.w.content_splitter.sizes()[-1]
        self.w.settings.setValue("coach/lastHeight", self.last_panel_size)
        return True
