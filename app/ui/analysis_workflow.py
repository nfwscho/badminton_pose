# 기존 분석 컨트롤을 재사용하는 통합 상태/실행 조건 진단 및 단계별 다음 작업 안내.
# 자동 확인 가능한 캐시·선수·규칙 조건과 사용자가 확인할 선수 연결/스윙 검토를 분리한다.
# 무거운 원본·품질 검사는 QThread에서 수행하고 결과를 현재 입력 세대에만 적용한다.
# 원인 코드/변경 내용/항목별 규칙 결과를 한글 진단 창에 표시하며 기존 로그·수동 자료를 보존한다.
import logging
from PySide6.QtCore import QObject, QTimer, Qt
from PySide6.QtWidgets import (QDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton, QTreeWidget,
                              QTreeWidgetItem, QVBoxLayout, QWidget)
from app.analysis.analysis_readiness import diagnose
from app.analysis.coach_models import event_key
from app.analysis.motion_models import validate_event
from app.core.readiness_worker import ReadinessWorker
from app.core.video_timeline import video_identity

log = logging.getLogger(__name__)


class AnalysisWorkflow(QObject):
    def __init__(self, window, layout):
        super().__init__(window)
        self.w, self.workers, self.previews = window, [], {}
        self.signature, self.asset_signature, self.asset_valid = None, None, None
        self.asset_reason, self.last_error = "", ""
        self.asset_action = "check"
        self.link_rejected = False
        self.last_log, self.dialog, self.tree = None, None, None
        self.area = QWidget()
        vertical = QVBoxLayout(self.area)
        vertical.setContentsMargins(0, 0, 0, 0)
        self.steps_label, self.summary_label = QLabel(), QLabel()
        for label in (self.steps_label, self.summary_label):
            label.setWordWrap(True)
            label.setTextFormat(Qt.TextFormat.PlainText)
            vertical.addWidget(label)
        row = QHBoxLayout()
        self.check_button, self.next_button, self.review_button = (QPushButton(text) for text in
            ("분석 상태 확인", "다음 단계 실행", "검토 완료"))
        for button in (self.check_button, self.next_button, self.review_button):
            row.addWidget(button)
        row.addStretch()
        vertical.addLayout(row)
        layout.insertWidget(1, self.area)
        self.conditions_button = QPushButton("실행 조건 확인")
        window.coach.controls.layout().insertWidget(2, self.conditions_button)
        self.conditions_button.setToolTip("코칭 버튼이 비활성화되어도 원인과 해결 방법을 확인할 수 있습니다.")
        self.check_button.clicked.connect(self.inspect)
        self.conditions_button.clicked.connect(self.inspect)
        self.next_button.clicked.connect(self.advance)
        self.review_button.clicked.connect(self.mark_reviewed)
        self.timer = QTimer(self)
        self.timer.setInterval(350)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()
        self.refresh(force=True)

    def input_signature(self):
        w = self.w
        return (w.coach.current_signature(), w.video_path, w.cache.run_id, w.cache.state, len(w.cache.frames),
                w.analysis_worker is not None, w.tracking_pending, w.measurement_pending, bool(w.motion.workers),
                w.motion.panel.selected_id, bool(w.coach.workers), w.coach.loaded_identity, w.coach.last_error, w.motion.last_error)

    def asset_key(self):
        w = self.w
        return (w.video_path, w.cache.key, w.cache.run_id, w.timeline.identity if w.timeline else None)

    def state(self):
        w, m, c = self.w, self.w.motion, self.w.coach
        context, event = m.current_context(), c.selected_event()
        identity = tuple(w.timeline.identity) if w.timeline else None
        key = w.cache.key
        # 원본 stat 확인은 상태 변화/사용자 진단 때만 수행하며 모델 해시는 작업 스레드에서 검사한다.
        asset_valid = bool(key and identity and tuple(key[:3]) == identity)
        if self.asset_signature == self.asset_key() and self.asset_valid is not None:
            asset_valid = asset_valid and self.asset_valid
        selected = bool(w.cache.anchor and w.cache.selections)
        event_valid, event_reason = event is not None, "목록에서 코칭할 스윙을 선택하세요."
        if event and w.timeline:
            try:
                validate_event(event, w.timeline, m.document.events)
            except Exception:
                event_valid, event_reason = False, "선택 스윙의 경계/PTS가 현재 영상과 다릅니다. 스윙을 수정하거나 다시 검출하세요."
        linked = bool(m.connection_ready())
        report = c.reports.get(event.event_id) if event else None
        preview = self.previews.get(event.event_id) if event else None
        current = bool(report and not c.is_stale(report))
        quality = report.quality if current else preview.quality if preview else None
        changes = list(m.change_reasons())
        if report and event:
            if report.event_fingerprint != event_key(event):
                changes.append(("coaching_event", "스윙 경계/단계/라벨 수정 — 코칭 결과만 재계산 필요"))
            if c.rules and report.rules_key != c.rules.key:
                changes.append(("coaching_rules", "코칭 규칙 버전/내용 변경 — 코칭 결과만 재계산 필요"))
        # 코칭 변경은 기존 스윙 실행을 막지 않는다. 해당 결과의 재계산 이유는 별도 표시한다.
        source_changes = tuple(item for item in changes if not item[0].startswith("coaching_"))
        self.coach_changes = tuple(item for item in changes if item[0].startswith("coaching_"))
        busy = w.analysis_worker is not None or w.tracking_pending or w.measurement_pending or bool(m.workers) or bool(c.workers) or bool(self.workers)
        return dict(video=bool(w.video_path and w.original is not None), timeline=w.timeline is not None,
                    timeline_error=w.timeline_status.text().startswith("시간축 오류"), pose_error=w.cache.state == "error",
                    coach_error=c.last_error, motion_error=m.last_error,
                    pose=bool(w.cache.frames) and w.cache.state in ("complete", "cancelled", "error"),
                    pose_count=len(w.cache.frames), pose_busy=w.analysis_worker is not None,
                    pose_reason="관절 분석 진행 중입니다." if w.analysis_worker else "현재 영상의 관절 분석이 없습니다. 관절 분석을 시작하세요.",
                    asset_valid=asset_valid, asset_reason=self.asset_reason or "영상·모델과 관절 캐시의 연결을 확인하세요.",
                    asset_action=self.asset_action if self.asset_signature == self.asset_key() else "check",
                    selected=selected, measurements=bool(w.cache.measurements) and not w.measurement_pending and not w.tracking_pending,
                    measured_count=sum(m.analyzable for m in w.cache.measurements.values()), measurement_busy=w.tracking_pending or w.measurement_pending,
                    hand=m.hand_combo.currentData() is not None, events=bool(m.document and m.document.events),
                    motion_busy=bool(m.workers), linked=linked, changes=source_changes,
                    context_mismatch=bool(context and m.document and context["athlete_key"] != m.document.automatic.athlete_key),
                    link_rejected=self.link_rejected,
                    link_reason=self.last_error if self.link_rejected else "저장된 스윙과 현재 분석 실행의 연결을 아직 확인하지 않았습니다. 영상·기준 프레임·몸통 위치·추적 구간을 검사한 뒤 직접 확인하세요." if m.historical else
                                "선수/관절 실행 또는 설정이 기존 스윙과 다릅니다. 변경 내용과 선수 연결을 확인하세요.",
                    event_valid=event_valid, event_reason=event_reason,
                    reviewed=bool(event and c.confirmations.get(event.event_id) == event_key(event)), rules=c.rules is not None,
                    storage_ready=bool(identity and c.loaded_identity == identity),
                    quality=quality, busy=busy, coach_busy=bool(c.workers), has_report=report is not None,
                    report_current=current, saved=not c.dirty, save_error=c.dirty and bool(self.last_error))

    def refresh(self, force=False):
        signature = self.input_signature()
        if not force and self.signature == signature:
            return
        if self.signature is not None and self.signature[0] != signature[0]:
            self.previews.clear()
            self.last_error = ""
            self.link_rejected = False
        self.signature = signature
        state = self.state()
        self.readiness = diagnose(state)
        r = self.readiness
        self.steps_label.setText("AI 분석 상태  |  " + " → ".join(f"{i + 1}. {name}: {status}" for i, (name, status) in enumerate(r.steps)))
        completed = sum(status == "완료" for _, status in r.steps)
        suffix = "; ".join(reason for _, reason in self.coach_changes)
        self.summary_label.setText(f"전체 작업 {completed}/7단계 완료 | 캐시: {r.cache_state} | {r.summary}" + (" | " + suffix if suffix else "") +
                                   (" | 최근 진단: " + self.last_error if self.last_error else ""))
        self.next_button.setText(self.next_action()[1])
        self.next_button.setEnabled(not state["busy"])
        self.review_button.setEnabled(state["event_valid"] and state["linked"] and not state["busy"])
        self.check_button.setEnabled(True)
        self.conditions_button.setEnabled(True)
        blocked = tuple(item.code for item in r.checks if item.blocking)
        log_key = (blocked, r.cache_state, state["reviewed"], tuple(state["changes"]), tuple(self.coach_changes))
        if log_key != self.last_log:
            log.info("분석 준비 상태 변경: disabled_codes=%s cache=%s reviewed=%s changes=%s coaching_changes=%s",
                     blocked, r.cache_state, state["reviewed"], state["changes"], self.coach_changes)
            self.last_log = log_key
        self.w.coach.analyze_button.setToolTip(r.summary)
        self.w.coach.panel.selected_button.setToolTip(r.summary)
        if self.dialog is not None:
            self.populate_dialog()

    def next_action(self):
        blocked = next((item for item in self.readiness.checks if item.blocking), None)
        if blocked:
            # 손/설정 변경은 선수 연결 재승인으로 해소되지 않는다. 앞선 기본 입력이
            # 준비되었다면 해당 스윙 갱신으로 안내한다(수동 교체 확인은 그대로 유지).
            if blocked.code == "PLAYER_LINK_REQUIRED" and any(code in ("hand", "config") for code, _ in self.state()["changes"]):
                return "refresh", "영향받는 스윙 갱신"
            return blocked.action, blocked.action_label
        if not self.state()["reviewed"]:
            return "review", "다음: 스윙 검토"
        if self.state()["report_current"]:
            return "results", "다음: 코칭 결과 확인·저장"
        return "coach", "다음: 선택 스윙 코칭"

    def inspect(self):
        self.w.coach.source_changed()
        self.refresh(force=True)
        self.show_dialog()
        self.start_check(False)

    def show_dialog(self):
        if self.dialog is None:
            self.dialog = QDialog(self.w)
            self.dialog.setWindowTitle("분석 상태 진단 — 이유와 해결 방법")
            self.dialog.resize(1050, 560)
            layout = QVBoxLayout(self.dialog)
            label = QLabel("자동 데이터 검사와 선수 연결 확인·스윙 검토는 별도입니다. 항목을 선택하고 해결 작업을 누르세요.")
            label.setWordWrap(True)
            layout.addWidget(label)
            self.tree = QTreeWidget()
            self.tree.setColumnCount(4)
            self.tree.setHeaderLabels(("검사 항목", "상태", "이유", "해결 방법"))
            self.tree.setColumnWidth(0, 155)
            self.tree.setColumnWidth(1, 100)
            self.tree.setColumnWidth(2, 490)
            layout.addWidget(self.tree, 1)
            row = QHBoxLayout()
            for text, action in (("선택 항목 해결", self.resolve_selected), ("다시 검사", lambda: self.start_check(False)), ("닫기", self.dialog.hide)):
                button = QPushButton(text)
                button.clicked.connect(action)
                row.addWidget(button)
            layout.addLayout(row)
        self.populate_dialog()
        self.dialog.show()
        self.dialog.raise_()

    def populate_dialog(self):
        previous = self.tree.currentItem().data(0, Qt.ItemDataRole.UserRole) if self.tree.currentItem() else None
        self.tree.clear()
        for item in self.readiness.checks:
            row = QTreeWidgetItem((item.title, item.status, item.reason, item.action_label))
            row.setData(0, Qt.ItemDataRole.UserRole, item.code)
            row.setData(1, Qt.ItemDataRole.UserRole, item.action)
            row.setToolTip(0, item.code)
            row.setToolTip(2, item.reason)
            self.tree.addTopLevelItem(row)
            if item.code == previous:
                self.tree.setCurrentItem(row)
        for code, reason in self.coach_changes:
            self.tree.addTopLevelItem(QTreeWidgetItem(("코칭 캐시 변경", "재계산 필요", reason, "선택 스윙 코칭")))
        event = self.w.coach.selected_event()
        report = self.previews.get(event.event_id) if event else None
        historical_rules = False
        if report is None and event:
            report = self.w.coach.reports.get(event.event_id)
            historical_rules = bool(report and self.w.coach.is_stale(report))
        if report:
            for message in report.messages:
                row = QTreeWidgetItem(("규칙: " + message.rule_name, "과거 기록·재검사 필요" if historical_rules else "평가 보류" if message.deferred_reasons else "관찰 가능",
                                      "; ".join(message.deferred_reasons) or message.observation, "근거 영상/측정 탭에서 확인"))
                row.setData(1, Qt.ItemDataRole.UserRole, "details")
                row.setToolTip(2, row.text(2))
                self.tree.addTopLevelItem(row)
        if self.last_error:
            self.tree.addTopLevelItem(QTreeWidgetItem(("최근 진단 오류", "오류", self.last_error, "입력을 확인하고 다시 검사")))

    def resolve_selected(self):
        if self.tree.currentItem():
            self.perform(self.tree.currentItem().data(1, Qt.ItemDataRole.UserRole) or "details")

    def advance(self):
        self.refresh(force=True)
        self.perform(self.next_action()[0])

    def perform(self, action):
        w = self.w
        if action == "open":
            w.choose_file()
        elif action == "pose":
            if not self.state()["pose"] or not self.state()["asset_valid"]:
                w.start_analysis()
        elif action == "player":
            w.begin_selection()
        elif action == "hand":
            w.motion.hand_combo.setFocus()
            w.motion.hand_combo.showPopup()
        elif action == "detect":
            w.motion.run_analysis()
        elif action == "link":
            self.start_check(True)
        elif action == "refresh":
            # 재검출의 수동 변경 확인은 기존 run_analysis/confirm_replace에서 수행한다.
            w.motion.run_analysis()
        elif action in ("swing", "review"):
            w.motion.panel_check.setChecked(True)
            if w.motion.document and w.motion.document.events:
                event = w.coach.selected_event() or w.motion.document.events[0]
                w.motion.select_event(event.event_id)
                w.motion.panel.highlight_selected()
                w.player.request("seek", event.start_frame)
                w.statusBar().showMessage("영상을 검토한 뒤 상단 '검토 완료'를 누르세요. 선수 연결·품질 검사와 별도입니다.")
        elif action == "coach":
            w.coach.run(False)
        elif action == "results":
            w.coach.panel_check.setChecked(True)
        elif action in ("check", "details"):
            w.coach.panel_check.setChecked(True)
            self.inspect()
        elif action == "rules":
            w.coach.choose_rules()
        self.refresh(force=True)

    def mark_reviewed(self):
        self.refresh(force=True)
        if self.state()["event_valid"] and self.state()["linked"]:
            self.w.coach.confirm_event(True)
            self.refresh(force=True)
            self.start_check(False)

    def start_check(self, link=False):
        w = self.w
        if self.workers or w.timeline is None or not w.video_path:
            return
        token = self.input_signature()
        worker = ReadinessWorker(w.video_path, w.cache, w.timeline, w.motion.document, w.motion.current_context(),
                                 w.motion.config, w.coach.rules, w.coach.confirmations, link, self)
        self.workers.append(worker)
        worker.ready.connect(lambda result: self.checked(token, result, link))
        worker.error.connect(lambda message: self.failed(token, message))
        worker.finished.connect(lambda: self.finished(worker))
        worker.start()
        self.refresh(force=True)

    def checked(self, token, result, link):
        if token != self.input_signature():
            return
        self.asset_signature, self.asset_valid = self.asset_key(), result["asset_valid"]
        self.asset_reason = result["asset_reason"]
        self.asset_action = result["asset_action"]
        self.previews = {report.event_id: report for report in result["reports"]}
        self.last_error = ""
        if link:
            proposal = result["proposal"]
            if proposal is None:
                self.last_error = "선수 연결 불가: " + "; ".join(result["link_reasons"] or (self.asset_reason,))
                self.link_rejected = True
                log.warning("선수 연결 검증 거부: %s", self.last_error)
            else:
                source = self.w.motion.document.automatic
                text = (f"동일 영상·기준 프레임 {source.source_anchor['frame']}·몸통 위치와 이벤트별 연속 추적을 확인했습니다.\n"
                        f"저장 세션 {source.player_id} → 현재 선택 세션 {proposal.automatic.player_id}\n"
                        "영상에 표시된 선수가 저장했던 동일 선수인지 직접 확인했습니까?\n"
                        "이벤트 ID·수동 경계·단계·라벨은 보존하며 수치만 현재 캐시로 갱신합니다.\n"
                        "이 확인은 스윙/단계 검토나 데이터 품질 승인을 대신하지 않습니다.")
                if self.confirm_link(text) and token == self.input_signature():
                    motion = self.w.motion
                    motion.document, motion.features = proposal, result["features"]
                    motion.historical, motion.stale, motion.unsaved = False, False, True
                    motion.save_document()
                    motion.refresh()
                    log.info("선수 연결 사용자 확인 완료: context=%s session=%s", proposal.automatic.context_id, proposal.automatic.player_id)
                else:
                    log.info("선수 연결 사용자 확인 취소 — 원본 유지")
        self.refresh(force=True)
        for item in self.readiness.checks:
            log.info("코칭 준비 검사: code=%s state=%s reason=%s action=%s", item.code, item.status, item.reason, item.action)

    def confirm_link(self, text):
        return QMessageBox.question(self.w, "선수 연결 직접 확인", text, QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                    QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes

    def failed(self, token, message):
        if token == self.input_signature():
            self.last_error = "진단을 완료하지 못했습니다. 영상과 저장 파일 상태를 확인하세요. " + message
            log.error("분석 준비 검사 오류: %s", message)
            self.refresh(force=True)

    def finished(self, worker):
        if worker in self.workers:
            self.workers.remove(worker)
        worker.deleteLater()
        self.w.coach.update_controls()
        self.refresh(force=True)

    def prepare_close(self):
        self.timer.stop()
        for worker in self.workers:
            worker.requestInterruption()
        for worker in self.workers:
            if not worker.wait(5000):
                self.timer.start()
                return False
        if self.dialog:
            self.dialog.close()
        return True
