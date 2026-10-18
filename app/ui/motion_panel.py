# 스윙 목록/상세/PTS 타임라인 및 입력식 수동 편집·분석 설정 대화상자.
# 원본 자동 결과는 읽기 전용으로 두며 편집 대화상자의 적용에서만 새 이벤트를
# 전달한다. 타임라인은 기존 pyqtgraph를 재사용하고 재생 중 현재 위치선만 갱신한다.
# 표시 라벨은 후보/보류와 불확실성을 명시하며 구질·임팩트·자세 점수를 표현하지 않는다.
from dataclasses import asdict
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
                              QFormLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QPlainTextEdit,
                              QPushButton, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem,
                              QTabWidget, QVBoxLayout, QWidget)
from app.analysis.motion_models import KINDS, PHASES, PHASE_LABELS, MotionConfig, phase_span

PHASE_COLORS = {"READY": "#5b8ff9", "BACKSWING": "#a978e8", "FORWARD_SWING": "#ff9d3f",
                "FOLLOW_THROUGH": "#52bb7c", "RECOVERY": "#48c7ce", "UNKNOWN_PHASE": "#888888"}
KIND_LABELS = {"OVERHEAD_CANDIDATE": "오버헤드 후보", "OTHER_SWING": "기타 스윙 후보", "UNKNOWN": "판단 보류", "NO_SWING": "스윙 아님"}


class MotionSettingsDialog(QDialog):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("스윙 검출 설정")
        self.config, self.result_config = config, None
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.fields = {}
        for name, label, lower, upper, step in (("sensitivity", "민감도 (클수록 민감)", 0.3, 3, 0.1),
                                               ("min_duration", "최소 후보 지속 시간(초)", 0.05, 3, 0.02),
                                               ("merge_gap", "활동 구간 병합 간격(초)", 0, 0.5, 0.02),
                                               ("max_time_gap", "연속 관측 최대 간격(초)", 0.03, 1, 0.01)):
            spin = QDoubleSpinBox()
            spin.setRange(lower, upper)
            spin.setSingleStep(step)
            spin.setValue(getattr(config, name))
            self.fields[name] = spin
            form.addRow(label, spin)
        self.method = QComboBox()
        self.method.addItem("분석용 중앙값(median)", "median")
        self.method.addItem("분석용 평활화 없음", "none")
        self.method.setCurrentIndex(self.method.findData(config.smoothing))
        form.addRow("분석용 평활화", self.method)
        self.window = QSpinBox()
        self.window.setRange(1, 15)
        self.window.setSingleStep(2)
        self.window.setValue(config.window)
        form.addRow("창 크기(홀수 프레임)", self.window)
        layout.addLayout(form)
        note = QLabel("표시용 3프레임 평활화와 독립적입니다. 결측/추적/시간 단절을 넘지 않습니다.\n고립 좌표 튐의 7점 품질 검사는 평활화 선택과 별개로 적용합니다.\n규칙 임계값은 촬영 조건별 검증이 필요합니다.")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.error_label = QLabel()
        layout.addWidget(self.error_label)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.apply)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def apply(self):
        try:
            values = asdict(self.config)
            values.update({name: spin.value() for name, spin in self.fields.items()})
            values.update(smoothing=self.method.currentData(), window=self.window.value())
            self.result_config = MotionConfig(**values)
            self.accept()
        except ValueError as exc:
            self.error_label.setText(str(exc))


class MotionEditDialog(QDialog):
    def __init__(self, timeline, factory, event=None, default_range=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("스윙 후보 수동 검토 / 수정")
        self.timeline, self.factory, self.result_event = timeline, factory, None
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.start, self.end = QSpinBox(), QSpinBox()
        for spin in (self.start, self.end):
            spin.setRange(0, len(timeline.seconds) - 1)
        a, b = (event.start_frame, event.end_frame) if event else default_range or (0, len(timeline.seconds) - 1)
        self.start.setValue(a)
        self.end.setValue(b)
        self.kind = QComboBox()
        for code in KINDS:
            self.kind.addItem(KIND_LABELS[code], code)
        self.kind.setCurrentIndex(self.kind.findData(event.kind if event else "UNKNOWN"))
        self.user_label = QLineEdit(event.user_label if event else "")
        self.user_label.setMaxLength(80)
        form.addRow("시작 프레임(포함)", self.start)
        form.addRow("종료 프레임(포함)", self.end)
        form.addRow("동작 후보 라벨", self.kind)
        form.addRow("사용자 라벨(선택)", self.user_label)
        layout.addLayout(form)
        self.time_label = QLabel()
        self.start.valueChanged.connect(self.show_times)
        self.end.valueChanged.connect(self.show_times)
        layout.addWidget(self.time_label)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(("단계", "시작 프레임", "종료 프레임"))
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)
        phase_controls = QHBoxLayout()
        add, remove, reset = (QPushButton(label) for label in ("단계 추가", "단계 삭제", "단계 초기화(보류)"))
        for button in (add, remove, reset):
            phase_controls.addWidget(button)
        layout.addLayout(phase_controls)
        add.clicked.connect(lambda: self.add_phase("UNKNOWN_PHASE", self.start.value(), self.end.value()))
        remove.clicked.connect(lambda: self.table.removeRow(self.table.currentRow()) if self.table.currentRow() >= 0 else None)
        reset.clicked.connect(self.reset_phases)
        for phase in event.phases if event else ():
            self.add_phase(phase.phase, phase.start_frame, phase.end_frame)
        self.error_label = QLabel("단계는 겹치지 않고 순서대로 지정하세요. 빈 구간은 UNKNOWN_PHASE로 남깁니다.")
        self.error_label.setWordWrap(True)
        layout.addWidget(self.error_label)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("적용")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("취소")
        buttons.accepted.connect(self.apply)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.resize(620, 520)
        self.show_times()

    def show_times(self):
        self.time_label.setText(f"실제 시간: {self.timeline.seconds[self.start.value()]:.3f} ~ {self.timeline.frame_end(self.end.value()):.3f}초")

    def add_phase(self, name, start, end):
        row = self.table.rowCount()
        self.table.insertRow(row)
        combo = QComboBox()
        for phase in (*PHASES, "UNKNOWN_PHASE"):
            combo.addItem(PHASE_LABELS[phase], phase)
        combo.setCurrentIndex(combo.findData(name))
        self.table.setCellWidget(row, 0, combo)
        for column, value in ((1, start), (2, end)):
            spin = QSpinBox()
            spin.setRange(0, len(self.timeline.seconds) - 1)
            spin.setValue(value)
            self.table.setCellWidget(row, column, spin)

    def reset_phases(self):
        self.table.setRowCount(0)
        self.add_phase("UNKNOWN_PHASE", self.start.value(), self.end.value())

    def apply(self):
        try:
            spans = [phase_span(self.table.cellWidget(row, 0).currentData(),
                                self.table.cellWidget(row, 1).value(), self.table.cellWidget(row, 2).value(),
                                self.timeline, "사용자 경계 지정") for row in range(self.table.rowCount())]
            self.result_event = self.factory(self.start.value(), self.end.value(), self.kind.currentData(), spans, self.user_label.text())
            self.accept()
        except (ValueError, IndexError) as exc:
            self.error_label.setText(str(exc))


class MotionPanel(QWidget):
    event_selected = Signal(str)
    time_clicked = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.document, self.timeline, self.selected_id = None, None, None
        self.regions, self.highlight = [], None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        tools = QHBoxLayout()
        labels = ("선택 스윙 재생", "이벤트→A/B", "수정", "추가", "삭제", "전체 자동 결과 복원", "저장", "불러오기", "시간축 초기화")
        self.play_button, self.ab_button, self.edit_button, self.add_button, self.delete_button, self.restore_button, self.save_button, self.load_button, self.reset_button = (QPushButton(t) for t in labels)
        self.loop_check = QCheckBox("스윙 구간 반복")
        for widget in (self.play_button, self.loop_check, self.ab_button, self.edit_button, self.add_button,
                       self.delete_button, self.restore_button, self.save_button, self.load_button, self.reset_button):
            tools.addWidget(widget)
        layout.addLayout(tools)
        self.status_label = QLabel("선수 선택·관절 측정 후 라켓 손을 지정하고 스윙 자동 검출을 실행하세요.")
        self.status_label.setWordWrap(True)
        self.status_label.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.status_label)
        self.tabs = QTabWidget()
        list_widget = QSplitter(Qt.Orientation.Horizontal)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(("순번", "시작(초)", "종료(초)", "지속(초)", "동작 후보", "단계 상태", "유효 비율"))
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.setMinimumSize(200, 80)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setMinimumSize(120, 80)
        list_widget.addWidget(self.table)
        list_widget.addWidget(self.details)
        list_widget.setSizes([690, 350])
        self.tabs.addTab(list_widget, "스윙 목록 / 상세")
        self.timeline_plot = pg.PlotWidget(background="#20242c")
        self.timeline_plot.setLabel("bottom", "원본 영상 시간", units="s")
        self.timeline_plot.hideAxis("left")
        self.timeline_plot.setYRange(0, 1, padding=0)
        self.timeline_plot.setMouseEnabled(x=True, y=False)
        self.timeline_plot.setMinimumSize(100, 80)
        self.cursor = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#ffffff", width=1.5))
        self.timeline_plot.addItem(self.cursor, ignoreBounds=True)
        self.timeline_plot.scene().sigMouseClicked.connect(self.on_click)
        timeline_widget = QWidget()
        tl = QVBoxLayout(timeline_widget)
        tl.setContentsMargins(0, 0, 0, 0)
        legend = QLabel("  ".join(f'<span style="color:{color}">■ {PHASE_LABELS[name]}</span>' for name, color in PHASE_COLORS.items()) + "  ▒ 관측 불가")
        tl.addWidget(legend)
        tl.addWidget(self.timeline_plot)
        self.tabs.addTab(timeline_widget, "동작 타임라인")
        layout.addWidget(self.tabs, 1)
        self.table.itemSelectionChanged.connect(self.row_selected)

    def set_document(self, document, timeline, historical=False, stale=False):
        self.document, self.timeline = document, timeline
        self.table.blockSignals(True)
        self.table.setRowCount(len(document.events) if document else 0)
        if document:
            events = sorted(document.events, key=lambda event: event.start_frame)
            for row, event in enumerate(events):
                known = sum(p.phase != "UNKNOWN_PHASE" for p in event.phases)
                phase_state = f"{known}단계 후보" if known else "단계 보류"
                if event.manually_modified:
                    phase_state += " / 수동"
                if event.clipped:
                    phase_state += " / 잘림"
                values = (str(row + 1), f"{event.start_seconds:.3f}", f"{event.end_seconds:.3f}", f"{event.duration:.3f}",
                          KIND_LABELS[event.kind] + (" · " + event.user_label if event.user_label else ""), phase_state,
                          f"{event.valid_ratio:.1%}" if event.valid_ratio is not None else "측정 불가")
                for column, text in enumerate(values):
                    item = QTableWidgetItem(text)
                    item.setData(Qt.ItemDataRole.UserRole, event.event_id)
                    item.setToolTip("\n".join((*event.evidence, *event.uncertainty)))
                    self.table.setItem(row, column, item)
                if event.event_id == self.selected_id:
                    self.table.selectRow(row)
            observed = sum(c.end_frame - c.start_frame + 1 for c in document.automatic.coverage if c.kind == "NO_SWING")
            usable = sum(c.end_frame - c.start_frame + 1 for c in document.automatic.coverage
                         if c.kind == "NO_SWING" or c.reason == "swing_candidate")
            state = (f"스윙 후보 {len(events)}개 — 모두 수동 검토 필요" if events else "스윙 후보 없음 (사용자 수정 결과)"
                     if document.modified else "스윙 후보 없음" if observed else "스윙 판단 보류 — 유효한 연속 관측 부족")
            if historical:
                state += " | 저장된 과거 관측: 현재 선수 캐시와 연결 미확인"
            if stale:
                state += " | 입력/설정 변경: 재검출 필요"
            total = document.automatic.scope[1] - document.automatic.scope[0] + 1
            self.status_label.setText(state + f" | 시계열 유효 {usable}/{total}프레임 | {document.automatic.racket_hand} / {document.automatic.algorithm_version}")
        else:
            self.status_label.setText("미분석 — 선수 선택·측정과 라켓 손 지정 후 자동 검출하세요.")
            self.details.clear()
        self.table.blockSignals(False)
        self.rebuild_timeline()
        self.show_details()

    def selected_event(self):
        return next((event for event in self.document.events if event.event_id == self.selected_id), None) if self.document else None

    def row_selected(self):
        row = self.table.currentRow()
        item = self.table.item(row, 0) if row >= 0 else None
        if item:
            self.selected_id = item.data(Qt.ItemDataRole.UserRole)
            self.show_details()
            self.highlight_selected()
            self.event_selected.emit(self.selected_id)

    def show_details(self):
        event = self.selected_event()
        if event is None:
            self.details.setPlainText("스윙을 선택하면 수치/근거/불확실성을 표시합니다.")
            return
        def number(value, unit=""):
            return f"{value:.3f}{unit}" if value is not None else "측정 불가"
        lines = [f"{KIND_LABELS[event.kind]} / {event.source}", f"프레임 {event.start_frame}~{event.end_frame} (포함)",
                 f"지속 시간 {event.duration:.3f}초", f"선수 키 {event.athlete_key} / 세션 {event.player_id}",
                 "손목 상대 이동량: " + number(event.details.get("wrist_path_torso_lengths"), " 몸통길이"),
                 "팔꿈치 최소/최대: " + number(event.details.get("elbow_min"), "°") + " / " + number(event.details.get("elbow_max"), "°"),
                 "어깨 최소/최대: " + number(event.details.get("shoulder_min"), "°") + " / " + number(event.details.get("shoulder_max"), "°"),
                 "전방 단계 팔꿈치 변화: " + number(event.details.get("forward_elbow_change"), "°"),
                 "움직임 강도 피크(임팩트 아님): " + number(event.details.get("peak_energy_seconds"), "초")]
        lines.extend(f"{PHASE_LABELS.get(name, name)}: {duration:.3f}초" for name, duration in event.details.get("phase_durations", {}).items())
        lines += ["근거:", *event.evidence, "불확실성:", *event.uncertainty]
        self.details.setPlainText("\n".join(lines))

    def rebuild_timeline(self):
        for region in self.regions:
            self.timeline_plot.removeItem(region)
        self.regions.clear()
        if self.document and self.timeline:
            for interval in self.document.automatic.coverage:
                if interval.kind == "UNKNOWN" and interval.reason != "swing_candidate":
                    self.add_region(self.timeline.seconds[interval.start_frame], self.timeline.frame_end(interval.end_frame),
                                    "#777777", (0.02, 0.17), "관측 불가: " + interval.reason)
            for event in self.document.events:
                self.add_region(event.start_seconds, event.end_seconds, "#6a778c", (0.2, 0.95), KIND_LABELS[event.kind])
                for phase in event.phases:
                    self.add_region(phase.start_seconds, phase.end_seconds, PHASE_COLORS[phase.phase], (0.28, 0.85),
                                    PHASE_LABELS[phase.phase] + ": " + phase.evidence)
        self.highlight_selected()

    def add_region(self, start, end, color, span, tooltip):
        brush = pg.mkColor(color)
        brush.setAlpha(110)
        region = pg.LinearRegionItem((start, end), movable=False, brush=brush, pen=pg.mkPen(color), span=span)
        region.setToolTip(tooltip)
        region.setZValue(-10)
        self.timeline_plot.addItem(region)
        self.regions.append(region)

    def highlight_selected(self):
        if self.highlight is not None:
            self.timeline_plot.removeItem(self.highlight)
            self.highlight = None
        event = self.selected_event()
        if event:
            self.highlight = pg.LinearRegionItem((event.start_seconds, event.end_seconds), movable=False,
                                                 brush=(0, 0, 0, 0), pen=pg.mkPen("#ffdd33", width=2), span=(0.18, 0.98))
            self.highlight.setZValue(-5)
            self.timeline_plot.addItem(self.highlight)

    def on_click(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.timeline_plot.getViewBox().sceneBoundingRect().contains(event.scenePos()):
            self.time_clicked.emit(float(self.timeline_plot.getViewBox().mapSceneToView(event.scenePos()).x()))
