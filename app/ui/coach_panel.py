# 기존 스윙/그래프를 유지하는 접이식 코칭 패널과 사용자 검토 대화상자.
# 분석 상태→실제 관찰→잠정 검토→촬영 제안 순으로 표시하며 기본 최대 3개만 노출한다.
# 추가 결과/숨긴 결과 펼치기, 단계별 실제 특징, 조건부 반복 비교 및 검토 메모를 제공한다.
# 모든 생성/파일/영상 작업은 컨트롤러에 위임하고 사용자 문자열은 일반 텍스트로 표시한다.
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout,
                              QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QPushButton,
                              QSplitter, QTabWidget, QTableWidget, QTableWidgetItem, QTextBrowser,
                              QTextEdit, QVBoxLayout, QWidget)
from app.analysis.coach_models import CoachReview, feature_label, review_key

STATE_LABELS = {"COACHING_DEFERRED": "평가 보류", "MEASUREMENT_ONLY": "측정 관찰만 가능",
                "COACHING_AVAILABLE": "코칭 가능"}
REVIEW_LABELS = {"unreviewed": "미검토", "confirmed": "확인", "deferred": "판단 보류", "error": "분석 오류"}


class ReviewDialog(QDialog):
    def __init__(self, review, parent=None):
        super().__init__(parent)
        self.setWindowTitle("코칭 결과 사용자 검토 — 모델 학습은 수행하지 않음")
        self.resize(560, 380)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.state = QComboBox()
        for key, label in REVIEW_LABELS.items():
            self.state.addItem(label, key)
        self.state.setCurrentIndex(self.state.findData(review.status))
        self.note = QTextEdit()
        self.note.setPlainText(review.note)
        self.reason = QLineEdit(review.error_reason)
        self.reason.setMaxLength(2000)
        self.hidden = QCheckBox("이 결과 숨김 (숨긴 결과 보기에서 복원 가능)")
        self.hidden.setChecked(review.hidden)
        form.addRow("검토 상태", self.state)
        form.addRow("메모 (최대 10000자)", self.note)
        form.addRow("분석 오류 이유", self.reason)
        form.addRow(self.hidden)
        layout.addLayout(form)
        self.error = QLabel()
        self.error.setWordWrap(True)
        layout.addWidget(self.error)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self.apply)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def apply(self):
        if len(self.note.toPlainText()) > 10000:
            self.error.setText("메모는 10000자 이하여야 합니다.")
        elif self.state.currentData() == "error" and not self.reason.text().strip():
            self.error.setText("분석 오류의 이유를 입력하세요.")
        else:
            self.accept()


class CoachPanel(QWidget):
    message_selected = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.report, self.reviews, self.stale = None, {}, True
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self.selected_button, self.all_button, self.evidence_button = (QPushButton(s) for s in
            ("선택 스윙 코칭", "전체 스윙 코칭", "근거 영상 보기"))
        self.loop_check = QCheckBox("근거 구간 반복 재생")
        self.save_button, self.reset_button, self.review_button = (QPushButton(s) for s in
            ("코칭 결과 저장", "코칭 결과 초기화", "확인·보류·오류·메모"))
        for control in (self.selected_button, self.all_button, self.evidence_button, self.loop_check,
                        self.review_button, self.save_button, self.reset_button):
            row.addWidget(control)
        layout.addLayout(row)
        row = QHBoxLayout()
        self.confirm_check = QCheckBox("현재 스윙·단계를 직접 확인함")
        self.confirm_check.setToolTip("현재 이벤트 경계와 관측된 단계를 확인합니다. 기술의 옳고 그름을 검증하는 기능은 아닙니다.")
        self.compare_check = QCheckBox("동일 고정 촬영·동일 기술 확인")
        self.group_edit = QLineEdit()
        self.group_edit.setPlaceholderText("비교 기술 그룹 (사용자 지정)")
        self.group_edit.setMaxLength(80)
        self.more_check, self.hidden_check = QCheckBox("나머지 펼치기"), QCheckBox("숨긴 결과 보기")
        self.rules_button = QPushButton("규칙 설정 불러오기")
        for control in (self.confirm_check, self.compare_check, self.group_edit, self.more_check,
                        self.hidden_check, self.rules_button):
            row.addWidget(control)
        layout.addLayout(row)
        self.status = QLabel("A. 분석 상태: 미분석 — 관절/스윙 캐시와 사용자 확인 필요")
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.status)
        self.tabs = QTabWidget()
        self.tabs.setMinimumHeight(100)
        split = QSplitter(Qt.Orientation.Horizontal)
        self.list = QListWidget()
        self.list.setMinimumWidth(140)
        self.details = QTextBrowser()
        self.details.setOpenExternalLinks(False)
        split.addWidget(self.list)
        split.addWidget(self.details)
        split.setSizes([280, 600])
        self.tabs.addTab(split, "관찰 및 교정 검토 포인트")
        self.features = QTextBrowser()
        self.tabs.addTab(self.features, "단계별 측정 근거")
        self.comparison = QTableWidget(0, 5)
        self.comparison.setHorizontalHeaderLabels(("항목", "현재", "기준", "차이", "단위/비교 불가 사유"))
        self.comparison.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.comparison.horizontalHeader().setStretchLastSection(True)
        self.tabs.addTab(self.comparison, "반복 스윙 관찰 차이")
        layout.addWidget(self.tabs, 1)
        self.setMinimumHeight(240)
        self.list.currentItemChanged.connect(self.on_selection)
        self.more_check.toggled.connect(self.populate)
        self.hidden_check.toggled.connect(self.populate)

    def selected_message(self):
        item = self.list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def set_report(self, report, reviews, stale):
        self.report, self.reviews, self.stale = report, reviews, stale
        if report is None:
            self.status.setText("A. 분석 상태: 선택 스윙 코칭 결과 없음")
            self.features.clear()
            self.comparison.setRowCount(0)
        else:
            q = report.quality
            detection = "관측 충분" if q.detection == "SUFFICIENT" else "관측 부족/무효"
            phase = {"UNKNOWN": "미확인", "PARTIAL_CANDIDATE": "일부 보류", "CANDIDATE": "자동 후보",
                     "USER_REVIEWED_CANDIDATE": "사용자 확인 후보"}.get(q.phase, q.phase)
            self.status.setText(f"A. 분석 상태: {STATE_LABELS[q.state]}" +
                (" | 오래된/저장된 결과 — 재계산 필요" if stale else "") +
                f" | {detection}: {q.valid_frames}/{q.total_frames} 유효 관측 (정확도 확률 아님)"
                f" | 단계 {phase} | 관찰 규칙만 적용\n" + "; ".join(q.reasons))
            lines = ["실제 캐시의 측정·파생값. 단계 이름은 후보이며 기술 정답 판정이 아닙니다.",
                     "극값의 peak_frame은 관측된 최소/최대 위치이며 타격 순간이 아닙니다."]
            for name, o in report.features.items():
                value = f"{o.value:.5g} {o.unit}" if o.valid else "계산 불가: " + o.reason
                lines.append(f"{feature_label(name)}: {value} | {o.level} | frames {o.start_frame}–{o.end_frame}" +
                             (f" | peak_frame {o.peak_frame}" if o.peak_frame is not None else ""))
            self.features.setPlainText("\n".join(lines))
            comp = report.comparison
            rows = list(comp.get("differences", {}).items())
            self.comparison.setRowCount(max(len(rows), 1))
            if not rows:
                self.comparison.setItem(0, 0, QTableWidgetItem("비교 보류"))
                self.comparison.setItem(0, 4, QTableWidgetItem("; ".join(comp.get("reasons", []))))
            for row, (key, value) in enumerate(rows):
                cells = (feature_label(key), *(f"{value[name]:.5g}" if value.get("valid") else "" for name in ("current", "reference", "difference")),
                         value.get("unit", value.get("reason", "")))
                for col, cell in enumerate(cells):
                    self.comparison.setItem(row, col, QTableWidgetItem(cell))
            self.comparison.setToolTip("기준 이벤트: " + str(comp.get("reference_event_id")) + " — 점수/우수도 없음")
        self.populate()

    def populate(self):
        previous = self.selected_message()
        self.list.blockSignals(True)
        self.list.clear()
        messages = list(self.report.messages) if self.report else []
        messages = [m for m in messages if self.hidden_check.isChecked() or not self.reviews.get(review_key(m.event_id, m.rule_id), CoachReview()).hidden]
        messages.sort(key=lambda m: (m.status == "COACHING_DEFERRED", m.rule_id == "quality_notice"))
        if not self.more_check.isChecked():
            # 실제 관찰 또는 품질 안내만 우선 노출한다. 미관측 단계의 보류 항목으로
            # 핵심 3개를 억지로 채우지 않으며 전체 보류 사유는 펼쳐서 확인할 수 있다.
            messages = [m for m in messages if m.rule_id == "quality_notice" or any(o.valid for o in m.measurements)]
            messages = messages[:3]
        selected = 0
        for row, message in enumerate(messages):
            review = self.reviews.get(review_key(message.event_id, message.rule_id), CoachReview())
            review_status = REVIEW_LABELS[review.status]
            if review.reviewed_input_key and review.reviewed_input_key != self.report.input_key:
                review_status += " (이전 근거 검토·재확인 필요)"
            item = QListWidgetItem(f"{message.rule_name or message.rule_id}\n{message.level} · {review_status}" + (" · 숨김" if review.hidden else ""))
            item.setData(Qt.ItemDataRole.UserRole, message)
            self.list.addItem(item)
            if previous and previous.result_id == message.result_id:
                selected = row
        if messages:
            self.list.setCurrentRow(selected)
        self.list.blockSignals(False)
        self.show_detail(self.selected_message())

    def show_detail(self, message):
        if message is None:
            self.details.setPlainText("표시할 결과가 없습니다. 나머지/숨긴 결과를 펼치거나 분석을 실행하세요.")
            return
        review = self.reviews.get(review_key(message.event_id, message.rule_id), CoachReview())
        self.details.setPlainText(f"B. 주요 관찰\n{message.observation}\n\nC. 코칭 검토 포인트\n"
            f"{message.interpretation}\n\nD. 연습 및 촬영 제안\n{message.suggestion}\n\n"
            f"근거 프레임 {message.start_frame}–{message.end_frame}, 원본 PTS {message.start_pts}–{message.end_pts}\n"
            f"규칙 {message.rule_id} v{message.rule_version} | {message.level}\n"
            f"사용자 메모: {review.note}\n분석 오류 이유: {review.error_reason}")

    def on_selection(self, item, previous):
        message = self.selected_message()
        self.show_detail(message)
        if message:
            self.message_selected.emit(message)
