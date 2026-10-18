# 관절 원시 측정값/통계 테이블과 pyqtgraph 시계열 표시 패널.
# 결측은 NaN으로 끊고 데이터 갱신은 선택/구간/표시 설정 변경 때만 수행한다.
# 재생 중에는 시간 위치선과 현재 값만 갱신한다. 그래프 클릭을 영상 탐색으로
# 전달하며 항목별 표시, 확대/축소, 범위 초기화와 원시/표시 평활화 선택을 제공한다.
# 이벤트 선택 시 그래프 표시 범위만 별도 전환할 수 있어 기존 A/B 통계를 바꾸지 않는다.
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QHBoxLayout, QLabel, QPushButton, QSplitter,
                             QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)
from app.analysis.joint_angles import METRICS, measurement_bounds, summarize

COLORS = ("#24b47e", "#3182f6", "#dd9933", "#cc55cc", "#66cccc", "#aa77ff", "#dddddd")


class MeasurementPanel(QWidget):
    time_clicked = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.measurements, self.timeline, self.bounds = {}, None, None
        self.reload_count = 0
        self.current_index = 0
        self.display_bounds = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        controls = QHBoxLayout()
        self.a_button, self.b_button, self.clear_range_button = (QPushButton(t) for t in ("구간 시작 A", "구간 종료 B", "구간 초기화"))
        self.smooth_check = QCheckBox("표시용 3프레임 평활화")
        self.reset_button, self.csv_button = QPushButton("그래프 초기화"), QPushButton("CSV 내보내기")
        for widget in (self.a_button, self.b_button, self.clear_range_button, self.smooth_check, self.reset_button, self.csv_button):
            controls.addWidget(widget)
        layout.addLayout(controls)
        self.range_label = QLabel("구간: 전체 추적 범위 | 분석 가능: 0프레임")
        self.range_label.setWordWrap(True)
        layout.addWidget(self.range_label)
        checks = QHBoxLayout()
        self.checks, self.curves = {}, {}
        self.plot = pg.PlotWidget(background="#20242c")
        self.plot.setLabel("bottom", "영상 시간", units="s")
        self.plot.setLabel("left", "영상 평면 각도", units="°")
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.setMinimumSize(100, 70)
        for (key, (label, _)), color in zip(list(METRICS.items())[:7], COLORS):
            check = QCheckBox(label.replace("(화면)", ""))
            check.setChecked(key in ("left_elbow", "right_elbow", "left_knee", "right_knee"))
            check.setStyleSheet(f"QCheckBox {{ color: {'#555555' if key == 'trunk_tilt' else color}; }}")
            self.checks[key] = check
            checks.addWidget(check)
            curve = self.plot.plot([], [], pen=pg.mkPen(color, width=1.5), connect="finite", name=label)
            curve.setVisible(check.isChecked())
            check.toggled.connect(curve.setVisible)
            self.curves[key] = curve
        layout.addLayout(checks)
        self.cursor = pg.InfiniteLine(pos=0, angle=90, movable=False, pen=pg.mkPen("#ffffff", width=1))
        self.plot.addItem(self.cursor, ignoreBounds=True)
        self.plot.scene().sigMouseClicked.connect(self.on_click)
        self.table = QTableWidget(len(METRICS), 6)
        self.table.setHorizontalHeaderLabels(("항목", "현재(원시)", "최소", "최대", "변화폭", "유효 수"))
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setDefaultSectionSize(22)
        self.table.setColumnWidth(0, 170)
        for column in range(1, 6):
            self.table.setColumnWidth(column, 70)
        self.table.setMinimumSize(200, 70)
        self.table.setToolTip("모든 통계는 원시값 기준. 각도는 도, 손목 높이는 몸통길이 단위. 변화폭 = 최대 - 최소.")
        for row, (_, (label, unit)) in enumerate(METRICS.items()):
            self.table.setItem(row, 0, QTableWidgetItem(f"{label} ({unit})"))
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(self.table)
        self.splitter.addWidget(self.plot)
        self.splitter.setSizes([440, 600])
        layout.addWidget(self.splitter, 1)
        self.reset_button.clicked.connect(self.reset_view)
        self.smooth_check.toggled.connect(self.reload_curves)
        self.smooth_check.setToolTip("그래프와 각도 오버레이에만 적용합니다. 현재 값·통계·CSV 원시 열은 원본 측정값입니다.")
        self.range_label.setToolTip("A와 B 프레임을 모두 포함합니다. 길이는 B 프레임의 표시 종료까지입니다. 분석 가능은 9개 측정 중 하나 이상 유효한 프레임 수이며 항목별 유효 수는 표에서 확인하세요.")

    def set_data(self, measurements, timeline, a=None, b=None):
        self.measurements, self.timeline = measurements, timeline
        self.display_bounds = None
        self.bounds = measurement_bounds(measurements, len(timeline.seconds), a, b) if timeline else None
        self.reload_curves()
        stats = summarize(measurements, timeline, self.bounds) if timeline else None
        for row, name in enumerate(METRICS):
            metric = stats["metrics"][name] if stats else {"count": 0, "min": None, "max": None, "range": None}
            for column, key in ((2, "min"), (3, "max"), (4, "range")):
                self.table.setItem(row, column, QTableWidgetItem(self.format_value(metric[key])))
            self.table.setItem(row, 5, QTableWidgetItem(str(metric["count"])))
        if timeline:
            duration = f"{stats['duration']:.3f}초" if stats["duration"] is not None else "측정 불가"
            ratio = f"{stats['ratio']:.1%}" if stats["ratio"] is not None else "측정 불가"
            scope = (f"A {timeline.seconds[a]:.3f}s" if a is not None else "A 미지정") + " / " + (f"B {timeline.seconds[b]:.3f}s" if b is not None else "B 미지정")
            if a is None and b is None:
                scope = "전체 추적 범위"
            else:
                scope += " (해당 프레임 포함)"
            self.range_label.setText(f"구간: {scope} | 길이 {duration} | 분석 가능 {stats['usable_frames']}/{stats['total_frames']} 프레임 ({ratio})")
        else:
            self.range_label.setText("구간: 영상 없음 | 분석 가능: 0프레임")
        self.update_current(self.current_index)

    @staticmethod
    def format_value(value):
        return f"{value:.1f}" if value is not None else "측정 불가"

    def update_current(self, index):
        self.current_index = index
        measurement = self.measurements.get(index)
        for row, name in enumerate(METRICS):
            metric = measurement.values[name] if measurement else None
            self.table.setItem(row, 1, QTableWidgetItem(self.format_value(metric.raw if metric and metric.valid else None)))
        if self.timeline and index < len(self.timeline.seconds):
            self.cursor.setValue(self.timeline.seconds[index])

    def reload_curves(self):
        self.reload_count += 1
        times = np.asarray(self.timeline.seconds) if self.timeline else np.array([])
        plot_bounds = self.display_bounds if self.display_bounds is not None else self.bounds
        for name, curve in self.curves.items():
            values = np.full(len(times), np.nan)
            for index, measurement in self.measurements.items():
                if index >= len(times) or not plot_bounds or not plot_bounds[0] <= index <= plot_bounds[1]:
                    continue
                metric = measurement.values[name]
                if metric.valid:
                    values[index] = metric.display if self.smooth_check.isChecked() else metric.raw
            # NaN뿐 아니라 서로 다른 추적 세션/연속 구간 사이도 연결하지 않는다.
            connections = np.zeros(len(times), dtype=bool)
            for i in range(max(0, len(times) - 1)):
                a, b = self.measurements.get(i), self.measurements.get(i + 1)
                connections[i] = (np.isfinite(values[i]) and np.isfinite(values[i + 1]) and a is not None and b is not None
                                  and a.track_segment is not None and a.track_segment == b.track_segment)
            curve.setData(times, values, connect=connections)

    def focus_event(self, bounds):
        self.display_bounds = bounds
        self.reload_curves()

    def reset_view(self):
        if self.timeline:
            start, end = self.bounds if self.bounds else (0, len(self.timeline.seconds) - 1)
            self.plot.setXRange(self.timeline.seconds[start], self.timeline.frame_end(end), padding=0.02)
        self.plot.setYRange(0, 180, padding=0.03)

    def on_click(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.plot.getViewBox().sceneBoundingRect().contains(event.scenePos()):
            self.time_clicked.emit(float(self.plot.getViewBox().mapSceneToView(event.scenePos()).x()))
