# 원본 해상도의 QPixmap 복사본에만 관절을 그리는 표시 모듈과 클릭 영상 위젯.
# 좌/우 관절 색상, 선택 선수의 별도 팔레트/테두리, 신뢰도 필터를 제공한다.
# 원본은 변경하지 않으며 표시 크기와 레터박스 여백을 클릭 좌표에 역변환한다.
# 선택 선수의 유효한 팔꿈치/무릎/어깨 각도를 별도 복사본에 표시하며 숫자
# 배치를 충돌 검사한다. 각도/골격 토글은 독립적이며 결측 수치는 그리지 않는다.
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QLabel

from app.core.pose_detector import CONNECTIONS


def image_rect(widget_width, widget_height, image_width, image_height):
    if min(widget_width, widget_height, image_width, image_height) <= 0:
        return QRectF()
    scale = min(widget_width / image_width, widget_height / image_height)
    width, height = image_width * scale, image_height * scale
    return QRectF((widget_width - width) / 2, (widget_height - height) / 2, width, height)


class VideoLabel(QLabel):
    image_clicked = Signal(float, float)
    resized = Signal()

    def __init__(self, text):
        super().__init__(text)
        self.source_size = (0, 0)
        self.selecting = False

    def mousePressEvent(self, event):
        if self.selecting and event.button() == Qt.MouseButton.LeftButton:
            rect = image_rect(self.width(), self.height(), *self.source_size)
            if not rect.isEmpty() and rect.contains(event.position()):
                self.image_clicked.emit((event.position().x() - rect.x()) / rect.width(),
                                        (event.position().y() - rect.y()) / rect.height())
        super().mousePressEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.resized.emit()


def draw_overlay(original, frame, selected_index=None):
    canvas = original.copy()
    painter = QPainter(canvas)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    width, height = canvas.width(), canvas.height()
    unit = max(1.0, min(width, height) / 400)
    for i, pose in enumerate(frame.poses):
        selected = i == selected_index

        def color(joint):
            left = joint.name.startswith("left") or joint.name.endswith("left")
            right = joint.name.startswith("right") or joint.name.endswith("right")
            if selected:
                return QColor("#ffff00" if left else "#ff40ff" if right else "#ffffff")
            return QColor("#32cd80" if left else "#38aaff" if right else "#cccccc")

        def point(joint):
            return QPointF(joint.x * (width - 1), joint.y * (height - 1))

        for a, b in CONNECTIONS:
            ja, jb = pose.joints[a], pose.joints[b]
            if ja.reliable and jb.reliable:
                painter.setPen(QPen(color(ja), unit * (2.2 if selected else 1.3)))
                painter.drawLine(point(ja), point(jb))
        for joint in pose.joints:
            if joint.reliable:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(color(joint))
                painter.drawEllipse(point(joint), unit * 2, unit * 2)
        if selected and pose.bbox:
            x1, y1, x2, y2 = pose.bbox
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor("#ffdd00"), unit * 1.5, Qt.PenStyle.DashLine))
            painter.drawRect(QRectF(x1 * width, y1 * height, (x2 - x1) * width, (y2 - y1) * height))
    painter.end()
    return canvas


def draw_angle_overlay(original, pose, measurement, smoothed=False):
    canvas = original.copy()
    painter = QPainter(canvas)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    unit = max(1.0, min(canvas.width(), canvas.height()) / 420)
    font = QFont()
    font.setPixelSize(round(11 * unit))
    painter.setFont(font)
    occupied = []
    for key, joint_index, label in (("left_elbow", 13, "L 팔꿈치"), ("right_elbow", 14, "R 팔꿈치"),
                                    ("left_knee", 25, "L 무릎"), ("right_knee", 26, "R 무릎"),
                                    ("left_shoulder", 11, "L 어깨"), ("right_shoulder", 12, "R 어깨")):
        metric, joint = measurement.values[key], pose.joints[joint_index]
        if not metric.valid or not joint.reliable:
            continue
        value = metric.display if smoothed else metric.raw
        text = f"{label} {value:.1f}°"
        size = painter.fontMetrics().boundingRect(text)
        width, height = size.width() + 8 * unit, size.height() + 4 * unit
        x, y = joint.x * (canvas.width() - 1), joint.y * (canvas.height() - 1)
        candidates = []
        for dx, dy in ((8 * unit, -height), (-width - 8 * unit, -height),
                       (8 * unit, 5 * unit), (-width - 8 * unit, 5 * unit),
                       (8 * unit, -height * 2), (8 * unit, height)):
            candidates.append(QRectF(max(0, min(canvas.width() - width, x + dx)),
                                     max(0, min(canvas.height() - height, y + dy)), width, height))
        box = next((r for r in candidates if not any(r.intersects(o) for o in occupied)), candidates[0])
        occupied.append(box)
        painter.fillRect(box, QColor(0, 0, 0, 190))
        painter.setPen(QColor("#ffff77" if key.startswith("left") else "#ff99ff"))
        painter.drawText(box.adjusted(4 * unit, 0, -4 * unit, 0), Qt.AlignmentFlag.AlignVCenter, text)
    painter.end()
    return canvas
