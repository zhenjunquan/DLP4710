"""
Custom UI widgets for focus assistant: interactive video canvas with ROI selection,
dynamic focus curve plotter, and real-time status banner.
"""

from typing import List, Optional, Tuple
import numpy as np
from PyQt6.QtCore import QPoint, QRect, Qt, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
)
from PyQt6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

from dlp_common.focus.focus_tracker import FocusState


class InteractiveVideoCanvas(QLabel):
    """
    Video display widget that supports mouse-drag ROI selection,
    coordinate mapping, and optional center crosshairs.
    """

    roi_selected = pyqtSignal(int, int, int, int)  # (x, y, w, h) in image coordinates
    roi_cleared = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(480, 360)
        self.setStyleSheet("background-color: #121216; border: 1px solid #333342; border-radius: 4px;")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._image_width: int = 0
        self._image_height: int = 0
        self._pixmap: Optional[QPixmap] = None

        # ROI selection state (in image coords)
        self.roi: Optional[Tuple[int, int, int, int]] = None
        self._selecting_roi: bool = False
        self._drag_start_pt: Optional[QPoint] = None
        self._drag_current_pt: Optional[QPoint] = None

        self.show_crosshair: bool = True

    def set_frame(self, frame_bgr: np.ndarray):
        """Receives a BGR numpy frame and updates display pixmap."""
        h, w, ch = frame_bgr.shape
        self._image_width = w
        self._image_height = h

        bytes_per_line = ch * w
        q_img = QImage(frame_bgr.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
        self._pixmap = QPixmap.fromImage(q_img)
        self.update()

    def set_roi(self, roi: Optional[Tuple[int, int, int, int]]):
        self.roi = roi
        self.update()

    def clear_roi(self):
        self.roi = None
        self.roi_cleared.emit()
        self.update()

    def _get_display_rect(self) -> QRect:
        """Returns the sub-rectangle in widget coordinates where image is rendered."""
        if not self._pixmap or self._pixmap.isNull():
            return QRect(0, 0, self.width(), self.height())

        scaled = self._pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        return QRect(x, y, scaled.width(), scaled.height())

    def _widget_to_image_coords(self, pt: QPoint) -> Tuple[int, int]:
        """Converts widget pixel coordinates to original image coordinates."""
        d_rect = self._get_display_rect()
        if d_rect.width() <= 0 or d_rect.height() <= 0 or self._image_width <= 0:
            return (0, 0)

        rel_x = pt.x() - d_rect.x()
        rel_y = pt.y() - d_rect.y()
        img_x = int(rel_x * self._image_width / d_rect.width())
        img_y = int(rel_y * self._image_height / d_rect.height())

        img_x = max(0, min(self._image_width - 1, img_x))
        img_y = max(0, min(self._image_height - 1, img_y))
        return (img_x, img_y)

    def _image_to_widget_rect(self, roi: Tuple[int, int, int, int]) -> QRect:
        """Converts image ROI (x, y, w, h) to widget coordinates."""
        d_rect = self._get_display_rect()
        if self._image_width <= 0 or self._image_height <= 0:
            return QRect()

        rx, ry, rw, rh = roi
        wx = d_rect.x() + int(rx * d_rect.width() / self._image_width)
        wy = d_rect.y() + int(ry * d_rect.height() / self._image_height)
        ww = int(rw * d_rect.width() / self._image_width)
        wh = int(rh * d_rect.height() / self._image_height)
        return QRect(wx, wy, ww, wh)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._selecting_roi = True
            self._drag_start_pt = event.pos()
            self._drag_current_pt = event.pos()
        elif event.button() == Qt.MouseButton.RightButton:
            self.clear_roi()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._selecting_roi:
            self._drag_current_pt = event.pos()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton and self._selecting_roi:
            self._selecting_roi = False
            if self._drag_start_pt and self._drag_current_pt:
                x1, y1 = self._widget_to_image_coords(self._drag_start_pt)
                x2, y2 = self._widget_to_image_coords(self._drag_current_pt)
                rx = min(x1, x2)
                ry = min(y1, y2)
                rw = abs(x2 - x1)
                rh = abs(y2 - y1)
                if rw > 15 and rh > 15:
                    self.roi = (rx, ry, rw, rh)
                    self.roi_selected.emit(rx, ry, rw, rh)
                else:
                    self.clear_roi()
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        # 1. Render scaled pixmap
        if self._pixmap and not self._pixmap.isNull():
            d_rect = self._get_display_rect()
            painter.drawPixmap(d_rect, self._pixmap)

            # Center crosshair
            if self.show_crosshair:
                cx = d_rect.x() + d_rect.width() // 2
                cy = d_rect.y() + d_rect.height() // 2
                painter.setPen(QPen(QColor(88, 166, 255, 120), 1, Qt.PenStyle.DashLine))
                painter.drawLine(cx, d_rect.y(), cx, d_rect.y() + d_rect.height())
                painter.drawLine(d_rect.x(), cy, d_rect.x() + d_rect.width(), cy)

            # Draw committed ROI
            if self.roi is not None:
                roi_w_rect = self._image_to_widget_rect(self.roi)
                painter.setPen(QPen(QColor(0, 255, 120), 2))
                painter.setBrush(QBrush(QColor(0, 255, 120, 25)))
                painter.drawRect(roi_w_rect)
                painter.setPen(QColor(0, 255, 120))
                painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
                painter.drawText(roi_w_rect.x() + 4, roi_w_rect.y() - 4, "对焦ROI")

            # Draw dragging ROI preview
            if self._selecting_roi and self._drag_start_pt and self._drag_current_pt:
                drag_rect = QRect(self._drag_start_pt, self._drag_current_pt).normalized()
                painter.setPen(QPen(QColor(255, 215, 0), 2, Qt.PenStyle.DashLine))
                painter.setBrush(QBrush(QColor(255, 215, 0, 30)))
                painter.drawRect(drag_rect)
        else:
            painter.setPen(QColor(150, 150, 160))
            painter.setFont(QFont("Microsoft YaHei", 12))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "等待相机画面...")


class FocusCurveWidget(QWidget):
    """
    Real-time rolling line chart displaying sharpness score history,
    peak line, and in-focus tolerance band.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(140)
        self.setStyleSheet("background-color: #14141a; border: 1px solid #333342; border-radius: 4px;")

        self.scores: List[float] = []
        self.peak_score: float = 0.0
        self.in_focus_ratio: float = 0.92

    def update_curve(self, scores: List[float], peak_score: float, in_focus_ratio: float):
        self.scores = scores
        self.peak_score = peak_score
        self.in_focus_ratio = in_focus_ratio
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        w = self.width()
        h = self.height()
        margin_left = 60
        margin_right = 16
        margin_top = 22
        margin_bottom = 26

        plot_w = w - margin_left - margin_right
        plot_h = h - margin_top - margin_bottom

        # Background grid
        painter.fillRect(margin_left, margin_top, plot_w, plot_h, QColor(18, 18, 24))
        painter.setPen(QPen(QColor(40, 40, 52), 1, Qt.PenStyle.DotLine))
        for gy in range(1, 4):
            y = margin_top + int(plot_h * gy / 4)
            painter.drawLine(margin_left, y, margin_left + plot_w, y)

        if not self.scores or len(self.scores) < 2:
            painter.setPen(QColor(100, 100, 120))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "正在采集清晰度数据...")
            return

        max_val = max(self.peak_score * 1.15, max(self.scores) * 1.15, 10.0)
        min_val = 0.0

        def val_to_y(v: float) -> int:
            norm = (v - min_val) / (max_val - min_val)
            return int(margin_top + plot_h * (1.0 - norm))

        # 1. In-focus shaded band [in_focus_ratio * peak, peak]
        if self.peak_score > 1.0:
            y_peak = val_to_y(self.peak_score)
            y_band_low = val_to_y(self.peak_score * self.in_focus_ratio)
            band_h = max(2, y_band_low - y_peak)
            painter.fillRect(
                margin_left,
                y_peak,
                plot_w,
                band_h,
                QColor(25, 135, 84, 50),
            )
            # Peak dashed line
            painter.setPen(QPen(QColor(40, 167, 69), 1, Qt.PenStyle.DashLine))
            painter.drawLine(margin_left, y_peak, margin_left + plot_w, y_peak)
            painter.setFont(QFont("Segoe UI", 9))
            painter.drawText(margin_left + 6, y_peak - 4, f"峰值景深带 (≥{int(self.in_focus_ratio*100)}%)")

        # 2. Plot score curve
        n = len(self.scores)
        dx = plot_w / max(1, n - 1)
        points = []
        for i, val in enumerate(self.scores):
            px = int(margin_left + i * dx)
            py = val_to_y(val)
            points.append(QPoint(px, py))

        pen_curve = QPen(QColor(88, 166, 255), 2)
        painter.setPen(pen_curve)
        for i in range(len(points) - 1):
            painter.drawLine(points[i], points[i + 1])

        # Current point marker
        cur_pt = points[-1]
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(0, 255, 160)))
        painter.drawEllipse(cur_pt, 4, 4)

        # 3. Y-axis labels
        painter.setPen(QColor(150, 150, 170))
        painter.setFont(QFont("Consolas", 8))
        painter.drawText(6, margin_top + 10, f"{max_val:.1f}")
        painter.drawText(6, margin_top + plot_h // 2, f"{(max_val/2):.1f}")
        painter.drawText(6, margin_top + plot_h, "0.0")

        # Title
        painter.drawText(margin_left, margin_top - 6, "清晰度趋势波形 (当前 / 峰值对比)")


class FocusStatusCard(QFrame):
    """
    Prominent status banner showing dynamic focus state,
    large percentage badge, and instant feedback.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            "background-color: #202028; border: 2px solid #363646; border-radius: 8px; padding: 10px;"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)

        self.label_state = QLabel(FocusState.OUT_OF_FOCUS.value)
        self.label_state.setFont(QFont("Microsoft YaHei", 15, QFont.Weight.Bold))
        self.label_state.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.label_ratio = QLabel("0.0%")
        self.label_ratio.setFont(QFont("Consolas", 28, QFont.Weight.Bold))
        self.label_ratio.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.label_metrics = QLabel("当前得分: 0.0  |  历史峰值: 0.0")
        self.label_metrics.setFont(QFont("Segoe UI", 10))
        self.label_metrics.setStyleSheet("color: #a0a0b0;")
        self.label_metrics.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self.label_state)
        layout.addWidget(self.label_ratio)
        layout.addWidget(self.label_metrics)

    def set_status(
        self,
        state_str: str,
        ratio: float,
        current_score: float,
        peak_score: float,
    ):
        percent_str = f"{ratio * 100:.1f}%"
        self.label_ratio.setText(percent_str)
        self.label_state.setText(state_str)
        self.label_metrics.setText(
            f"当前清晰度: {current_score:.1f}   |   历史峰值: {peak_score:.1f}"
        )

        # Dynamic color styling based on state
        if state_str == FocusState.IN_FOCUS.value:
            self.setStyleSheet(
                "background-color: #0f381f; border: 2px solid #28a745; border-radius: 8px; padding: 10px;"
            )
            self.label_state.setStyleSheet("color: #4cd964;")
            self.label_ratio.setStyleSheet("color: #4cd964;")
        elif state_str == FocusState.OVERSHOT.value:
            self.setStyleSheet(
                "background-color: #3d2314; border: 2px solid #fd7e14; border-radius: 8px; padding: 10px;"
            )
            self.label_state.setStyleSheet("color: #ff922b;")
            self.label_ratio.setStyleSheet("color: #ff922b;")
        elif state_str == FocusState.APPROACHING.value:
            self.setStyleSheet(
                "background-color: #383314; border: 2px solid #ffc107; border-radius: 8px; padding: 10px;"
            )
            self.label_state.setStyleSheet("color: #ffd43b;")
            self.label_ratio.setStyleSheet("color: #ffd43b;")
        else:
            self.setStyleSheet(
                "background-color: #202028; border: 2px solid #363646; border-radius: 8px; padding: 10px;"
            )
            self.label_state.setStyleSheet("color: #adb5bd;")
            self.label_ratio.setStyleSheet("color: #adb5bd;")

