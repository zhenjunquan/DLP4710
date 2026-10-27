"""
Reusable UI widgets for image preview with corner overlay and pose management table.
"""

from typing import List, Optional
import cv2
import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QLabel,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class ImageDisplayWidget(QLabel):
    """
    High-performance image display widget supporting OpenCV Mat,
    aspect-ratio preserving scaling, crosshairs, live badges, and overlaying chessboard corners.
    """

    def __init__(self, title: str = "", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.title = title
        self._current_pixmap: Optional[QPixmap] = None
        self._corners: Optional[np.ndarray] = None
        self._raw_size = (0, 0)
        self.show_crosshair: bool = False
        self.badge_text: str = ""

        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(320, 260)
        self.setStyleSheet("background-color: #121217; border: 1px solid #2d2d38; border-radius: 4px;")
        if title:
            self.setText(f"<div style='color:#666680; font-size:14px;'>{title}</div>")

    def set_mat_image(
        self,
        mat: Optional[np.ndarray],
        corners: Optional[np.ndarray] = None,
        badge: str = "",
    ):
        """
        Updates the displayed image from an OpenCV numpy array (BGR, Grayscale, or Float).
        corners: Optional (N, 2) corner coordinates in original image resolution.
        badge: Optional badge text (e.g. FPS / resolution).
        """
        self.badge_text = badge

        if mat is None or mat.size == 0:
            self._current_pixmap = None
            self._corners = None
            self._raw_size = (0, 0)
            self.setText(f"<div style='color:#666680; font-size:12px;'>{self.title} (待机 / 无画面)</div>")
            self.update()
            return

        h, w = mat.shape[:2]
        self._raw_size = (w, h)
        self._corners = corners

        # Normalize float image (e.g. phase map) to uint8 for display
        if mat.dtype == np.float32 or mat.dtype == np.float64:
            valid_mask = np.isfinite(mat)
            if np.any(valid_mask):
                min_v = float(np.min(mat[valid_mask]))
                max_v = float(np.max(mat[valid_mask]))
                if max_v > min_v:
                    norm = np.zeros(mat.shape, dtype=np.uint8)
                    norm[valid_mask] = np.clip(
                        (mat[valid_mask] - min_v) / (max_v - min_v) * 255.0,
                        0,
                        255,
                    ).astype(np.uint8)
                else:
                    norm = np.zeros_like(mat, dtype=np.uint8)
            else:
                norm = np.zeros_like(mat, dtype=np.uint8)
            display_mat = cv2.cvtColor(
                cv2.applyColorMap(norm, cv2.COLORMAP_VIRIDIS),
                cv2.COLOR_BGR2RGB,
            )
        elif mat.ndim == 2:
            display_mat = cv2.cvtColor(mat, cv2.COLOR_GRAY2RGB)
        else:
            display_mat = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)

        q_img = QImage(
            display_mat.data,
            w,
            h,
            display_mat.strides[0],
            QImage.Format.Format_RGB888,
        )
        self._current_pixmap = QPixmap.fromImage(q_img)
        self.setText("")
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._current_pixmap is None or self._current_pixmap.isNull():
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        widget_w = self.width()
        widget_h = self.height()
        raw_w, raw_h = self._raw_size

        if raw_w <= 0 or raw_h <= 0:
            return

        # Maintain aspect ratio
        scale = min(widget_w / raw_w, widget_h / raw_h)
        target_w = int(raw_w * scale)
        target_h = int(raw_h * scale)
        target_x = (widget_w - target_w) // 2
        target_y = (widget_h - target_h) // 2

        # Draw scaled image
        scaled_pixmap = self._current_pixmap.scaled(
            target_w,
            target_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        painter.drawPixmap(target_x, target_y, scaled_pixmap)

        # Draw center crosshair if enabled
        if self.show_crosshair:
            pen_cross = QPen(QColor(255, 0, 80, 160), 1.0, Qt.PenStyle.DashLine)
            painter.setPen(pen_cross)
            cx = target_x + target_w // 2
            cy = target_y + target_h // 2
            painter.drawLine(cx, target_y, cx, target_y + target_h)
            painter.drawLine(target_x, cy, target_x + target_w, cy)
            # Center target box
            painter.drawRect(cx - 30, cy - 30, 60, 60)

        # Draw corner overlay if present
        if self._corners is not None and len(self._corners) > 0:
            pen_line = QPen(QColor(255, 60, 60, 190), 1.5)
            painter.setPen(pen_line)

            screen_pts = []
            for pt in self._corners:
                sx = target_x + pt[0] * scale
                sy = target_y + pt[1] * scale
                screen_pts.append(QPointF(sx, sy))

            # Connect consecutive corners with lines
            for i in range(len(screen_pts) - 1):
                painter.drawLine(screen_pts[i], screen_pts[i + 1])

            # Draw circle markers on corners
            pen_dot = QPen(QColor(0, 255, 128), 2.0)
            painter.setPen(pen_dot)
            painter.setBrush(QColor(0, 255, 128, 180))
            for pt in screen_pts:
                painter.drawEllipse(pt, 3.5, 3.5)

        # Title badge top-left
        if self.title:
            painter.setPen(QColor(220, 220, 240))
            painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            painter.drawText(12, 22, self.title)

        # Live status badge top-right
        if self.badge_text:
            painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            badge_w = painter.fontMetrics().horizontalAdvance(self.badge_text) + 16
            badge_h = 22
            badge_x = widget_w - badge_w - 12
            badge_y = 10
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(20, 20, 30, 200))
            painter.drawRoundedRect(badge_x, badge_y, badge_w, badge_h, 4, 4)

            painter.setPen(QColor(0, 255, 128) if "LIVE" in self.badge_text else QColor(140, 180, 255))
            painter.drawText(badge_x + 8, badge_y + 15, self.badge_text)


class PoseTableWidget(QTableWidget):
    """
    Table widget for displaying and managing calibration board poses.
    """

    COL_INDEX = 0
    COL_NAME = 1
    COL_CORNER = 2
    COL_PHASE = 3
    COL_CAM_RMS = 4
    COL_PROJ_RMS = 5

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(0, 6, parent)
        headers = ["序号", "姿态名称", "角点状态", "相位状态", "相机RMS (px)", "投影仪RMS (px)"]
        self.setHorizontalHeaderLabels(headers)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.horizontalHeader().setStretchLastSection(True)
        self.horizontalHeader().setMinimumSectionSize(48)
        self.setColumnWidth(self.COL_INDEX, 48)
        self.setColumnWidth(self.COL_NAME, 90)
        self.setColumnWidth(self.COL_CORNER, 76)
        self.setColumnWidth(self.COL_PHASE, 76)
        self.setColumnWidth(self.COL_CAM_RMS, 105)
        self.setColumnWidth(self.COL_PROJ_RMS, 116)
        self.setMinimumWidth(0)
        self.verticalHeader().setVisible(False)
        self.setAlternatingRowColors(True)

    def set_pose_row(
        self,
        row: int,
        index_num: int,
        name: str,
        corner_ok: bool,
        phase_ok: bool,
        cam_err: float = -1.0,
        proj_err: float = -1.0,
    ):
        if row >= self.rowCount():
            self.insertRow(row)

        item_idx = QTableWidgetItem(f"{index_num:02d}")
        item_idx.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setItem(row, self.COL_INDEX, item_idx)

        item_name = QTableWidgetItem(name)
        self.setItem(row, self.COL_NAME, item_name)

        corner_text = "已检测" if corner_ok else "未检测"
        item_corner = QTableWidgetItem(corner_text)
        item_corner.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        if corner_ok:
            item_corner.setForeground(QColor("#28a745"))
        else:
            item_corner.setForeground(QColor("#dc3545"))
        self.setItem(row, self.COL_CORNER, item_corner)

        phase_text = "已解算" if phase_ok else "未解算"
        item_phase = QTableWidgetItem(phase_text)
        item_phase.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        if phase_ok:
            item_phase.setForeground(QColor("#28a745"))
        else:
            item_phase.setForeground(QColor("#dc3545"))
        self.setItem(row, self.COL_PHASE, item_phase)

        cam_text = f"{cam_err:.3f}" if cam_err >= 0 else "-"
        item_cam = QTableWidgetItem(cam_text)
        item_cam.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setItem(row, self.COL_CAM_RMS, item_cam)

        proj_text = f"{proj_err:.3f}" if proj_err >= 0 else "-"
        item_proj = QTableWidgetItem(proj_text)
        item_proj.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setItem(row, self.COL_PROJ_RMS, item_proj)
