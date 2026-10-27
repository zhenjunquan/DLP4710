"""
2D image viewers and ROI selector for 3D measurement software.
"""

from typing import Optional
import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QGridLayout,
    QLabel,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


class ScaledImageLabel(QLabel):
    """Aspect-ratio preserving image display label."""

    def __init__(self, title: str = "", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.title = title
        self._pixmap: Optional[QPixmap] = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(220, 180)
        self.setStyleSheet("background-color: #121217; border: 1px solid #282834; border-radius: 4px;")
        if title:
            self.setText(f"<div style='color:#555566; font-size:13px;'>{title} (待机)</div>")

    def set_mat(self, mat: Optional[np.ndarray]):
        if mat is None or mat.size == 0:
            self._pixmap = None
            self.setText(f"<div style='color:#555566; font-size:13px;'>{self.title} (无数据)</div>")
            self.update()
            return

        h, w = mat.shape[:2]
        if mat.dtype == np.float32 or mat.dtype == np.float64:
            valid = np.isfinite(mat)
            if np.any(valid):
                vmin, vmax = float(np.min(mat[valid])), float(np.max(mat[valid]))
                if vmax > vmin:
                    norm = ((mat - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
                else:
                    norm = np.zeros_like(mat, dtype=np.uint8)
            else:
                norm = np.zeros_like(mat, dtype=np.uint8)
            color_mat = cv2.applyColorMap(norm, cv2.COLORMAP_VIRIDIS)
        elif mat.ndim == 2:
            color_mat = cv2.cvtColor(mat, cv2.COLOR_GRAY2RGB)
        else:
            color_mat = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)

        q_img = QImage(color_mat.data, w, h, color_mat.strides[0], QImage.Format.Format_RGB888)
        self._pixmap = QPixmap.fromImage(q_img)
        self.setText("")
        self._scale_and_set()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._scale_and_set()

    def _scale_and_set(self):
        if self._pixmap is not None and not self._pixmap.isNull():
            scaled = self._pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.setPixmap(scaled)


class MultiImageViewerWidget(QWidget):
    """
    Displays 2x2 grid or tabbed view of 2D process images:
    1. Raw Frame
    2. Wrapped Phase
    3. Unwrapped Absolute Phase
    4. Modulation / Background Mask
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QGridLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self.view_raw = ScaledImageLabel("原始条纹帧 (Raw Frame)")
        self.view_wrapped = ScaledImageLabel("包裹相位 (Wrapped Phase)")
        self.view_unwrapped = ScaledImageLabel("绝对展开相位 (Unwrapped Phase)")
        self.view_mask = ScaledImageLabel("调制与有效掩膜 (Modulation Mask)")

        layout.addWidget(self.view_raw, 0, 0)
        layout.addWidget(self.view_wrapped, 0, 1)
        layout.addWidget(self.view_unwrapped, 1, 0)
        layout.addWidget(self.view_mask, 1, 1)

    def update_views(
        self,
        raw_frame: Optional[np.ndarray] = None,
        wrapped_phase: Optional[np.ndarray] = None,
        unwrapped_phase: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
    ):
        if raw_frame is not None:
            self.view_raw.set_mat(raw_frame)
        if wrapped_phase is not None:
            self.view_wrapped.set_mat(wrapped_phase)
        if unwrapped_phase is not None:
            self.view_unwrapped.set_mat(unwrapped_phase)
        if mask is not None:
            self.view_mask.set_mat(mask)

