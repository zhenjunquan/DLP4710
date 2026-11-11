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
        self.setMinimumSize(180, 140)
        self.setStyleSheet("background-color: #121217; border: 1px solid #282834; border-radius: 4px;")
        if title:
            self.setText(f"<div style='color:#555566; font-size:12px;'>{title} (待机)</div>")

    def set_mat(self, mat: Optional[np.ndarray]):
        if mat is None or mat.size == 0:
            self._pixmap = None
            self.setText(f"<div style='color:#555566; font-size:12px;'>{self.title} (无数据)</div>")
            self.update()
            return

        h, w = mat.shape[:2]
        if mat.dtype == np.float32 or mat.dtype == np.float64:
            valid = np.isfinite(mat)
            if np.any(valid):
                vmin, vmax = float(np.min(mat[valid])), float(np.max(mat[valid]))
                if vmax > vmin:
                    norm = np.zeros(mat.shape, dtype=np.uint8)
                    norm[valid] = np.clip(
                        (mat[valid] - vmin) / (vmax - vmin) * 255.0,
                        0,
                        255,
                    ).astype(np.uint8)
                else:
                    norm = np.zeros_like(mat, dtype=np.uint8)
            else:
                norm = np.zeros_like(mat, dtype=np.uint8)
            color_mat = cv2.cvtColor(
                cv2.applyColorMap(norm, cv2.COLORMAP_VIRIDIS),
                cv2.COLOR_BGR2RGB,
            )
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


class MultiImageViewerWidget(QTabWidget):
    """
    Displays 2D process images across tabs:
    - Tab 1: 2x2 grid (Raw/Fused frame, Wrapped phase, Unwrapped phase, Modulation mask)
    - Tab 2: HDR multi-exposure analysis (Fused HDR fringe & Exposure bracket index heatmap)
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # Tab 1: Standard 2x2 process views
        tab_grid = QWidget()
        grid_layout = QGridLayout(tab_grid)
        grid_layout.setContentsMargins(6, 6, 6, 6)
        grid_layout.setSpacing(6)

        self.view_raw = ScaledImageLabel("原始/融合条纹帧 (Fringe Frame)")
        self.view_wrapped = ScaledImageLabel("包裹相位 (Wrapped Phase)")
        self.view_unwrapped = ScaledImageLabel("绝对展开相位 (Unwrapped Phase)")
        self.view_mask = ScaledImageLabel("调制与有效掩膜 (Modulation Mask)")

        grid_layout.addWidget(self.view_raw, 0, 0)
        grid_layout.addWidget(self.view_wrapped, 0, 1)
        grid_layout.addWidget(self.view_unwrapped, 1, 0)
        grid_layout.addWidget(self.view_mask, 1, 1)
        self.addTab(tab_grid, "四分格过程全景")

        # Tab 2: HDR multi-exposure fusion views
        tab_hdr = QWidget()
        hdr_layout = QGridLayout(tab_hdr)
        hdr_layout.setContentsMargins(6, 6, 6, 6)
        hdr_layout.setSpacing(6)

        self.view_hdr_fused = ScaledImageLabel("HDR 融合高对比度条纹 (Fused Fringe)")
        self.view_hdr_indices = ScaledImageLabel("各像素最优曝光档位分布 (HDR Index Heatmap)")

        hdr_layout.addWidget(self.view_hdr_fused, 0, 0)
        hdr_layout.addWidget(self.view_hdr_indices, 0, 1)
        self.addTab(tab_hdr, "HDR 曝光融合图谱")

    def update_views(
        self,
        raw_frame: Optional[np.ndarray] = None,
        wrapped_phase: Optional[np.ndarray] = None,
        unwrapped_phase: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        fused_frame: Optional[np.ndarray] = None,
        exposure_indices: Optional[np.ndarray] = None,
        num_brackets: int = 1,
    ):
        if raw_frame is not None:
            self.view_raw.set_mat(raw_frame)
        if wrapped_phase is not None:
            self.view_wrapped.set_mat(wrapped_phase)
        if unwrapped_phase is not None:
            self.view_unwrapped.set_mat(unwrapped_phase)
        if mask is not None:
            self.view_mask.set_mat(mask)

        if fused_frame is not None:
            self.view_hdr_fused.set_mat(fused_frame)

        if exposure_indices is not None and num_brackets > 1:
            denom = max(1, num_brackets - 1)
            idx_norm = np.clip(
                exposure_indices.astype(np.float32) / float(denom) * 255.0, 0, 255
            ).astype(np.uint8)
            heatmap = cv2.applyColorMap(idx_norm, cv2.COLORMAP_TURBO)
            if mask is not None:
                heatmap[mask == 0] = [20, 20, 25]  # Dark background for masked
            rgb_heat = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            self.view_hdr_indices.set_mat(rgb_heat)

