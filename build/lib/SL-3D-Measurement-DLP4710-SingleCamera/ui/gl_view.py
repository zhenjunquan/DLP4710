"""
Interactive 3D OpenGL point cloud visualization widget using PyQtGraph.
"""

from typing import Optional, Tuple
import numpy as np
import pyqtgraph.opengl as gl
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class GLPointCloudWidget(QWidget):
    """
    3D Point Cloud viewer supporting mouse rotation, pan, zoom,
    height-based color mapping, and point size control.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._current_points: Optional[np.ndarray] = None
        self._point_size: float = 2.0
        self._colormap_mode: str = "Turbo/Viridis (按Z深度)"

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # 3D GL View
        self.gl_view = gl.GLViewWidget()
        self.gl_view.setBackgroundColor("#121216")
        self.gl_view.setCameraPosition(distance=250, elevation=30, azimuth=45)

        # Add grid and axes
        self.grid = gl.GLGridItem()
        self.grid.setSize(200, 200)
        self.grid.setSpacing(10, 10)
        self.grid.setColor((60, 60, 80, 120))
        self.gl_view.addItem(self.grid)

        self.axis = gl.GLAxisItem()
        self.axis.setSize(30, 30, 30)
        self.gl_view.addItem(self.axis)

        # Scatter plot for points
        self.scatter = gl.GLScatterPlotItem()
        self.gl_view.addItem(self.scatter)

        layout.addWidget(self.gl_view, 1)

        # Bottom toolbar
        bar = QWidget()
        bar_lay = QHBoxLayout(bar)
        bar_lay.setContentsMargins(8, 4, 8, 4)
        bar_lay.setSpacing(10)

        bar_lay.addWidget(QLabel("点云颜色:"))
        self.combo_color = QComboBox()
        self.combo_color.addItems(["Turbo/Viridis (按Z深度)", "彩虹色 (Rainbow)", "单色 (白/绿)"])
        self.combo_color.currentTextChanged.connect(self._on_color_mode_changed)
        bar_lay.addWidget(self.combo_color)

        bar_lay.addWidget(QLabel("点大小:"))
        self.spin_pt_size = QDoubleSpinBox()
        self.spin_pt_size.setRange(0.5, 10.0)
        self.spin_pt_size.setSingleStep(0.5)
        self.spin_pt_size.setValue(2.0)
        self.spin_pt_size.valueChanged.connect(self._on_pt_size_changed)
        bar_lay.addWidget(self.spin_pt_size)

        self.btn_reset_cam = QPushButton("重置视角")
        self.btn_reset_cam.clicked.connect(self.reset_camera)
        bar_lay.addWidget(self.btn_reset_cam)

        self.lbl_info = QLabel("点数: 0")
        self.lbl_info.setStyleSheet("color: #8da4c4; font-weight: bold;")
        bar_lay.addWidget(self.lbl_info)

        bar_lay.addStretch()
        layout.addWidget(bar)

    def set_point_cloud(
        self,
        xyz_points: Optional[np.ndarray],
        colors: Optional[np.ndarray] = None,
    ):
        """
        Updates the 3D scatter plot with new points.
        xyz_points: (N, 3) float64/float32
        colors: Optional (N, 4) float (RGBA in [0, 1])
        """
        if xyz_points is None or len(xyz_points) == 0:
            self.clear_point_cloud()
            return

        self._current_points = np.asarray(xyz_points, dtype=np.float32)
        n_pts = len(self._current_points)
        self.lbl_info.setText(f"点数: {n_pts:,}")

        # Compute centroid for auto-centering camera
        center = np.mean(self._current_points, axis=0)

        # Generate colors if not provided
        if colors is None:
            colors = self._compute_depth_colors(self._current_points)

        self.scatter.setData(
            pos=self._current_points,
            color=colors,
            size=self._point_size,
            pxMode=True,
        )

        # Position camera looking at the cloud
        extent = np.ptp(self._current_points, axis=0)
        max_extent = max(20.0, float(np.max(extent)))
        self.gl_view.opts["center"] = center
        self.gl_view.opts["distance"] = max_extent * 2.2
        self.gl_view.update()

    def _compute_depth_colors(self, pts: np.ndarray) -> np.ndarray:
        z = pts[:, 2]
        z_min = float(np.min(z))
        z_max = float(np.max(z))
        dz = max(1e-6, z_max - z_min)
        norm_z = np.clip((z - z_min) / dz, 0.0, 1.0)

        n = len(pts)
        rgba = np.ones((n, 4), dtype=np.float32)

        mode = self.combo_color.currentText()
        if "单色" in mode:
            rgba[:, 0] = 0.2
            rgba[:, 1] = 0.85
            rgba[:, 2] = 0.5
        elif "Rainbow" in mode:
            # Hue from 0.7 (blue) down to 0.0 (red)
            h = (1.0 - norm_z) * 0.7
            rgba[:, 0] = np.clip(np.abs(h * 6.0 - 3.0) - 1.0, 0.0, 1.0)
            rgba[:, 1] = np.clip(2.0 - np.abs(h * 6.0 - 2.0), 0.0, 1.0)
            rgba[:, 2] = np.clip(2.0 - np.abs(h * 6.0 - 4.0), 0.0, 1.0)
        else:
            # Viridis-like: deep purple -> teal -> green -> yellow
            rgba[:, 0] = np.clip(0.28 + 0.7 * norm_z, 0.0, 1.0)
            rgba[:, 1] = np.clip(0.15 + 0.8 * norm_z**0.7, 0.0, 1.0)
            rgba[:, 2] = np.clip(0.48 * (1.0 - norm_z) + 0.1, 0.0, 1.0)

        rgba[:, 3] = 0.95
        return rgba

    def _on_color_mode_changed(self):
        if self._current_points is not None:
            self.set_point_cloud(self._current_points)

    def _on_pt_size_changed(self, val: float):
        self._point_size = val
        if self._current_points is not None:
            self.scatter.setData(size=self._point_size)

    def reset_camera(self):
        if self._current_points is not None and len(self._current_points) > 0:
            center = np.mean(self._current_points, axis=0)
            extent = np.ptp(self._current_points, axis=0)
            max_extent = max(20.0, float(np.max(extent)))
            self.gl_view.opts["center"] = center
            self.gl_view.opts["distance"] = max_extent * 2.2
        else:
            self.gl_view.setCameraPosition(distance=250, elevation=30, azimuth=45)
        self.gl_view.update()

    def clear_point_cloud(self):
        self._current_points = None
        self.scatter.setData(pos=np.empty((0, 3)))
        self.lbl_info.setText("点数: 0")
        self.gl_view.update()

