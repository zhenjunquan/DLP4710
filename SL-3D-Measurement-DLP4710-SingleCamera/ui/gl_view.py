"""
Interactive 3D OpenGL point cloud visualization widget using PyQtGraph.
"""

from typing import Optional, Tuple
import numpy as np
import pyqtgraph.opengl as gl
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QVector3D
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

    MAX_DISPLAY_POINTS = 500_000

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._current_points: Optional[np.ndarray] = None
        self._total_point_count: int = 0
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

        points = np.asarray(xyz_points)
        if points.ndim != 2 or points.shape[1] != 3:
            self.clear_point_cloud()
            return

        finite_mask = np.all(np.isfinite(points), axis=1)
        points = points[finite_mask]
        if len(points) == 0:
            self.clear_point_cloud()
            return

        point_colors = None
        if colors is not None:
            colors = np.asarray(colors)
            if (
                colors.ndim == 2
                and colors.shape == (len(finite_mask), 4)
            ):
                point_colors = colors[finite_mask]

        self._total_point_count = len(points)
        if self._total_point_count > self.MAX_DISPLAY_POINTS:
            display_indices = np.linspace(
                0,
                self._total_point_count - 1,
                self.MAX_DISPLAY_POINTS,
                dtype=np.int64,
            )
            points = points[display_indices]
            if point_colors is not None:
                point_colors = point_colors[display_indices]

        self._current_points = np.ascontiguousarray(points, dtype=np.float32)
        finite_display = np.all(np.isfinite(self._current_points), axis=1)
        self._current_points = self._current_points[finite_display]
        if point_colors is not None:
            point_colors = point_colors[finite_display]
        if len(self._current_points) == 0:
            self.clear_point_cloud()
            return

        displayed_count = len(self._current_points)
        if displayed_count < self._total_point_count:
            self.lbl_info.setText(
                f"点数: {self._total_point_count:,}（显示 {displayed_count:,}）"
            )
        else:
            self.lbl_info.setText(f"点数: {self._total_point_count:,}")

        # Generate colors if not provided
        if point_colors is None:
            point_colors = self._compute_depth_colors(self._current_points)
        else:
            point_colors = np.ascontiguousarray(point_colors, dtype=np.float32)

        self.scatter.setData(
            pos=self._current_points,
            color=point_colors,
            size=self._point_size,
            pxMode=True,
        )

        center, distance = self._camera_position(self._current_points)
        self.gl_view.setCameraPosition(pos=center, distance=distance)

    @staticmethod
    def _camera_position(points: np.ndarray) -> Tuple[QVector3D, float]:
        """Return a robust Qt camera target that ignores sparse extreme outliers."""
        if len(points) >= 100:
            lower, upper = np.percentile(points, [1.0, 99.0], axis=0)
        else:
            lower = np.min(points, axis=0)
            upper = np.max(points, axis=0)
        center_values = (lower + upper) * 0.5
        max_extent = max(20.0, float(np.max(upper - lower)))
        center = QVector3D(
            float(center_values[0]),
            float(center_values[1]),
            float(center_values[2]),
        )
        return center, max_extent * 2.2

    def _compute_depth_colors(self, pts: np.ndarray) -> np.ndarray:
        z = pts[:, 2]
        if len(z) >= 100:
            z_min, z_max = np.percentile(z, [1.0, 99.0])
            z_min = float(z_min)
            z_max = float(z_max)
        else:
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
        if self._current_points is not None and len(self._current_points) > 0:
            colors = self._compute_depth_colors(self._current_points)
            self.scatter.setData(color=colors)

    def _on_pt_size_changed(self, val: float):
        self._point_size = val
        if self._current_points is not None:
            self.scatter.setData(size=self._point_size)

    def reset_camera(self):
        if self._current_points is not None and len(self._current_points) > 0:
            center, distance = self._camera_position(self._current_points)
            self.gl_view.setCameraPosition(
                pos=center,
                distance=distance,
                elevation=30,
                azimuth=45,
            )
        else:
            self.gl_view.setCameraPosition(distance=250, elevation=30, azimuth=45)

    def clear_point_cloud(self):
        self._current_points = None
        self._total_point_count = 0
        self.scatter.setData(pos=np.empty((0, 3), dtype=np.float32))
        self.lbl_info.setText("点数: 0")
        self.gl_view.update()

