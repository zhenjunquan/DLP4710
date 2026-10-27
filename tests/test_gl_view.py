"""Tests for robust OpenGL point-cloud display preparation."""

import importlib
import os
import sys
import unittest

import numpy as np


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PyQt6.QtGui import QVector3D
from PyQt6.QtWidgets import QApplication


GLPointCloudWidget = importlib.import_module(
    "SL-3D-Measurement-DLP4710-SingleCamera.ui.gl_view"
).GLPointCloudWidget


class TestGLPointCloudWidget(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_camera_center_uses_qvector3d_and_invalid_points_are_removed(self):
        widget = GLPointCloudWidget()
        points = np.array(
            [
                [0.0, 0.0, 10.0],
                [1.0, 2.0, 11.0],
                [2.0, 4.0, 12.0],
                [np.nan, 3.0, 13.0],
                [4.0, np.inf, 14.0],
            ],
            dtype=np.float64,
        )

        widget.set_point_cloud(points)

        self.assertIsInstance(widget.gl_view.opts["center"], QVector3D)
        self.assertIsNotNone(widget.gl_view.viewMatrix())
        self.assertEqual(len(widget._current_points), 3)
        self.assertTrue(np.all(np.isfinite(widget._current_points)))
        widget.close()

    def test_large_cloud_is_sampled_for_display_only(self):
        widget = GLPointCloudWidget()
        widget.MAX_DISPLAY_POINTS = 100
        points = np.column_stack(
            (
                np.arange(1000, dtype=np.float32),
                np.zeros(1000, dtype=np.float32),
                np.ones(1000, dtype=np.float32),
            )
        )

        widget.set_point_cloud(points)

        self.assertEqual(widget._total_point_count, 1000)
        self.assertEqual(len(widget._current_points), 100)
        self.assertIn("显示 100", widget.lbl_info.text())
        widget.combo_color.setCurrentText("彩虹色 (Rainbow)")
        self.assertEqual(widget._total_point_count, 1000)
        self.assertIn("显示 100", widget.lbl_info.text())
        widget.close()


if __name__ == "__main__":
    unittest.main()
