"""Tests for organized point-cloud subject extraction."""

import importlib
import os
import sys
import unittest

import numpy as np


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
point_cloud_io = importlib.import_module("dlp_common.reconstruction.point_cloud_io")
MeasurementController = importlib.import_module(
    "SL-3D-Measurement-DLP4710-SingleCamera.core.controller"
).MeasurementController


class TestPointCloudProcessing(unittest.TestCase):
    def _make_cloud(self):
        xyz = np.zeros((60, 100, 3), dtype=np.float64)
        yy, xx = np.mgrid[20:45, 20:75]
        xyz[20:45, 20:75, 0] = xx
        xyz[20:45, 20:75, 1] = yy
        xyz[20:45, 20:75, 2] = 180.0 + xx * 0.01
        xyz[20:45, 47:49] = 0.0  # short gap within the measured subject

        xyz[5:8, 5:30, 0] = 1.0  # separate background strip
        xyz[5:8, 5:30, 2] = 195.0
        xyz[52, 90] = [4.0, 5.0, 170.0]  # isolated outlier
        return xyz

    def test_keep_largest_component_removes_background_and_bridges_gap(self):
        xyz = self._make_cloud()
        filtered, mask, original_count, retained_count = (
            point_cloud_io.keep_largest_image_component(xyz, close_radius=2)
        )

        self.assertGreater(retained_count, 1200)
        self.assertLess(retained_count, original_count)
        self.assertTrue(np.any(filtered[20:45, 20:47]))
        self.assertTrue(np.any(filtered[20:45, 49:75]))
        self.assertFalse(np.any(mask[5:8, 5:30]))
        self.assertFalse(mask[52, 90])

    def test_controller_can_restore_original_cloud(self):
        xyz = self._make_cloud()
        points, _ = point_cloud_io.extract_valid_points(xyz)
        controller = MeasurementController()
        controller.latest_result = {
            "xyz_map": xyz,
            "valid_points": points,
            "point_count": len(points),
        }

        ok, message = controller.keep_largest_subject(close_radius=2)
        self.assertTrue(ok, message)
        self.assertLess(controller.latest_result["point_count"], len(points))

        ok, message = controller.restore_original_point_cloud()
        self.assertTrue(ok, message)
        np.testing.assert_array_equal(controller.latest_result["xyz_map"], xyz)
        self.assertEqual(controller.latest_result["point_count"], len(points))

    def test_subject_radius_control_allows_large_gaps(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt6.QtWidgets import QApplication

        window_class = importlib.import_module(
            "SL-3D-Measurement-DLP4710-SingleCamera.ui.main_window"
        ).MeasurementMainWindow
        app = QApplication.instance() or QApplication([])
        window = window_class()
        self.assertEqual(window.spin_subject_radius.maximum(), 100)
        window.close()


if __name__ == "__main__":
    unittest.main()
