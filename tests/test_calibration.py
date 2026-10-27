"""
Unit tests for calibration models, corner detection, and YAML serialization.
"""

import os
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dlp_common.calibration.calib_io import CalibrationIO
from dlp_common.calibration.corner_detector import CornerDetector
from dlp_common.calibration.models import CalibrationResult, CalibrationView
from dlp_common.calibration.telecentric_camera import (
    decompose_affine_camera,
    fit_linear_affine,
    project_telecentric_point,
    undistort_telecentric_pixel,
)


class TestCalibration(unittest.TestCase):
    def test_telecentric_forward_and_undistort(self):
        affine = np.array([
            [100.0, 0.0, 0.0, 500.0],
            [0.0, 100.0, 0.0, 400.0],
        ], dtype=np.float64)
        dist5 = np.array([-1e-4, 5e-5, 0.0, 0.0, 0.0], dtype=np.float64)
        size = (1000, 800)

        pt_3d = np.array([2.5, 1.8, 150.0], dtype=np.float64)
        pred_uv = project_telecentric_point(pt_3d, affine, dist5, size)
        self.assertEqual(len(pred_uv), 2)
        self.assertTrue(np.all(np.isfinite(pred_uv)))

        ok, ideal = undistort_telecentric_pixel(pred_uv, affine, dist5, size)
        self.assertTrue(ok)
        # Expected ideal without distortion is affine @ [2.5, 1.8, 150, 1]^T = [750, 580]
        self.assertAlmostEqual(ideal[0], 750.0, places=2)
        self.assertAlmostEqual(ideal[1], 580.0, places=2)

    def test_fit_linear_affine(self):
        # Generate known affine transformation
        true_affine = np.array([
            [105.0, -6.0, 2.0, 1950.0],
            [6.5, 95.0, -45.0, 12900.0],
        ], dtype=np.float64)

        # 50 random 3D points
        np.random.seed(42)
        pts_3d = np.random.uniform(-20, 20, size=(50, 3))
        homo_3d = np.hstack([pts_3d, np.ones((50, 1))])
        pts_cam = (true_affine @ homo_3d.T).T

        ok, fitted_affine = fit_linear_affine([pts_3d], [pts_cam])
        self.assertTrue(ok)
        np.testing.assert_allclose(fitted_affine, true_affine, atol=1e-5)

    def test_object_points_follow_opencv_corner_order(self):
        detector = CornerDetector(board_cols=3, board_rows=2, square_size_mm=3.0)
        expected = np.array(
            [
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [6.0, 0.0, 0.0],
                [0.0, 3.0, 0.0],
                [3.0, 3.0, 0.0],
                [6.0, 3.0, 0.0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(detector.create_object_points(), expected)

    def test_calibration_io_roundtrip(self):
        result = CalibrationResult()
        result.projector_matrix = np.eye(3, dtype=np.float64) * 4000.0
        result.projector_matrix[2, 2] = 1.0
        result.projector_dist_14 = np.zeros((14, 1), dtype=np.float64)
        result.telecentric_projection_2x4 = np.ones((2, 4), dtype=np.float64)
        result.telecentric_dist_5 = np.zeros((5, 1), dtype=np.float64)
        result.board_rvecs = [np.zeros((3, 1), dtype=np.float64)]
        result.board_tvecs = [np.array([[0.0], [0.0], [200.0]], dtype=np.float64)]
        result.view_names = ["Pose_01"]
        result.camera_rms = 0.25
        result.projector_rms = 0.22
        result.joint_rms = 0.235

        with tempfile.TemporaryDirectory() as tmp_dir:
            yml_path = os.path.join(tmp_dir, "test_calib.yml")
            txt_path = os.path.join(tmp_dir, "test_calib.txt")
            ok, msg = CalibrationIO.save_yml_and_txt(
                yml_path=yml_path,
                txt_path=txt_path,
                result=result,
                camera_image_size=(5328, 4608),
                projector_image_size=(1920, 1080),
                board_size=(11, 8),
                square_size_mm=5.0,
                fringe_periods=[70.0, 64.0, 59.0],
            )
            self.assertTrue(ok)
            self.assertTrue(os.path.isfile(yml_path))
            self.assertTrue(os.path.isfile(txt_path))

            # Load back
            ok_load, loaded_res, meta, msg_load = CalibrationIO.load_yml(yml_path)
            self.assertTrue(ok_load)
            self.assertAlmostEqual(loaded_res.joint_rms, 0.235, places=3)
            self.assertEqual(meta["board_width"], 11)
            self.assertEqual(meta["board_height"], 8)


if __name__ == "__main__":
    unittest.main()

