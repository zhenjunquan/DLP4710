"""
Unit tests for 3D reconstruction using real calibration parameters.
"""

import os
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dlp_common.reconstruction.point_cloud_io import save_point_cloud_ply, save_point_cloud_txt
from dlp_common.reconstruction.reconstructor import MonoSLReconstructor


class TestReconstruction(unittest.TestCase):
    def setUp(self):
        self.calib_file = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "bin_and_templates",
                "MonoSLCalibration.yml",
            )
        )
        self.reconstructor = MonoSLReconstructor()

    def test_load_real_calibration(self):
        self.assertTrue(os.path.isfile(self.calib_file), "Template calibration file must exist")
        ok = self.reconstructor.load_parameters(self.calib_file)
        self.assertTrue(ok, f"Failed to load calibration parameters: {self.reconstructor.last_error}")
        self.assertTrue(self.reconstructor.has_calibration_parameters())
        self.assertGreater(self.reconstructor.max_projector_z, self.reconstructor.min_projector_z)

    def test_reconstruct_point_synthetic(self):
        ok = self.reconstructor.load_parameters(self.calib_file)
        self.assertTrue(ok)

        # Center camera pixel and middle DLP row (v ~ 540)
        cw, ch = self.reconstructor.camera_image_size
        pw, ph = self.reconstructor.projector_image_size

        mid_cam = (float(cw) * 0.5, float(ch) * 0.5)
        mid_v = float(ph) * 0.5

        # Check if ray intersects within depth range
        ok_pt, pt3d = self.reconstructor.reconstruct_one_point(mid_cam, mid_v)
        if ok_pt:
            self.assertEqual(len(pt3d), 3)
            self.assertGreater(pt3d[2], self.reconstructor.min_projector_z)
            self.assertLess(pt3d[2], self.reconstructor.max_projector_z)

    def test_point_cloud_export(self):
        # Create a small dummy xyz map (10, 10, 3)
        xyz_map = np.zeros((10, 10, 3), dtype=np.float64)
        xyz_map[0, 0] = [1.0, 2.0, 180.0]
        xyz_map[1, 1] = [3.0, 4.0, 185.0]

        with tempfile.TemporaryDirectory() as tmp_dir:
            ply_path = os.path.join(tmp_dir, "test.ply")
            txt_path = os.path.join(tmp_dir, "test.txt")

            ok_ply, count_ply, _ = save_point_cloud_ply(ply_path, xyz_map)
            self.assertTrue(ok_ply)
            self.assertEqual(count_ply, 2)
            self.assertTrue(os.path.isfile(ply_path))

            ok_txt, count_txt, _ = save_point_cloud_txt(txt_path, xyz_map)
            self.assertTrue(ok_txt)
            self.assertEqual(count_txt, 2)
            self.assertTrue(os.path.isfile(txt_path))

    def test_reconstruct_cropped_roi_phase_map(self):
        self.assertTrue(self.reconstructor.load_parameters(self.calib_file))
        roi = (100, 200, 3, 2)
        phase = np.ones((2, 3), dtype=np.float64)
        mask = np.full((2, 3), 255, dtype=np.uint8)
        seen_pixels = []

        def fake_reconstruct(camera_pixel, projector_v):
            seen_pixels.append(camera_pixel)
            return True, np.array([camera_pixel[0], camera_pixel[1], projector_v])

        self.reconstructor.reconstruct_one_point = fake_reconstruct
        ok, xyz_map, msg = self.reconstructor.reconstruct_from_phase(
            phase,
            mask=mask,
            roi=roi,
            sample_step=1,
        )

        self.assertTrue(ok, msg)
        self.assertEqual(xyz_map.shape, (2, 3, 3))
        self.assertEqual(seen_pixels[0], (100.0, 200.0))
        self.assertEqual(seen_pixels[-1], (102.0, 201.0))

    def test_vectorized_reconstruction_matches_scalar_geometry(self):
        self.assertTrue(self.reconstructor.load_parameters(self.calib_file))
        camera_uv = np.array(
            [
                [1332.0, 1152.0],
                [1998.0, 1152.0],
                [2664.0, 1728.0],
                [3330.0, 1728.0],
            ],
            dtype=np.float64,
        )
        projector_v = np.full(len(camera_uv), 162.0, dtype=np.float64)
        scalar_points = []
        for uv, proj_v in zip(camera_uv, projector_v):
            ok, point = self.reconstructor.reconstruct_one_point(tuple(uv), float(proj_v))
            self.assertTrue(ok)
            scalar_points.append(point)

        vector_points, valid = self.reconstructor._reconstruct_points_vectorized(
            camera_uv,
            projector_v,
        )
        self.assertTrue(np.all(valid))
        np.testing.assert_allclose(vector_points, np.asarray(scalar_points), atol=1e-3)


if __name__ == "__main__":
    unittest.main()

