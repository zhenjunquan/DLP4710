"""Tests for measurement intermediate-result export."""

import importlib
import json
import os
import sys
import tempfile
import unittest

import cv2
import numpy as np


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
MeasurementController = importlib.import_module(
    "SL-3D-Measurement-DLP4710-SingleCamera.core.controller"
).MeasurementController


class TestMeasurementExport(unittest.TestCase):
    def setUp(self):
        self.controller = MeasurementController()
        height, width = 18, 24
        frames = [np.full((height, width), index, dtype=np.uint8) for index in range(12)]
        wrapped = [
            np.linspace(-np.pi, np.pi, height * width, dtype=np.float32).reshape(height, width)
            + index
            for index in range(3)
        ]
        unwrapped = np.linspace(0, 30, height * width, dtype=np.float32).reshape(height, width)
        modulation = np.linspace(2, 40, height * width, dtype=np.float32).reshape(height, width)
        mask = modulation > 5
        xyz_map = np.zeros((height, width, 3), dtype=np.float64)
        yy, xx = np.mgrid[:height, :width]
        xyz_map[..., 0] = xx
        xyz_map[..., 1] = yy
        xyz_map[..., 2] = 180.0 + xx * 0.01
        self.controller.latest_result = {
            "frames": frames,
            "wrapped_phases": wrapped,
            "unwrapped_phase": unwrapped,
            "modulation": modulation,
            "mask": mask,
            "xyz_map": xyz_map,
            "point_count": height * width,
            "elapsed_seconds": 1.25,
        }

    def test_save_intermediate_results(self):
        with tempfile.TemporaryDirectory() as parent:
            output = os.path.join(parent, "sample_result")
            ok, message = self.controller.save_intermediate_results(output)

            self.assertTrue(ok, message)
            self.assertTrue(os.path.isfile(os.path.join(output, "measurement_info.txt")))
            self.assertTrue(os.path.isfile(os.path.join(output, "capture_parameters.json")))

            # Check capture_parameters.json content
            with open(os.path.join(output, "capture_parameters.json"), "r", encoding="utf-8") as f:
                params = json.load(f)
            self.assertIn("point_cloud", params)
            self.assertIn("camera", params)
            self.assertIn("dlp", params)

            # Check 12 raw fringe images saved in 01_fringes
            for index in range(1, 13):
                self.assertTrue(
                    os.path.isfile(
                        os.path.join(output, "01_fringes", f"H_Fringe_{index}.bmp")
                    )
                )
            ok, frames, message = self.controller.load_offline_frames(
                os.path.join(output, "01_fringes")
            )
            self.assertTrue(ok, message)
            self.assertEqual(len(frames), 12)

            # Check that slow intermediate phase files are NOT created
            self.assertFalse(os.path.exists(os.path.join(output, "02_wrapped_phase")))
            self.assertFalse(os.path.exists(os.path.join(output, "03_unwrapped_phase")))
            self.assertFalse(os.path.exists(os.path.join(output, "04_auxiliary")))

            # Check point cloud saved in fast binary PLY format
            ply_path = os.path.join(output, "02_point_cloud", "original_cloud.ply")
            self.assertTrue(os.path.isfile(ply_path))
            # Verify no duplicate slow TXT points file created in package
            txt_path = os.path.join(output, "02_point_cloud", "original_points.txt")
            self.assertFalse(os.path.exists(txt_path))

            expected_point_count = int(
                np.prod(self.controller.latest_result["xyz_map"].shape[:2])
            )
            # Read binary PLY back
            from dlp_common.reconstruction.point_cloud_io import read_point_cloud_ply, ply_to_txt
            ok_read, read_pts, _, _ = read_point_cloud_ply(ply_path)
            self.assertTrue(ok_read)
            self.assertEqual(len(read_pts), expected_point_count)

            # Test PLY to TXT conversion
            temp_txt = os.path.join(output, "converted_from_ply.txt")
            ok_conv, conv_count, _ = ply_to_txt(ply_path, temp_txt)
            self.assertTrue(ok_conv)
            self.assertEqual(conv_count, expected_point_count)
            self.assertTrue(os.path.isfile(temp_txt))

    def test_save_contains_different_original_and_processed_clouds(self):
        original_xyz = self.controller.latest_result["xyz_map"]
        processed_xyz = original_xyz.copy()
        processed_xyz[:, :8] = 0.0
        self.controller.latest_result["original_xyz_map"] = original_xyz
        self.controller.latest_result["xyz_map"] = processed_xyz
        self.controller.latest_result["point_cloud_processed"] = True

        with tempfile.TemporaryDirectory() as parent:
            output = os.path.join(parent, "processed_result")
            ok, message = self.controller.save_intermediate_results(output)
            self.assertTrue(ok, message)

            from dlp_common.reconstruction.point_cloud_io import read_point_cloud_ply
            original_ply = os.path.join(
                output, "02_point_cloud", "original_cloud.ply"
            )
            processed_ply = os.path.join(
                output, "02_point_cloud", "processed_cloud.ply"
            )
            self.assertTrue(os.path.isfile(original_ply))
            self.assertTrue(os.path.isfile(processed_ply))

            _, orig_pts, _, _ = read_point_cloud_ply(original_ply)
            _, proc_pts, _, _ = read_point_cloud_ply(processed_ply)
            self.assertGreater(len(orig_pts), len(proc_pts))

    def test_existing_result_folder_is_rejected(self):
        with tempfile.TemporaryDirectory() as parent:
            output = os.path.join(parent, "already_exists")
            os.mkdir(output)
            ok, message = self.controller.save_intermediate_results(output)
            self.assertFalse(ok)
            self.assertIn("已存在", message)


if __name__ == "__main__":
    unittest.main()
