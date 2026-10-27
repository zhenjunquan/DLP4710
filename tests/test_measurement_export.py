"""Tests for measurement intermediate-result export."""

import importlib
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
            for index in range(1, 4):
                base = os.path.join(
                    output,
                    "02_wrapped_phase",
                    f"wrapped_frequency_{index:02d}",
                )
                self.assertTrue(os.path.isfile(base + ".npy"))
                self.assertTrue(os.path.isfile(base + ".tiff"))
                self.assertTrue(os.path.isfile(base + "_preview.png"))

            loaded = np.load(
                os.path.join(output, "03_unwrapped_phase", "unwrapped_phase.npy")
            )
            np.testing.assert_array_equal(
                loaded,
                self.controller.latest_result["unwrapped_phase"],
            )
            saved_mask = cv2.imread(
                os.path.join(output, "04_auxiliary", "mask.png"),
                cv2.IMREAD_GRAYSCALE,
            )
            np.testing.assert_array_equal(
                saved_mask,
                self.controller.latest_result["mask"].astype(np.uint8) * 255,
            )
            expected_point_count = int(
                np.prod(self.controller.latest_result["xyz_map"].shape[:2])
            )
            for prefix in ("original", "processed"):
                ply_path = os.path.join(
                    output,
                    "05_point_cloud",
                    f"{prefix}_cloud.ply",
                )
                txt_path = os.path.join(
                    output,
                    "05_point_cloud",
                    f"{prefix}_points.txt",
                )
                self.assertTrue(os.path.isfile(ply_path))
                self.assertTrue(os.path.isfile(txt_path))
                with open(ply_path, "r", encoding="ascii") as ply_file:
                    ply_text = ply_file.read()
                self.assertIn(f"element vertex {expected_point_count}", ply_text)
                with open(txt_path, "r", encoding="ascii") as txt_file:
                    self.assertEqual(sum(1 for _ in txt_file), expected_point_count)

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

            original_txt = os.path.join(
                output, "05_point_cloud", "original_points.txt"
            )
            processed_txt = os.path.join(
                output, "05_point_cloud", "processed_points.txt"
            )
            with open(original_txt, "r", encoding="ascii") as txt_file:
                original_count = sum(1 for _ in txt_file)
            with open(processed_txt, "r", encoding="ascii") as txt_file:
                processed_count = sum(1 for _ in txt_file)
            self.assertGreater(original_count, processed_count)

    def test_existing_result_folder_is_rejected(self):
        with tempfile.TemporaryDirectory() as parent:
            output = os.path.join(parent, "already_exists")
            os.mkdir(output)
            ok, message = self.controller.save_intermediate_results(output)
            self.assertFalse(ok)
            self.assertIn("已存在", message)


if __name__ == "__main__":
    unittest.main()
