"""
Unit tests for DLP4710 Projector Light Intensity Control and Multi-Exposure HDR 3D Measurement.
"""

import importlib
import os
import shutil
import sys
import tempfile
import unittest
import cv2
import numpy as np

# Ensure root workspace is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dlp_common.fringe.multi_exposure import (
    fuse_multi_exposure_fringe_patterns,
    fuse_multi_exposure_phases,
)
from dlp_common.hardware.dlp_controller import DLP4710Controller

MeasurementController = importlib.import_module(
    "SL-3D-Measurement-DLP4710-SingleCamera.core.controller"
).MeasurementController
MultiExposureMeasureWorker = importlib.import_module(
    "SL-3D-Measurement-DLP4710-SingleCamera.core.workers"
).MultiExposureMeasureWorker
MultiExposureCaptureWorker = importlib.import_module(
    "SL-3D-Measurement-DLP4710-SingleCamera.core.workers"
).MultiExposureCaptureWorker


class TestProjectorBrightnessControl(unittest.TestCase):
    """Test TI DLPC3479 Command 0x54 LED Current & Brightness Scaling."""

    def setUp(self):
        self.dlp = DLP4710Controller()

    def test_default_brightness(self):
        self.assertEqual(self.dlp.brightness_percent, 100.0)
        self.assertEqual(self.dlp.led_currents, (1000, 1000, 1000))

    def test_set_brightness_percent(self):
        ok, msg = self.dlp.set_brightness_percent(50)
        self.assertTrue(ok)
        self.assertAlmostEqual(self.dlp.brightness_percent, 50.0, delta=0.5)
        r, g, b = self.dlp.led_currents
        self.assertEqual(r, g)
        self.assertEqual(g, b)
        # 50% between 51 and 1000 is ~526
        self.assertTrue(500 <= r <= 550)

    def test_set_brightness_extremes(self):
        # 0% turns off LEDs
        ok, msg = self.dlp.set_brightness_percent(0)
        self.assertTrue(ok)
        self.assertEqual(self.dlp.led_currents, (0, 0, 0))

        # 100% full safe brightness
        ok, msg = self.dlp.set_brightness_percent(100)
        self.assertTrue(ok)
        self.assertEqual(self.dlp.led_currents, (1000, 1000, 1000))

        # Values > 100 clamped to 100
        ok, msg = self.dlp.set_brightness_percent(150)
        self.assertTrue(ok)
        self.assertEqual(self.dlp.brightness_percent, 100.0)

    def test_set_led_current_direct(self):
        ok, msg = self.dlp.set_led_current(300, 400, 500)
        self.assertTrue(ok)
        self.assertEqual(self.dlp.led_currents, (300, 400, 500))

        # Out-of-bounds clamped to 0..1023
        ok, msg = self.dlp.set_led_current(-50, 1500, 800)
        self.assertTrue(ok)
        self.assertEqual(self.dlp.led_currents, (0, 1023, 800))


class TestMultiExposurePhaseFusion(unittest.TestCase):
    """Test HDR Phase Fusion with saturation detection and modulation maximization."""

    def test_single_exposure_passthrough(self):
        phase = np.full((10, 10), 1.5, dtype=np.float64)
        mod = np.full((10, 10), 20.0, dtype=np.float64)
        fused_phase, fused_mod, mask, idx_map = fuse_multi_exposure_phases([phase], [mod])
        np.testing.assert_array_equal(fused_phase, phase)
        np.testing.assert_array_equal(fused_mod, mod)
        self.assertTrue(np.all(mask == 255))
        self.assertTrue(np.all(idx_map == 0))

    def test_two_exposure_saturation_rejection(self):
        h, w = 20, 20
        # Phase truth
        true_phase = np.linspace(0.5, 2.5, h * w).reshape((h, w))

        # Exposure 0 (Short): low modulation, no saturation anywhere
        phase_0 = true_phase.copy()
        mod_0 = np.full((h, w), 8.0, dtype=np.float64)
        frames_0 = [np.full((h, w), 60, dtype=np.uint8) for _ in range(12)]

        # Exposure 1 (Long): high modulation, but left half is shiny/saturated (>= 250)
        phase_1 = true_phase.copy()
        mod_1 = np.full((h, w), 35.0, dtype=np.float64)
        frames_1 = [np.full((h, w), 180, dtype=np.uint8) for _ in range(12)]
        for f in frames_1:
            f[:, :10] = 254  # Saturated shiny highlight in left half

        fused_phase, fused_mod, mask, idx_map = fuse_multi_exposure_phases(
            phases=[phase_0, phase_1],
            modulations=[mod_0, mod_1],
            frames_list=[frames_0, frames_1],
            sat_threshold=250.0,
            min_modulation=5.0,
        )

        # Left half (saturated in exp 1) must be selected from exp 0
        self.assertTrue(np.all(idx_map[:, :10] == 0))
        # Right half (unsaturated, higher modulation in exp 1) must be selected from exp 1
        self.assertTrue(np.all(idx_map[:, 10:] == 1))

        # Overall fused phase must match ground truth across the entire image
        np.testing.assert_allclose(fused_phase, true_phase, rtol=1e-5)
        self.assertTrue(np.all(mask == 255))

    def test_fuse_fringe_patterns_jiang_zhu(self):
        """Test fringe pattern fusion strictly following Jiang [9] & Zhu [10]."""
        h, w = 16, 20
        # Low exposure: modulation ~10, baseline ~50
        frames_low = []
        for i in range(12):
            shift = (i % 4) * (np.pi / 2.0)
            img = 50 + 10 * np.cos(shift)
            frames_low.append(np.full((h, w), int(img), dtype=np.uint8))

        # High exposure: modulation ~40, baseline ~150, but saturated on left half
        frames_high = []
        for i in range(12):
            shift = (i % 4) * (np.pi / 2.0)
            img = np.full((h, w), 150 + 40 * np.cos(shift), dtype=np.float32)
            img[:, :10] = 255.0  # Saturated specular reflection
            frames_high.append(np.clip(img, 0, 255).astype(np.uint8))

        fused_frames, fused_mod, mask, idx_map = fuse_multi_exposure_fringe_patterns(
            brackets_frames=[frames_low, frames_high],
            sat_threshold=250.0,
            min_modulation=5.0,
        )

        self.assertEqual(len(fused_frames), 12)
        # Left half was saturated in high exposure, so it must be taken from frames_low
        for k in range(12):
            np.testing.assert_array_equal(fused_frames[k][:, :10], frames_low[k][:, :10])
            np.testing.assert_array_equal(fused_frames[k][:, 10:], frames_high[k][:, 10:])

        self.assertTrue(np.all(idx_map[:, :10] == 0))
        self.assertTrue(np.all(idx_map[:, 10:] == 1))

    def test_13_exposures_paper_gradient(self):
        """Test 13 exposure gradient brackets as conducted in paper section 4.2."""
        h, w = 12, 16
        # Generate 13 brackets with increasing brightness/contrast
        brackets = []
        for m in range(13):
            # Contrast increases from m=0 to m=12
            mod = 5.0 + m * 3.0
            base = 30 + m * 15
            frames_m = []
            for k in range(12):
                shift = (k % 4) * (np.pi / 2.0)
                val = base + mod * np.cos(shift)
                # Saturated if base + mod exceeds 250
                frames_m.append(np.clip(np.full((h, w), val, dtype=np.float32), 0, 255).astype(np.uint8))
            brackets.append(frames_m)

        fused_frames, fused_mod, mask, idx_map = fuse_multi_exposure_fringe_patterns(
            brackets_frames=brackets,
            sat_threshold=250.0,
            min_modulation=5.0,
        )
        self.assertEqual(len(fused_frames), 12)
        # Pixel should select the highest unsaturated bracket
        self.assertTrue(np.all(fused_mod >= 5.0))
        self.assertTrue(np.all(mask == 255))
        # The selected index should be >= 0 and < 13
        self.assertTrue(np.all((idx_map >= 0) & (idx_map < 13)))


class TestMultiExposureExportAndIO(unittest.TestCase):
    """Test saving and loading multi-exposure intermediate results."""

    def setUp(self):
        self.controller = MeasurementController()
        h, w = 16, 20
        bracket_1_frames = [np.full((h, w), 40 + i, dtype=np.uint8) for i in range(12)]
        bracket_2_frames = [np.full((h, w), 120 + i, dtype=np.uint8) for i in range(12)]

        wrapped_1 = [np.full((h, w), 0.5 + i, dtype=np.float32) for i in range(3)]
        wrapped_2 = [np.full((h, w), 0.8 + i, dtype=np.float32) for i in range(3)]

        unwrapped_1 = np.full((h, w), 10.0, dtype=np.float32)
        unwrapped_2 = np.full((h, w), 10.0, dtype=np.float32)
        fused_phase = np.full((h, w), 10.0, dtype=np.float32)

        mod_1 = np.full((h, w), 12.0, dtype=np.float32)
        mod_2 = np.full((h, w), 30.0, dtype=np.float32)
        fused_mod = np.full((h, w), 30.0, dtype=np.float32)

        mask = np.ones((h, w), dtype=bool)
        exp_indices = np.zeros((h, w), dtype=np.int32)
        exp_indices[:, 10:] = 1  # Bracket 1 for right half

        xyz_map = np.zeros((h, w, 3), dtype=np.float64)
        yy, xx = np.mgrid[:h, :w]
        xyz_map[..., 0] = xx
        xyz_map[..., 1] = yy
        xyz_map[..., 2] = 200.0

        self.controller.latest_result = {
            "is_multi_exposure": True,
            "brackets_frames": [bracket_1_frames, bracket_2_frames],
            "brackets_info": [
                {"exposure_us": 1000.0, "led_percent": 30.0},
                {"exposure_us": 4000.0, "led_percent": 80.0},
            ],
            "frames": bracket_1_frames,
            "wrapped_phases": wrapped_1,
            "all_wrapped_phases": [wrapped_1, wrapped_2],
            "all_unwrapped_phases": [unwrapped_1, unwrapped_2],
            "all_modulations": [mod_1, mod_2],
            "unwrapped_phase": fused_phase,
            "modulation": fused_mod,
            "mask": mask,
            "exposure_indices": exp_indices,
            "bracket_counts": {0: h * 10, 1: h * 10},
            "xyz_map": xyz_map,
            "point_count": h * w,
            "elapsed_seconds": 2.15,
        }

    def test_save_and_load_multi_exposure_results(self):
        with tempfile.TemporaryDirectory() as parent:
            out_dir = os.path.join(parent, "hdr_result")
            ok, msg = self.controller.save_intermediate_results(out_dir)
            self.assertTrue(ok, msg)

            # Check directory hierarchy
            self.assertTrue(os.path.isfile(os.path.join(out_dir, "measurement_info.txt")))
            self.assertTrue(os.path.isdir(os.path.join(out_dir, "01_fringes", "bracket_1")))
            self.assertTrue(os.path.isdir(os.path.join(out_dir, "01_fringes", "bracket_2")))

            # Check that root 01_fringes has 12 frames for backward compatibility
            self.assertTrue(os.path.isfile(os.path.join(out_dir, "01_fringes", "H_Fringe_1.bmp")))
            self.assertTrue(os.path.isfile(os.path.join(out_dir, "01_fringes", "bracket_1", "H_Fringe_12.bmp")))
            self.assertTrue(os.path.isfile(os.path.join(out_dir, "01_fringes", "bracket_2", "H_Fringe_12.bmp")))

            # Check auxiliary intermediate outputs are omitted (streamlined export)
            self.assertFalse(os.path.exists(os.path.join(out_dir, "04_auxiliary")))
            self.assertTrue(os.path.isfile(os.path.join(out_dir, "capture_parameters.json")))

            # Check point clouds (streamlined to 02_point_cloud, only original point cloud saved)
            self.assertTrue(os.path.isfile(os.path.join(out_dir, "02_point_cloud", "original_cloud.ply")))

            # Check measurement_info.txt contains HDR distribution
            with open(os.path.join(out_dir, "measurement_info.txt"), "r", encoding="utf-8") as f:
                content = f.read()
                self.assertIn("多曝光", content)
                self.assertIn("各曝光档位贡献点数", content)

            # Test offline multi-exposure loader
            ok_load, loaded_brackets, msg_load = self.controller.load_offline_multi_exposure_frames(out_dir)
            self.assertTrue(ok_load, msg_load)
            self.assertEqual(len(loaded_brackets), 2)
            self.assertEqual(len(loaded_brackets[0]), 12)
            self.assertEqual(len(loaded_brackets[1]), 12)


class TestMultiExposureWorker(unittest.TestCase):
    """Test MultiExposureMeasureWorker pipeline end-to-end."""

    def test_worker_run(self):
        from dlp_common.fringe.phase_unwrapping import PhaseProcessor

        class MockReconstructor:
            def reconstruct_from_phase(self, phase_map, mask=None, roi=None, sample_step=1, progress_callback=None):
                xyz = np.zeros((*phase_map.shape, 3), dtype=np.float64)
                xyz[..., 2] = 200.0  # valid points
                if progress_callback:
                    progress_callback(1.0)
                return True, xyz, "Mock Reconstruct OK"

        processor = PhaseProcessor()

        h, w = 60, 80
        # 12 synthetic frames for bracket 1 and bracket 2
        bracket_1 = [np.full((h, w), 80, dtype=np.uint8) for _ in range(12)]
        bracket_2 = [np.full((h, w), 140, dtype=np.uint8) for _ in range(12)]

        worker = MultiExposureMeasureWorker(
            reconstructor=MockReconstructor(),
            processor=processor,
            brackets_frames=[bracket_1, bracket_2],
            sample_step=1,
            min_modulation=0.0,
            brackets_info=[
                {"exposure_us": 1000.0, "led_percent": 40.0},
                {"exposure_us": 5000.0, "led_percent": 80.0},
            ],
        )

        results = []
        worker.finished.connect(lambda ok, res, msg: results.append((ok, res, msg)))
        worker.run()

        self.assertEqual(len(results), 1)
        ok, res, msg = results[0]
        self.assertTrue(ok, msg)
        self.assertTrue(res.get("is_multi_exposure"))
        self.assertIn("unwrapped_phase", res)
        self.assertIn("exposure_indices", res)
        self.assertIn("xyz_map", res)
        self.assertEqual(len(res["all_unwrapped_phases"]), 2)


class TestMultiExposureCaptureWorker(unittest.TestCase):
    """Test MultiExposureCaptureWorker with multiple brackets and automatic retry on partial frames."""

    def test_capture_worker_13_brackets_with_bracket4_partial_frame_recovery(self):
        h, w = 32, 48
        frames_12 = [np.full((h, w), 100, dtype=np.uint8) for _ in range(12)]
        frames_6 = [np.full((h, w), 100, dtype=np.uint8) for _ in range(6)]

        call_records = []
        bracket_attempts = {}

        class MockCamera:
            def __init__(self):
                self.last_error = ""
                self._current_bracket = 0
                self.exposure = 3000.0
                self.gain = 0.0
                self.is_opened = True

            def stop_grabbing(self):
                return True

            def start_grabbing(self):
                return True

            def set_trigger_mode(self, enabled, source="Software"):
                return True

            def set_exposure(self, exp_us):
                self.exposure = exp_us
                return True

            def set_gain(self, gain_db):
                self.gain = gain_db
                return True

            def grab_frame(self, timeout_ms=3000):
                return True, np.zeros((h, w), dtype=np.uint8)

            def flush_buffer(self):
                return True

            def grab_triggered_sequence(self, count, frame_timeout_ms=1500, overall_timeout_ms=15000, *args, **kwargs):
                call_records.append(count)
                # On bracket 4, attempt 1 returns 6 frames (partial), attempt 2 returns 12 frames
                current_b = getattr(self, "_current_bracket", 1)
                attempt = bracket_attempts.get(current_b, 0) + 1
                bracket_attempts[current_b] = attempt
                if current_b == 4 and attempt == 1:
                    return False, frames_6
                return True, frames_12

        class MockDlp:
            def __init__(self):
                self.last_error = ""
                self.brightness_percent = 100.0
                self.work_starts = 0

            def set_brightness_percent(self, pct):
                self.brightness_percent = pct
                return True, "OK"

            def start_work(self, repeat_num=0):
                self.work_starts += 1
                return True

        cam = MockCamera()
        dlp = MockDlp()

        # 13 brackets gradient
        brackets = [
            {"exposure_us": 3000.0, "led_percent": float(pct), "gain_db": 0.0}
            for pct in np.linspace(10, 100, 13)
        ]

        worker = MultiExposureCaptureWorker(
            camera=cam,
            dlp=dlp,
            brackets=brackets,
            gain_db=0.0,
            gray_exposure_us=50000.0,
        )

        # Hook progress to update current bracket tracking in mock camera
        def on_prog(val, text):
            for i in range(1, 14):
                if f"档位 {i}/13" in text:
                    cam._current_bracket = i

        worker.progress.connect(on_prog)

        results = []
        worker.completed.connect(lambda ok, brackets_frames, msg: results.append((ok, brackets_frames, msg)))
        worker.run()

        self.assertEqual(len(results), 1)
        ok, all_frames, msg = results[0]
        self.assertTrue(ok, msg)
        self.assertEqual(len(all_frames), 13)
        for i, b_frames in enumerate(all_frames):
            self.assertEqual(len(b_frames), 12, f"Bracket {i+1} does not have 12 frames")

        # Verify bracket 4 was attempted twice and recovered
        self.assertEqual(bracket_attempts[4], 2)
        # Total dlp start_work should be 14 (13 brackets + 1 retry on bracket 4)
        self.assertEqual(dlp.work_starts, 14)


if __name__ == "__main__":
    unittest.main()
