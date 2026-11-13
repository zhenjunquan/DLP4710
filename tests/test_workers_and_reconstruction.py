"""
Tests for Workers (MeasureWorker, MultiExposureMeasureWorker, BurnMeasurementBinWorker)
verifying object_mask support, fast multi-exposure pipeline, and flash burning.
"""

import importlib
import os
import sys
import unittest
import numpy as np

from PyQt6.QtWidgets import QApplication

from dlp_common.fringe.phase_unwrapping import PhaseProcessor
from dlp_common.hardware import DLP4710Controller
from dlp_common.reconstruction.reconstructor import MonoSLReconstructor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
_workers_mod = importlib.import_module("SL-3D-Measurement-DLP4710-SingleCamera.core.workers")
MeasureWorker = _workers_mod.MeasureWorker
MultiExposureMeasureWorker = _workers_mod.MultiExposureMeasureWorker
BurnMeasurementBinWorker = _workers_mod.BurnMeasurementBinWorker


class DummyReconstructor:
    def __init__(self):
        self.enable_gpu = False

    def reconstruct_from_phase(self, phase_map, mask, roi=None, sample_step=4, progress_callback=None):
        h, w = phase_map.shape
        xyz_map = np.zeros((h, w, 3), dtype=np.float32)
        # Create non-zero coordinates only where mask > 0
        valid_indices = np.where(mask > 0)
        xyz_map[valid_indices[0], valid_indices[1], 0] = valid_indices[1].astype(np.float32)
        xyz_map[valid_indices[0], valid_indices[1], 1] = valid_indices[0].astype(np.float32)
        xyz_map[valid_indices[0], valid_indices[1], 2] = 50.0
        if progress_callback:
            progress_callback(1.0)
        return True, xyz_map, "Dummy success"


class TestWorkersAndReconstruction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        self.h, self.w = 120, 160
        self.processor = PhaseProcessor()
        self.reconstructor = DummyReconstructor()

        # Generate 12 synthetic fringe frames
        x = np.arange(self.w)
        self.frames = []
        for f in (70, 64, 59):
            for step in range(4):
                phase_shift = step * np.pi / 2.0
                fringe_row = 128.0 + 80.0 * np.cos(2 * np.pi * f * x / self.w + phase_shift)
                frame = np.tile(fringe_row, (self.h, 1)).astype(np.uint8)
                self.frames.append(frame)

    def test_measure_worker_reconstruction(self):
        worker = MeasureWorker(
            reconstructor=self.reconstructor,
            processor=self.processor,
            frames=self.frames,
            roi=None,
            sample_step=1,
            min_modulation=5.0,
        )
        res = {}
        worker.finished.connect(lambda ok, r, m: res.update(r))
        worker.run()

        self.assertIn("valid_points", res)
        self.assertGreater(len(res["valid_points"]), 100)
        self.assertIn("unwrapped_phase", res)
        self.assertIn("modulation", res)

    def test_multi_exposure_worker_fast_pipeline(self):
        # Create 3 brackets
        brackets = [self.frames, self.frames, self.frames]
        worker = MultiExposureMeasureWorker(
            reconstructor=self.reconstructor,
            processor=self.processor,
            brackets_frames=brackets,
            roi=None,
            sample_step=2,
            min_modulation=5.0,
            compute_per_bracket_phases=False,  # Core optimization: bypass redundant 13-bracket loop!
        )
        res = {}
        worker.finished.connect(lambda ok, r, m: res.update(r))
        worker.run()

        self.assertIn("valid_points", res)
        self.assertGreater(len(res["valid_points"]), 0)
        self.assertTrue(res["is_multi_exposure"])
        self.assertIn("fused_frames", res)

    def test_burn_bin_worker_signals(self):
        dlp = DLP4710Controller(dll_path="nonexistent.dll")
        dlp.connect()

        from dlp_common.hardware.patn_toolkit import build_standard_phase_shift_bin
        bin_bytes, _ = build_standard_phase_shift_bin()
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(bin_bytes)
            tmp_path = f.name

        try:
            worker = BurnMeasurementBinWorker(dlp, tmp_path, verify_projection=True)
            signals = {"progress": [], "verified": [], "completed": []}
            worker.progress.connect(lambda p, m: signals["progress"].append(p))
            worker.verified.connect(lambda ok, r, m: signals["verified"].append(ok))
            worker.completed.connect(lambda ok, m: signals["completed"].append(ok))
            worker.run()

            self.assertGreater(len(signals["progress"]), 0)
            self.assertEqual(signals["verified"], [True])
            self.assertEqual(signals["completed"], [True])
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
