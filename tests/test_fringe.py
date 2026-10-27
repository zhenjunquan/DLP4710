"""
Unit tests for fringe generation and multi-frequency phase unwrapping.
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dlp_common.fringe.generator import FringeGenerator
from dlp_common.fringe.phase_unwrapping import PhaseProcessor
from dlp_common.fringe.utils import bilinear_sample, bilinear_sample_points


class TestFringeProcessing(unittest.TestCase):
    def setUp(self):
        self.periods = [70.0, 64.0, 59.0]
        self.gen = FringeGenerator(width=192, height=108, periods=self.periods, step_num=4)
        self.processor = PhaseProcessor(periods=self.periods, step_num=4)

    def test_horizontal_fringe_generation(self):
        frames = self.gen.generate_directional_fringes(direction=0)
        self.assertEqual(len(frames), 12)
        for f in frames:
            self.assertEqual(f.shape, (108, 192))
            self.assertEqual(f.dtype, np.uint8)

    def test_vertical_fringe_generation(self):
        frames = self.gen.generate_directional_fringes(direction=1)
        self.assertEqual(len(frames), 12)
        for f in frames:
            self.assertEqual(f.shape, (108, 192))
            self.assertEqual(f.dtype, np.uint8)

    def test_phase_unwrapping_monotonicity(self):
        # Full 1080p horizontal stripes vary monotonically along vertical axis (y)
        full_gen = FringeGenerator(width=1920, height=1080, periods=self.periods, step_num=4)
        h_frames = full_gen.generate_directional_fringes(direction=0)
        unwrapped, mod = self.processor.unwrap_phase(h_frames)
        self.assertEqual(unwrapped.shape, (1080, 1920))

        # Check interior column-wise monotonicity along Y (avoiding Gaussian blur border effects)
        col_slice = unwrapped[10:1060, 960]
        diffs = np.diff(col_slice)
        self.assertTrue(np.all(diffs > 0), "Horizontal phase unwrapping should be strictly monotonic along column")

        wrapped, modulation = self.processor.solve_wrapped_phases(h_frames)
        from_wrapped = self.processor.unwrap_wrapped_phases(wrapped)
        np.testing.assert_allclose(from_wrapped, unwrapped, atol=1e-12)
        self.assertEqual(modulation.shape, unwrapped.shape)

    def test_bilinear_sample(self):
        mat = np.arange(100, dtype=np.float64).reshape(10, 10)
        # Center of (0,0) and (0,1)
        sampled = bilinear_sample(mat, 0.5, 0.0)
        self.assertAlmostEqual(sampled, 0.5, places=5)

        # Boundary checks
        nan_val = bilinear_sample(mat, -1.0, 5.0)
        self.assertTrue(np.isnan(nan_val))


if __name__ == "__main__":
    unittest.main()
