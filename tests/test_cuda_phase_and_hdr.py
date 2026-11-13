"""
Unit tests for GPU (CUDA) accelerated phase calculation, heterodyne unwrapping,
and multi-exposure HDR fusion.
Verifies numerical consistency between CPU and GPU pipelines.
"""

import unittest
import numpy as np
from dlp_common.fringe.phase_unwrapping import PhaseProcessor
from dlp_common.fringe.multi_exposure import fuse_multi_exposure_fringe_patterns
from dlp_common.fringe.cuda_phase import (
    is_cuda_available,
    cuda_solve_wrapped_phases,
    cuda_unwrap_wrapped_phases,
    cuda_compute_mask,
)
from dlp_common.fringe.cuda_multi_exposure import cuda_fuse_multi_exposure_fringe_patterns


class TestCudaPhaseAndHdr(unittest.TestCase):
    """Verifies GPU acceleration against CPU baseline."""

    def setUp(self):
        np.random.seed(42)
        self.periods = [70.0, 64.0, 59.0]
        self.h, self.w = 120, 160

        # Create realistic synthetic 12-frame fringes (3 freq x 4 step)
        self.frames = []
        y, x = np.mgrid[0 : self.h, 0 : self.w]
        for p in self.periods:
            phase_ideal = 2.0 * np.pi * x / p
            for s in range(4):
                fringe = 128.0 + 80.0 * np.cos(phase_ideal - s * np.pi * 0.5)
                fringe += np.random.normal(0, 1.0, (self.h, self.w))
                self.frames.append(np.clip(fringe, 0, 255).astype(np.uint8))

    def test_cuda_wrapped_phase_consistency(self):
        if not is_cuda_available():
            self.skipTest("CUDA GPU is not available on this machine")

        # 1. CPU solve
        cpu_proc = PhaseProcessor(periods=self.periods, enable_gpu=False)
        cpu_wrapped, cpu_mod = cpu_proc.solve_wrapped_phases(self.frames)

        # 2. GPU solve
        gpu_proc = PhaseProcessor(periods=self.periods, enable_gpu=True)
        gpu_wrapped, gpu_mod = gpu_proc.solve_wrapped_phases(self.frames)

        # Check modulation
        mod_diff = np.max(np.abs(cpu_mod - gpu_mod))
        self.assertLess(mod_diff, 1.0, f"Modulation diff {mod_diff:.4f} exceeds threshold")

        # Check wrapped phases
        for f_idx in range(3):
            # Phase is in [0, 1), check cyclic diff
            diff = np.abs(cpu_wrapped[f_idx] - gpu_wrapped[f_idx])
            diff = np.minimum(diff, 1.0 - diff)
            self.assertLess(np.max(diff), 0.05, f"Freq {f_idx} phase diff too large")
            self.assertLess(np.mean(diff), 0.005, f"Freq {f_idx} mean phase diff too large")

    def test_cuda_heterodyne_unwrapped_consistency(self):
        if not is_cuda_available():
            self.skipTest("CUDA GPU is not available on this machine")

        cpu_proc = PhaseProcessor(periods=self.periods, enable_gpu=False)
        gpu_proc = PhaseProcessor(periods=self.periods, enable_gpu=True)

        # 1. Exact mathematical identity test on identical wrapped phase inputs
        w_cpu, _ = cpu_proc.solve_wrapped_phases(self.frames)
        u_cpu_from_same = cpu_proc.unwrap_wrapped_phases(w_cpu)
        u_gpu_from_same = cuda_unwrap_wrapped_phases(w_cpu, self.periods)
        self.assertLess(np.max(np.abs(u_cpu_from_same - u_gpu_from_same)), 1e-4)

        # 2. End-to-end solve and unwrap pipeline
        w_gpu, _ = gpu_proc.solve_wrapped_phases(self.frames)
        u_gpu = gpu_proc.unwrap_wrapped_phases(w_gpu)

        diff = np.abs(u_cpu_from_same - u_gpu)
        same_order_ratio = np.mean(diff < 1.0)
        self.assertGreater(same_order_ratio, 0.99, f"Same fringe order ratio {same_order_ratio:.4f} too low")
        self.assertLess(np.mean(diff[diff < 1.0]), 1e-3)

    def test_cuda_multi_exposure_fusion_consistency(self):
        if not is_cuda_available():
            self.skipTest("CUDA GPU is not available on this machine")

        # Create 3 brackets: low exposure (0.4x), medium (1.0x), high exposure (1.8x, saturating)
        bracket_1 = [np.clip(f.astype(np.float32) * 0.4, 0, 255).astype(np.uint8) for f in self.frames]
        bracket_2 = [f.copy() for f in self.frames]
        bracket_3 = [np.clip(f.astype(np.float32) * 1.8, 0, 255).astype(np.uint8) for f in self.frames]
        brackets = [bracket_1, bracket_2, bracket_3]

        # 1. CPU fusion
        cpu_frames, cpu_mod, cpu_mask, cpu_idx = fuse_multi_exposure_fringe_patterns(
            brackets_frames=brackets,
            sat_threshold=250.0,
            min_modulation=5.0,
        )

        # 2. GPU fusion
        gpu_frames, gpu_mod, gpu_mask, gpu_idx = cuda_fuse_multi_exposure_fringe_patterns(
            brackets_frames=brackets,
            sat_threshold=250.0,
            min_modulation=5.0,
        )

        # Optimal exposure index must match identically
        idx_match = np.mean(cpu_idx == gpu_idx)
        self.assertGreater(idx_match, 0.99, f"Index match ratio {idx_match:.4f} is too low")

        # Fused modulation must match closely
        mod_diff = np.max(np.abs(cpu_mod - gpu_mod))
        self.assertLess(mod_diff, 1e-4)

        # Masks should match identically
        mask_match = np.mean(cpu_mask == gpu_mask)
        self.assertGreater(mask_match, 0.99)

        # Fused fringe frames must match
        for k in range(12):
            diff_k = np.abs(cpu_frames[k].astype(np.int32) - gpu_frames[k].astype(np.int32))
            self.assertLess(np.max(diff_k), 2)

    def test_cpu_fallback_when_gpu_disabled(self):
        # Explicitly disable GPU
        proc = PhaseProcessor(periods=self.periods, enable_gpu=False)
        self.assertFalse(proc.enable_gpu)
        wrapped, mod = proc.solve_wrapped_phases(self.frames)
        self.assertEqual(len(wrapped), 3)
        unwrapped = proc.unwrap_wrapped_phases(wrapped)
        self.assertEqual(unwrapped.shape, (self.h, self.w))


if __name__ == "__main__":
    unittest.main()
