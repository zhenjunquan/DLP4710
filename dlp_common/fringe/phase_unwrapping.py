"""
Phase calculation and 3-frequency 4-step phase unwrapping.
Ported with exact numerical fidelity from fringeProcessing.cpp.
"""

from typing import List, Optional, Tuple
import cv2
import numpy as np

try:
    from .cuda_phase import (
        is_cuda_available,
        cuda_solve_wrapped_phases,
        cuda_unwrap_wrapped_phases,
        cuda_compute_mask,
    )
except ImportError:
    def is_cuda_available() -> bool:
        return False
    cuda_solve_wrapped_phases = None
    cuda_unwrap_wrapped_phases = None
    cuda_compute_mask = None


class PhaseProcessor:
    """
    Solves wrapped phase and unwraps 3-frequency 4-step fringe patterns.
    Supports transparent NVIDIA GPU (CUDA) acceleration with CPU fallback.
    """

    def __init__(
        self,
        periods: List[float] = None,
        step_num: int = 4,
        blur_kernel_size: Tuple[int, int] = (7, 7),
        blur_sigma: float = 1.57,
        enable_gpu: bool = True,
    ):
        self.periods = list(periods) if periods is not None else [70.0, 64.0, 59.0]
        self.step_num = step_num
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma
        self.enable_gpu = enable_gpu

    def solve_wrapped_phases(
        self,
        fringe_frames: List[np.ndarray],
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Calculates normalized wrapped phase [0, 1) for each frequency group.
        Also calculates average modulation intensity.

        fringe_frames: List of 12 images (3 frequencies x 4 steps).
        roi: Optional (x, y, w, h). If None, full image.

        Returns:
          wrapped_phases: List of 3 float64 arrays in range [0, 1).
          modulation: float64 array of modulation amplitude.
        """
        if len(fringe_frames) != len(self.periods) * self.step_num:
            raise ValueError(
                f"Expected {len(self.periods) * self.step_num} frames, got {len(fringe_frames)}."
            )

        if self.enable_gpu and is_cuda_available() and cuda_solve_wrapped_phases is not None:
            try:
                return cuda_solve_wrapped_phases(
                    fringe_frames=fringe_frames,
                    periods=self.periods,
                    step_num=self.step_num,
                    blur_kernel_size=self.blur_kernel_size,
                    blur_sigma=self.blur_sigma,
                    roi=roi,
                )
            except Exception:
                pass

        h, w = fringe_frames[0].shape[:2]
        if roi is not None:
            rx, ry, rw, rh = roi
            rx = max(0, min(rx, w - 1))
            ry = max(0, min(ry, h - 1))
            rw = max(1, min(rw, w - rx))
            rh = max(1, min(rh, h - ry))
        else:
            rx, ry, rw, rh = 0, 0, w, h

        wrapped_phases: List[np.ndarray] = []
        modulations: List[np.ndarray] = []

        for freq_idx in range(len(self.periods)):
            # Load and blur the 4 phase-shifted frames
            i_mats = []
            for s in range(self.step_num):
                frame = fringe_frames[freq_idx * self.step_num + s]
                if frame.ndim == 3:
                    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    frame_gray = frame

                # Crop to ROI if specified
                roi_gray = frame_gray[ry : ry + rh, rx : rx + rw]

                # Gaussian blur matching C++ GaussianBlur(img, img, Size(7, 7), 1.57)
                blurred = cv2.GaussianBlur(
                    roi_gray, self.blur_kernel_size, self.blur_sigma
                ).astype(np.float64)
                i_mats.append(blurred)

            # 4-step phase shifting:
            # upDiff = I_3 - I_1
            # downDiff = I_0 - I_2
            up_diff = i_mats[3] - i_mats[1]
            down_diff = i_mats[0] - i_mats[2]

            # Modulation amplitude for this frequency: 0.5 * sqrt(upDiff^2 + downDiff^2)
            mod = 0.5 * np.sqrt(up_diff**2 + down_diff**2)
            modulations.append(mod)

            # Phase in [0, 2*pi) matching C++ atan2 quadrant mapping
            phase = (np.arctan2(up_diff, down_diff) + 2.0 * np.pi) % (2.0 * np.pi)

            # Normalized to [0, 1) matching C++ phaseMatVec[i] = phaseMat / 2.0 / PI
            norm_phase = phase / (2.0 * np.pi)
            wrapped_phases.append(norm_phase)

        avg_modulation = np.mean(modulations, axis=0)
        return wrapped_phases, avg_modulation

    def solve_wrapped_phases_from_files(
        self,
        frame_paths: List[str],
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """Decode disk-backed frames while loading only four images at a time."""
        expected = len(self.periods) * self.step_num
        if len(frame_paths) != expected:
            raise ValueError(f"Expected {expected} frame paths, got {len(frame_paths)}.")

        if self.enable_gpu and is_cuda_available():
            frames = []
            for path in frame_paths:
                frame = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if frame is None:
                    raise OSError(f"条纹图像无法读取: {path}")
                frames.append(frame)
            return self.solve_wrapped_phases(frames, roi=roi)

        image_h, image_w = None, None
        rx, ry, rw, rh = 0, 0, 0, 0

        wrapped_phases = []
        modulations = []
        for frequency_index in range(len(self.periods)):
            mats = []
            start = frequency_index * self.step_num
            for path in frame_paths[start : start + self.step_num]:
                frame = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if frame is None:
                    raise OSError(f"条纹图像无法读取: {path}")

                if image_h is None:
                    image_h, image_w = frame.shape[:2]
                    if roi is None:
                        rx, ry, rw, rh = 0, 0, image_w, image_h
                    else:
                        rx, ry, rw, rh = roi
                        rx = max(0, min(rx, image_w - 1))
                        ry = max(0, min(ry, image_h - 1))
                        rw = max(1, min(rw, image_w - rx))
                        rh = max(1, min(rh, image_h - ry))
                elif frame.shape[:2] != (image_h, image_w):
                    raise OSError(f"条纹图像尺寸不一致: {path}")

                cropped = frame[ry : ry + rh, rx : rx + rw]
                mats.append(
                    cv2.GaussianBlur(cropped, self.blur_kernel_size, self.blur_sigma).astype(np.float64)
                )

            up_diff = mats[3] - mats[1]
            down_diff = mats[0] - mats[2]
            modulations.append(0.5 * np.sqrt(up_diff**2 + down_diff**2))
            phase = (np.arctan2(up_diff, down_diff) + 2.0 * np.pi) % (2.0 * np.pi)
            wrapped_phases.append(phase / (2.0 * np.pi))

        return wrapped_phases, np.mean(modulations, axis=0)

    def unwrap_phase(
        self,
        fringe_frames: List[np.ndarray],
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Executes the 3-frequency 4-step heterodyne phase unwrapping.
        Returns:
          unwrapped_phase: Absolute phase map referenced to fringe_periods[0].
          modulation: Average modulation map.
        """
        wrapped, modulation = self.solve_wrapped_phases(fringe_frames, roi)
        return self.unwrap_wrapped_phases(wrapped), modulation

    def unwrap_wrapped_phases(self, wrapped_phases: List[np.ndarray]) -> np.ndarray:
        """Unwrap three already-computed normalized phase maps."""
        if len(wrapped_phases) != 3:
            raise ValueError(f"Expected 3 wrapped phase maps, got {len(wrapped_phases)}.")

        if self.enable_gpu and is_cuda_available() and cuda_unwrap_wrapped_phases is not None:
            try:
                return cuda_unwrap_wrapped_phases(
                    wrapped_phases=wrapped_phases,
                    periods=self.periods,
                )
            except Exception:
                pass

        wrapped = [np.asarray(phase, dtype=np.float64) for phase in wrapped_phases]
        if wrapped[1].shape != wrapped[0].shape or wrapped[2].shape != wrapped[0].shape:
            raise ValueError("Wrapped phase maps must have identical shapes.")

        phase1, phase2, phase3 = wrapped[0], wrapped[1], wrapped[2]

        # Phase differences in [0, 1):
        diff_1_2 = phase1 - phase2
        diff_1_2 = np.where(diff_1_2 >= -1e-5, diff_1_2, 1.0 + diff_1_2)
        diff_2_3 = phase2 - phase3
        diff_2_3 = np.where(diff_2_3 >= -1e-5, diff_2_3, 1.0 + diff_2_3)
        diff_12_23 = diff_1_2 - diff_2_3
        diff_12_23 = np.where(diff_12_23 >= -1e-5, diff_12_23, 1.0 + diff_12_23)

        # Periods: p = 1 / lambda
        p1 = 1.0 / self.periods[0]
        p2 = 1.0 / self.periods[1]
        p3 = 1.0 / self.periods[2]

        p12 = p1 * p2 / (p2 - p1)
        p23 = p2 * p3 / (p3 - p2)
        p123 = p12 * p23 / (p23 - p12)

        # Equivalent phase order 12 and 23
        phase12_eq_raw = p23 * diff_12_23 / (p23 - p12) - diff_1_2
        k12 = np.round(phase12_eq_raw)
        phase12_eq = k12 + diff_1_2

        phase23_eq_raw = p12 * diff_12_23 / (p23 - p12) - diff_2_3
        k23 = np.round(phase23_eq_raw)
        phase23_eq = k23 + diff_2_3

        phase12_23_eq = (p12 * phase12_eq + p23 * phase23_eq) / 2.0

        # Fringe order for each individual frequency
        k1 = np.round(phase12_23_eq / p1 - phase1)
        k2 = np.round(phase12_23_eq / p2 - phase2)
        k3 = np.round(phase12_23_eq / p3 - phase3)

        # Weighted least-squares phase combination
        sum1 = (1.0 / 3.0) * p1 * k1 + (p1 / 3.0) * phase1
        sum2 = (1.0 / 3.0) * p2 * k2 + (p2 / 3.0) * phase2
        sum3 = (1.0 / 3.0) * p3 * k3 + (p3 / 3.0) * phase3

        return (1.0 / p1) * (sum1 + sum2 + sum3)

    def compute_mask(
        self,
        modulation: np.ndarray,
        min_modulation: float = 5.0,
    ) -> np.ndarray:
        """
        Computes a binary uint8 mask (0 or 255) based on modulation threshold.
        """
        if self.enable_gpu and is_cuda_available() and cuda_compute_mask is not None:
            try:
                return cuda_compute_mask(modulation, min_modulation=min_modulation)
            except Exception:
                pass
        mask = (modulation >= min_modulation).astype(np.uint8) * 255
        return mask

