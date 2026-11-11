"""
GPU (CUDA / PyTorch) Accelerated Phase Solving and Heterodyne Phase Unwrapping.
Provides massively parallel 2D separable Gaussian blur, 4-step phase shifting,
and 3-frequency heterodyne phase unwrapping.
100% numerically equivalent to PhaseProcessor on CPU.
"""

import math
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np


def is_cuda_available() -> bool:
    """Checks whether PyTorch with CUDA support is available."""
    try:
        import torch

        return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
    except Exception:
        return False


def get_gaussian_kernel_tensors(
    kernel_size: Tuple[int, int] = (7, 7),
    sigma: float = 1.57,
    device_str: str = "cuda:0",
):
    """Generates 1D separable Gaussian kernel tensors for PyTorch Conv2D."""
    import torch

    kx = cv2.getGaussianKernel(kernel_size[0], sigma).ravel()
    ky = cv2.getGaussianKernel(kernel_size[1], sigma).ravel()
    t_kx = torch.from_numpy(kx).to(device=device_str, dtype=torch.float32).view(1, 1, 1, -1)
    t_ky = torch.from_numpy(ky).to(device=device_str, dtype=torch.float32).view(1, 1, -1, 1)
    return t_kx, t_ky


def cuda_solve_wrapped_phases(
    fringe_frames: List[np.ndarray],
    periods: List[float],
    step_num: int = 4,
    blur_kernel_size: Tuple[int, int] = (7, 7),
    blur_sigma: float = 1.57,
    roi: Optional[Tuple[int, int, int, int]] = None,
    device_str: str = "cuda:0",
    return_tensors: bool = False,
) -> Tuple[List[Union[np.ndarray, "torch.Tensor"]], Union[np.ndarray, "torch.Tensor"]]:
    """
    Massively parallel GPU calculation of wrapped phases [0, 1) and average modulation.
    fringe_frames: List of 12 images (3 frequencies x 4 steps).
    roi: Optional (rx, ry, rw, rh).
    """
    import torch
    import torch.nn.functional as F

    device = torch.device(device_str)
    num_frames = len(fringe_frames)
    expected = len(periods) * step_num
    if num_frames != expected:
        raise ValueError(f"Expected {expected} frames, got {num_frames}.")

    sample_frame = fringe_frames[0]
    full_h, full_w = sample_frame.shape[:2]

    if roi is not None:
        rx, ry, rw, rh = roi
        rx = max(0, min(rx, full_w - 1))
        ry = max(0, min(ry, full_h - 1))
        rw = max(1, min(rw, full_w - rx))
        rh = max(1, min(rh, full_h - ry))
    else:
        rx, ry, rw, rh = 0, 0, full_w, full_h

    # Stack cropped images into tensor (N, H, W)
    cropped_list = []
    for f in fringe_frames:
        if f.ndim == 3:
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        else:
            gray = f
        cropped_list.append(gray[ry : ry + rh, rx : rx + rw])

    stacked_np = np.stack(cropped_list, axis=0)  # (12, H, W)
    t_frames = torch.from_numpy(stacked_np).to(device=device, dtype=torch.float32)

    # Separable 2D Gaussian blur for all 12 frames simultaneously
    pad_x = blur_kernel_size[0] // 2
    pad_y = blur_kernel_size[1] // 2
    t_kx, t_ky = get_gaussian_kernel_tensors(blur_kernel_size, blur_sigma, device_str)

    input_4d = t_frames.unsqueeze(1)  # (12, 1, H, W)
    padded = F.pad(input_4d, (pad_x, pad_x, pad_y, pad_y), mode="reflect")
    blurred_raw = F.conv2d(F.conv2d(padded, t_kx), t_ky).squeeze(1)  # (12, H, W)
    # Match OpenCV cv2.GaussianBlur(uint8, ...) rounding exactly
    blurred = torch.clamp(torch.round(blurred_raw), 0.0, 255.0).to(torch.float64)

    # Group into (num_freqs, 4, H, W)
    num_freqs = len(periods)
    freq_groups = blurred.view(num_freqs, step_num, rh, rw)
    # up_diff = I_3 - I_1
    # down_diff = I_0 - I_2
    up_diff = freq_groups[:, 3] - freq_groups[:, 1]
    down_diff = freq_groups[:, 0] - freq_groups[:, 2]

    # Modulation amplitude
    mods = 0.5 * torch.sqrt(up_diff**2 + down_diff**2)
    avg_modulation = torch.mean(mods, dim=0)

    # Phase in [0, 2*pi) matching C++ atan2 quadrant mapping
    two_pi = 2.0 * math.pi
    phases = (torch.atan2(up_diff, down_diff) + two_pi) % two_pi
    norm_phases = phases / two_pi  # in [0, 1)

    if return_tensors:
        wrapped_list = [norm_phases[i] for i in range(num_freqs)]
        return wrapped_list, avg_modulation

    wrapped_np = [norm_phases[i].cpu().numpy().astype(np.float64) for i in range(num_freqs)]
    avg_mod_np = avg_modulation.cpu().numpy().astype(np.float64)
    return wrapped_np, avg_mod_np


def cuda_unwrap_wrapped_phases(
    wrapped_phases: List[Union[np.ndarray, "torch.Tensor"]],
    periods: List[float],
    device_str: str = "cuda:0",
    return_tensor: bool = False,
) -> Union[np.ndarray, "torch.Tensor"]:
    """
    Massively parallel GPU 3-frequency heterodyne phase unwrapping.
    Numerically identical to PhaseProcessor.unwrap_wrapped_phases.
    """
    import torch

    device = torch.device(device_str)
    if len(wrapped_phases) != 3:
        raise ValueError(f"Expected 3 wrapped phase maps, got {len(wrapped_phases)}.")

    # Convert to tensors if needed
    t_phases = []
    for p in wrapped_phases:
        if isinstance(p, torch.Tensor):
            t_phases.append(p.to(device=device, dtype=torch.float64))
        else:
            t_phases.append(torch.from_numpy(np.asarray(p, dtype=np.float64)).to(device=device))

    phase1, phase2, phase3 = t_phases[0], t_phases[1], t_phases[2]

    # Phase differences in [0, 1):
    diff_1_2 = phase1 - phase2
    diff_1_2 = torch.where(diff_1_2 >= -1e-5, diff_1_2, 1.0 + diff_1_2)

    diff_2_3 = phase2 - phase3
    diff_2_3 = torch.where(diff_2_3 >= -1e-5, diff_2_3, 1.0 + diff_2_3)

    diff_12_23 = diff_1_2 - diff_2_3
    diff_12_23 = torch.where(diff_12_23 >= -1e-5, diff_12_23, 1.0 + diff_12_23)

    # Periods: p = 1 / lambda
    p1 = 1.0 / periods[0]
    p2 = 1.0 / periods[1]
    p3 = 1.0 / periods[2]

    p12 = p1 * p2 / (p2 - p1)
    p23 = p2 * p3 / (p3 - p2)

    # Equivalent phase order 12 and 23
    phase12_eq_raw = p23 * diff_12_23 / (p23 - p12) - diff_1_2
    k12 = torch.round(phase12_eq_raw)
    phase12_eq = k12 + diff_1_2

    phase23_eq_raw = p12 * diff_12_23 / (p23 - p12) - diff_2_3
    k23 = torch.round(phase23_eq_raw)
    phase23_eq = k23 + diff_2_3

    phase12_23_eq = (p12 * phase12_eq + p23 * phase23_eq) / 2.0

    # Fringe order for each individual frequency
    k1 = torch.round(phase12_23_eq / p1 - phase1)
    k2 = torch.round(phase12_23_eq / p2 - phase2)
    k3 = torch.round(phase12_23_eq / p3 - phase3)

    # Weighted least-squares phase combination
    sum1 = (1.0 / 3.0) * p1 * k1 + (p1 / 3.0) * phase1
    sum2 = (1.0 / 3.0) * p2 * k2 + (p2 / 3.0) * phase2
    sum3 = (1.0 / 3.0) * p3 * k3 + (p3 / 3.0) * phase3

    unwrapped = (1.0 / p1) * (sum1 + sum2 + sum3)

    if return_tensor:
        return unwrapped
    return unwrapped.cpu().numpy()


def cuda_compute_mask(
    modulation: Union[np.ndarray, "torch.Tensor"],
    min_modulation: float = 5.0,
    device_str: str = "cuda:0",
    return_tensor: bool = False,
) -> Union[np.ndarray, "torch.Tensor"]:
    """Computes binary uint8 mask (0 or 255) on GPU based on modulation threshold."""
    import torch

    device = torch.device(device_str)
    if isinstance(modulation, torch.Tensor):
        mod_t = modulation.to(device=device, dtype=torch.float32)
    else:
        mod_t = torch.from_numpy(np.asarray(modulation, dtype=np.float32)).to(device=device)

    mask = (mod_t >= min_modulation).to(torch.uint8) * 255
    if return_tensor:
        return mask
    return mask.cpu().numpy()
