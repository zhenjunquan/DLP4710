"""
Multi-Exposure HDR Structured Light Phase & Fringe Fusion.
Implements the multi-exposure fusion algorithm from:
  - Jiang et al., "High dynamic range fringe projection 3D measurement...", Opt. Lasers Eng. 2012 [9]
  - Zhu et al., "High-speed high-dynamic-range 3D shape measurement...", Opt. Express 2023 [10]

Principle:
  Fixed camera exposure, step-wise projector light intensity (or multi-bracket exposure).
  For each pixel:
    1. Saturation rejection: If max(I_{m, n}(x, y)) >= sat_threshold, discard bracket m.
    2. Maximum modulation selection: m*(x, y) = argmax_{unsaturated} B_m(x, y).
    3. Fringe & Phase fusion: Assemble fused fringe patterns I^{fused} and compute absolute phase.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    from .cuda_phase import is_cuda_available
    from .cuda_multi_exposure import cuda_fuse_multi_exposure_fringe_patterns
except ImportError:
    def is_cuda_available() -> bool:
        return False
    cuda_fuse_multi_exposure_fringe_patterns = None


def compute_sequence_modulation(fringe_frames: List[np.ndarray]) -> np.ndarray:
    """
    Computes four-step modulation B for 12 fringe frames (3 frequencies x 4 steps).
    B = 0.5 * sqrt((I4 - I2)^2 + (I1 - I3)^2)
    Returns the average modulation across the 3 frequencies.
    """
    mods = []
    num_freqs = len(fringe_frames) // 4
    for freq in range(max(1, num_freqs)):
        i1 = fringe_frames[freq * 4 + 0].astype(np.float64, copy=False)
        i2 = fringe_frames[freq * 4 + 1].astype(np.float64, copy=False)
        i3 = fringe_frames[freq * 4 + 2].astype(np.float64, copy=False)
        i4 = fringe_frames[freq * 4 + 3].astype(np.float64, copy=False)
        up_diff = i4 - i2
        down_diff = i1 - i3
        mod = 0.5 * np.sqrt(up_diff**2 + down_diff**2)
        mods.append(mod)
    return np.mean(mods, axis=0)


def fuse_multi_exposure_fringe_patterns(
    brackets_frames: List[List[np.ndarray]],
    sat_threshold: float = 250.0,
    min_modulation: float = 5.0,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """
    Fuses multiple exposure fringe pattern sequences into a single set of 12 HDR fringe patterns
    according to Jiang et al. (Opt. Lasers Eng. 2012) and Zhu et al. (Opt. Express 2023).

    Steps:
      1. Saturated pixel detection: If max(I_{m, k}(x, y)) >= sat_threshold, discard bracket m at (x, y).
      2. Modulation calculation: B_m(x, y) = 0.5 * sqrt((I4 - I2)^2 + (I1 - I3)^2).
      3. Best-exposure selection: m*(x, y) = argmax_{m in unsaturated} B_m(x, y).
      4. Fringe fusion: I_k^{fused}(x, y) = I_{m*(x, y), k}(x, y) for k = 1..12.

    Returns:
      fused_frames: List of 12 uint8 fused fringe images.
      fused_modulation: (H, W) float64 modulation map corresponding to m*.
      fused_mask: (H, W) uint8 binary mask (255 = valid candidate).
      best_idx: (H, W) int32 optimal exposure index map.
    """
    num_brackets = len(brackets_frames)
    if num_brackets == 0:
        raise ValueError("至少需要提供1组条纹图像序列")

    sample_frame = brackets_frames[0][0]
    h, w = sample_frame.shape[:2]

    if num_brackets == 1:
        fused_frames = [np.asarray(f, dtype=np.uint8).copy() for f in brackets_frames[0]]
        mod = compute_sequence_modulation(fused_frames)
        mask = (mod >= min_modulation).astype(np.uint8) * 255
        idx_map = np.zeros((h, w), dtype=np.int32)
        return fused_frames, mod, mask, idx_map

    if is_cuda_available() and cuda_fuse_multi_exposure_fringe_patterns is not None:
        try:
            return cuda_fuse_multi_exposure_fringe_patterns(
                brackets_frames=brackets_frames,
                sat_threshold=sat_threshold,
                min_modulation=min_modulation,
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("GPU多曝光条纹融合异常，回退至CPU计算: %s", e)

    # 1. Compute modulations and saturation masks for all M brackets (streaming in float32)
    mods_list = []
    sat_masks = []
    for m in range(num_brackets):
        frames_m = brackets_frames[m]
        max_int = np.max([np.asarray(img if img.ndim == 2 else img[..., 0], dtype=np.uint8) for img in frames_m], axis=0)
        sat_masks.append(max_int >= sat_threshold)
        mods_list.append(compute_sequence_modulation(frames_m).astype(np.float32, copy=False))

    mods_stack = np.stack(mods_list, axis=0)  # (M, H, W) float32
    sat_stack = np.stack(sat_masks, axis=0)   # (M, H, W) bool

    # 2. Unsaturated valid candidates
    valid_candidates = (~sat_stack) & (mods_stack >= min_modulation)

    # 3. Best bracket selection: m*(x, y) = argmax_{unsaturated} B_m(x, y)
    effective_mods = mods_stack.copy()
    effective_mods[~valid_candidates] = -1.0

    has_valid_unsaturated = np.any(valid_candidates, axis=0)
    best_idx = np.argmax(effective_mods, axis=0).astype(np.int32)

    # 4. Fallback for pixels where all exposures were saturated (extreme mirror highlight):
    # Select the lowest exposure (index 0)
    all_sat_pixels = ~has_valid_unsaturated
    if np.any(all_sat_pixels):
        best_idx[all_sat_pixels] = 0

    # 5. Assemble 12 fused fringe images: I_k^{fused}(x, y) = I_{m*(x, y), k}(x, y)
    num_frames = len(brackets_frames[0])
    fused_frames = [np.empty((h, w), dtype=np.uint8) for _ in range(num_frames)]
    for m in range(num_brackets):
        mask_m = (best_idx == m)
        if not np.any(mask_m):
            continue
        for k in range(num_frames):
            frame_k = brackets_frames[m][k]
            if frame_k.ndim == 3:
                frame_k = frame_k[..., 0]
            fused_frames[k][mask_m] = frame_k[mask_m]

    # Composite modulation
    fused_mod = np.take_along_axis(mods_stack, best_idx[None, ...], axis=0)[0].astype(np.float64)
    fused_mask = ((fused_mod >= min_modulation) | np.all(sat_stack, axis=0)).astype(np.uint8) * 255

    return fused_frames, fused_mod, fused_mask, best_idx


def fuse_multi_exposure_phases(
    phases: List[np.ndarray],
    modulations: List[np.ndarray],
    frames_list: Optional[List[List[np.ndarray]]] = None,
    sat_threshold: float = 250.0,
    min_modulation: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fuses multiple unwrapped phase maps acquired at different exposure times or projector intensities.
    Preserved for backward compatibility.
    """
    if len(phases) == 0:
        raise ValueError("至少需要提供1组相位图")
    if len(phases) != len(modulations):
        raise ValueError("相位图组数与调制图组数必须一致")

    num_exposures = len(phases)
    if num_exposures == 1:
        single_phase = phases[0]
        single_mod = modulations[0]
        mask = ((single_mod >= min_modulation) & np.isfinite(single_phase) & (single_phase > 1e-9)).astype(np.uint8) * 255
        idx_map = np.zeros(single_phase.shape, dtype=np.int32)
        return single_phase, single_mod, mask, idx_map

    h, w = phases[0].shape
    phases_stack = np.stack([p.astype(np.float64, copy=False) for p in phases], axis=0)  # (K, H, W)
    mods_stack = np.stack([m.astype(np.float64, copy=False) for m in modulations], axis=0)  # (K, H, W)

    finite_mask = np.isfinite(phases_stack) & (phases_stack > 1e-9) & (mods_stack >= min_modulation)

    sat_mask = np.zeros((num_exposures, h, w), dtype=bool)
    if frames_list is not None and len(frames_list) == num_exposures:
        for k in range(num_exposures):
            f_group = frames_list[k]
            if len(f_group) > 0:
                max_intensity = np.max([np.asarray(img, dtype=np.float32) for img in f_group], axis=0)
                sat_mask[k] = max_intensity >= sat_threshold

    valid_candidates = finite_mask & (~sat_mask)

    effective_mods = mods_stack.copy()
    effective_mods[~valid_candidates] = -1.0

    has_valid_unsaturated = np.any(valid_candidates, axis=0)
    best_idx = np.argmax(effective_mods, axis=0).astype(np.int32)

    all_sat_pixels = (~has_valid_unsaturated) & np.any(finite_mask, axis=0)
    if np.any(all_sat_pixels):
        best_idx[all_sat_pixels] = 0

    has_any_valid = has_valid_unsaturated | all_sat_pixels

    fused_phase = np.take_along_axis(phases_stack, best_idx[None, ...], axis=0)[0]
    fused_modulation = np.take_along_axis(mods_stack, best_idx[None, ...], axis=0)[0]

    fused_phase[~has_any_valid] = 0.0
    fused_mask = (has_any_valid & (fused_modulation >= min_modulation)).astype(np.uint8) * 255

    return fused_phase, fused_modulation, fused_mask, best_idx

