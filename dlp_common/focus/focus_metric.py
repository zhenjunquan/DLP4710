"""
Focus measure metrics and peaking algorithms for optical sharp focus evaluation.
"""

from enum import Enum
from typing import Optional, Tuple
import cv2
import numpy as np


class FocusMetricType(str, Enum):
    LAPLACIAN_VAR = "Laplacian方差"
    TENENGRAD = "Tenengrad梯度能量"
    BRENNER = "Brenner差分梯度"
    MODIFIED_LAPLACIAN = "SML修正常数拉普拉斯"


def extract_roi(
    image: np.ndarray,
    roi: Optional[Tuple[int, int, int, int]] = None,
) -> np.ndarray:
    """
    Extracts Region of Interest from an image.
    roi: (x, y, width, height)
    """
    if roi is None:
        return image
    x, y, w, h = roi
    img_h, img_w = image.shape[:2]
    x0 = max(0, min(x, img_w - 1))
    y0 = max(0, min(y, img_h - 1))
    x1 = max(x0 + 1, min(x + w, img_w))
    y1 = max(y0 + 1, min(y + h, img_h))
    return image[y0:y1, x0:x1]


def compute_focus_score(
    image: np.ndarray,
    metric: FocusMetricType = FocusMetricType.LAPLACIAN_VAR,
    roi: Optional[Tuple[int, int, int, int]] = None,
    downsample_factor: int = 1,
) -> float:
    """
    Computes sharpness / focus metric score on the specified ROI or full image.
    Returns a positive float where higher values indicate sharper focus.
    """
    if image is None or image.size == 0:
        return 0.0

    # Ensure grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Crop to ROI
    patch = extract_roi(gray, roi)
    if patch.size == 0:
        return 0.0

    # Optional downsample for ultra-fast full-frame calculation
    if downsample_factor > 1 and patch.shape[0] > 600 and patch.shape[1] > 600:
        patch = cv2.resize(
            patch,
            (patch.shape[1] // downsample_factor, patch.shape[0] // downsample_factor),
            interpolation=cv2.INTER_AREA,
        )

    if metric == FocusMetricType.LAPLACIAN_VAR:
        # Variance of Laplacian: Var(∇² I)
        lap = cv2.Laplacian(patch, cv2.CV_64F, ksize=3)
        return float(lap.var())

    elif metric == FocusMetricType.TENENGRAD:
        # Tenengrad: Mean energy of Sobel gradients
        gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
        mag_sq = gx * gx + gy * gy
        return float(np.mean(mag_sq))

    elif metric == FocusMetricType.BRENNER:
        # Brenner: sum of squared differences of second neighbor pixels
        p32 = patch.astype(np.float32)
        diff_x = p32[:, 2:] - p32[:, :-2]
        return float(np.mean(diff_x * diff_x))

    elif metric == FocusMetricType.MODIFIED_LAPLACIAN:
        # Sum of Modified Laplacian (SML)
        p32 = patch.astype(np.float32)
        lx = np.abs(2 * p32[1:-1, 1:-1] - p32[1:-1, :-2] - p32[1:-1, 2:])
        ly = np.abs(2 * p32[1:-1, 1:-1] - p32[:-2, 1:-1] - p32[2:, 1:-1])
        return float(np.mean(lx + ly))

    else:
        lap = cv2.Laplacian(patch, cv2.CV_64F, ksize=3)
        return float(lap.var())


def compute_focus_peaking_mask(
    image: np.ndarray,
    highpass_threshold: float = 35.0,
    roi: Optional[Tuple[int, int, int, int]] = None,
) -> np.ndarray:
    """
    Computes a binary mask where edge sharpness exceeds highpass_threshold.
    Used for live focus peaking overlay.
    """
    if image is None or image.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    abs_lap = np.abs(lap)

    mask = np.zeros(gray.shape, dtype=np.uint8)
    if roi is not None:
        x, y, w, h = roi
        img_h, img_w = gray.shape
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(img_w, x + w), min(img_h, y + h)
        sub_lap = abs_lap[y0:y1, x0:x1]
        mask[y0:y1, x0:x1] = (sub_lap > highpass_threshold).astype(np.uint8) * 255
    else:
        mask = (abs_lap > highpass_threshold).astype(np.uint8) * 255

    return mask

