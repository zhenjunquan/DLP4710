"""
Utility functions for fringe processing and image sampling.
"""

import numpy as np


def bilinear_sample(image: np.ndarray, x: float, y: float) -> float:
    """
    Sub-pixel bilinear interpolation on a 2D float image.
    Returns np.nan if out of bounds.
    """
    h, w = image.shape[:2]
    if not (0.0 <= x <= w - 1.0 and 0.0 <= y <= h - 1.0):
        return float(np.nan)

    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)

    dx = x - x0
    dy = y - y0

    v00 = image[y0, x0]
    v01 = image[y0, x1]
    v10 = image[y1, x0]
    v11 = image[y1, x1]

    return float(
        (1.0 - dx) * (1.0 - dy) * v00
        + dx * (1.0 - dy) * v01
        + (1.0 - dx) * dy * v10
        + dx * dy * v11
    )


def bilinear_sample_points(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Vectorized bilinear interpolation for multiple (x, y) points.
    points: array of shape (N, 2), where points[:, 0] is x, points[:, 1] is y.
    Returns: array of shape (N,)
    """
    h, w = image.shape[:2]
    pts = np.asarray(points, dtype=np.float64)
    x = pts[:, 0]
    y = pts[:, 1]

    valid = (x >= 0.0) & (x <= w - 1.0) & (y >= 0.0) & (y <= h - 1.0)
    result = np.full(len(pts), np.nan, dtype=np.float64)

    if not np.any(valid):
        return result

    xv = x[valid]
    yv = y[valid]

    x0 = np.floor(xv).astype(np.int32)
    y0 = np.floor(yv).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)

    dx = xv - x0
    dy = yv - y0

    v00 = image[y0, x0]
    v01 = image[y0, x1]
    v10 = image[y1, x0]
    v11 = image[y1, x1]

    sampled = (
        (1.0 - dx) * (1.0 - dy) * v00
        + dx * (1.0 - dy) * v01
        + (1.0 - dx) * dy * v10
        + dx * dy * v11
    )
    result[valid] = sampled
    return result

