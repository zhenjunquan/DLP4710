"""
Scheimpflug DLP projector projection, distortion, and calibration routines.
"""

from typing import List, Optional, Tuple
import cv2
import numpy as np


def compute_tilt_matrix(tau_x: float, tau_y: float) -> np.ndarray:
    """
    Computes the OpenCV Scheimpflug tilted-image-plane 3x3 transformation matrix.
    """
    cx = np.cos(tau_x)
    sx = np.sin(tau_x)
    cy = np.cos(tau_y)
    sy = np.sin(tau_y)

    rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cx, sx],
        [0.0, -sx, cx],
    ], dtype=np.float64)

    ry = np.array([
        [cy, 0.0, -sy],
        [0.0, 1.0, 0.0],
        [sy, 0.0, cy],
    ], dtype=np.float64)

    r_xy = ry @ rx

    pz = np.array([
        [r_xy[2, 2], 0.0, -r_xy[0, 2]],
        [0.0, r_xy[2, 2], -r_xy[1, 2]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    return pz @ r_xy


def project_to_dlp(
    points_3d: np.ndarray,
    projector_matrix: np.ndarray,
    distortion_14: np.ndarray,
    tilt_matrix: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Projects 3D points in DLP coordinate system to distorted DLP pixel coordinates (u, v).
    points_3d: shape (3,) or (N, 3)
    projector_matrix: 3x3
    distortion_14: 14x1 or (14,)
    tilt_matrix: optional precomputed 3x3 tilt matrix from tauX and tauY
    Returns: shape (2,) or (N, 2)
    """
    pts = np.asarray(points_3d, dtype=np.float64)
    is_single = pts.ndim == 1
    if is_single:
        pts = pts.reshape(1, 3)

    num_pts = len(pts)
    out_uv = np.full((num_pts, 2), np.nan, dtype=np.float64)

    z = pts[:, 2]
    valid_z = z > 1e-12
    if not np.any(valid_z):
        return out_uv[0] if is_single else out_uv

    x = pts[valid_z, 0] / z[valid_z]
    y = pts[valid_z, 1] / z[valid_z]

    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2

    d = np.asarray(distortion_14, dtype=np.float64).flatten()
    k1, k2, p1, p2 = d[0], d[1], d[2], d[3]
    k3 = d[4] if len(d) > 4 else 0.0
    k4 = d[5] if len(d) > 5 else 0.0
    k5 = d[6] if len(d) > 6 else 0.0
    k6 = d[7] if len(d) > 7 else 0.0
    s1 = d[8] if len(d) > 8 else 0.0
    s2 = d[9] if len(d) > 9 else 0.0
    s3 = d[10] if len(d) > 10 else 0.0
    s4 = d[11] if len(d) > 11 else 0.0
    tau_x = d[12] if len(d) > 12 else 0.0
    tau_y = d[13] if len(d) > 13 else 0.0

    denom = 1.0 + k4 * r2 + k5 * r4 + k6 * r6
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    numer = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
    radial = numer / denom

    dist_x = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x) + s1 * r2 + s2 * r4
    dist_y = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y + s3 * r2 + s4 * r4

    if tilt_matrix is None:
        tilt_matrix = compute_tilt_matrix(tau_x, tau_y)

    dist_homo = np.vstack([dist_x, dist_y, np.ones_like(dist_x)])  # (3, M)
    tilted = tilt_matrix @ dist_homo  # (3, M)

    tz = np.where(np.abs(tilted[2]) < 1e-12, np.nan, tilted[2])
    norm_x = tilted[0] / tz
    norm_y = tilted[1] / tz

    u = projector_matrix[0, 0] * norm_x + projector_matrix[0, 2]
    v = projector_matrix[1, 1] * norm_y + projector_matrix[1, 2]

    out_uv[valid_z, 0] = u
    out_uv[valid_z, 1] = v

    return out_uv[0] if is_single else out_uv


def build_dlp_row_plane_normals(
    projector_matrix: np.ndarray,
    distortion_14: np.ndarray,
    projector_width: int = 1920,
    projector_height: int = 1080,
) -> np.ndarray:
    """
    Computes unit normal vectors for each integer DLP row plane.
    Returns: shape (projector_height, 3), float64.
    """
    distorted_pts = []
    for y in range(projector_height):
        distorted_pts.append([0.0, float(y)])
        distorted_pts.append([float(projector_width - 1), float(y)])

    distorted_pts = np.array(distorted_pts, dtype=np.float32).reshape(-1, 1, 2)

    # Use cv2.undistortPoints
    try:
        norm_pts = cv2.undistortPoints(
            distorted_pts,
            projector_matrix.astype(np.float64),
            distortion_14.astype(np.float64),
        ).reshape(-1, 2)
    except Exception as e:
        raise RuntimeError(f"cv2.undistortPoints failed for DLP: {e}")

    row_normals = np.zeros((projector_height, 3), dtype=np.float64)

    for y in range(projector_height):
        p_left = norm_pts[2 * y]
        p_right = norm_pts[2 * y + 1]

        ray_left = np.array([p_left[0], p_left[1], 1.0], dtype=np.float64)
        ray_right = np.array([p_right[0], p_right[1], 1.0], dtype=np.float64)

        normal = np.cross(ray_left, ray_right)
        norm_len = np.linalg.norm(normal)
        if norm_len > 1e-12:
            normal /= norm_len
        row_normals[y] = normal

    return row_normals

