"""
Telecentric camera projection and distortion models.
Exact numerical match with C++ CalibrationEngine and MonoSLReconstructor.
"""

from typing import List, Optional, Tuple
import cv2
import numpy as np


def project_telecentric_point(
    projector_point: np.ndarray,
    affine_2x4: np.ndarray,
    distortion_5: Optional[np.ndarray],
    image_size: Tuple[int, int],
) -> np.ndarray:
    """
    Forward projection from 3D DLP coordinates (X, Y, Z) to distorted 2D camera coordinates (u, v).
    projector_point: shape (3,) or (N, 3)
    affine_2x4: shape (2, 4)
    distortion_5: shape (5,) or (5, 1) [k1, k2, p1, p2, k3] in normalized image coordinates.
    image_size: (width, height)
    Returns: shape (2,) or (N, 2)
    """
    pts = np.asarray(projector_point, dtype=np.float64)
    is_single = pts.ndim == 1
    if is_single:
        pts = pts.reshape(1, 3)

    num_pts = len(pts)
    homo_pts = np.hstack([pts, np.ones((num_pts, 1), dtype=np.float64)])  # (N, 4)
    ideal_uv = (affine_2x4 @ homo_pts.T).T  # (N, 2)

    if distortion_5 is None or len(distortion_5) == 0:
        return ideal_uv[0] if is_single else ideal_uv

    k_params = np.asarray(distortion_5, dtype=np.float64).flatten()
    k1 = k_params[0] if len(k_params) > 0 else 0.0
    k2 = k_params[1] if len(k_params) > 1 else 0.0
    p1 = k_params[2] if len(k_params) > 2 else 0.0
    p2 = k_params[3] if len(k_params) > 3 else 0.0
    k3 = k_params[4] if len(k_params) > 4 else 0.0

    width, height = image_size
    center_x = (width - 1) * 0.5
    center_y = (height - 1) * 0.5
    norm_factor = max(1.0, 0.5 * max(width, height))

    x = (ideal_uv[:, 0] - center_x) / norm_factor
    y = (ideal_uv[:, 1] - center_y) / norm_factor
    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2

    radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
    distorted_x = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    distorted_y = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y

    out_uv = np.empty_like(ideal_uv)
    out_uv[:, 0] = center_x + norm_factor * distorted_x
    out_uv[:, 1] = center_y + norm_factor * distorted_y

    return out_uv[0] if is_single else out_uv


def undistort_telecentric_pixel(
    observed_uv: np.ndarray,
    affine_2x4: np.ndarray,
    distortion_5: np.ndarray,
    image_size: Tuple[int, int],
    max_iter: int = 6,
) -> Tuple[bool, np.ndarray]:
    """
    Inverse distortion: maps observed camera pixel (u_obs, v_obs) back to ideal linear pixel (u_ideal, v_ideal).
    observed_uv: (2,) float64
    Returns: (success: bool, ideal_uv: (2,) float64)
    """
    obs = np.asarray(observed_uv, dtype=np.float64)
    ideal = obs.copy()

    for _ in range(max_iter):
        # We distort ideal pixel by passing dummy 3D or direct pixel distortion
        w, h = image_size
        cx = (w - 1) * 0.5
        cy = (h - 1) * 0.5
        norm_factor = max(1.0, 0.5 * max(w, h))

        x = (ideal[0] - cx) / norm_factor
        y = (ideal[1] - cy) / norm_factor
        r2 = x * x + y * y
        r4 = r2 * r2
        r6 = r4 * r2

        k_params = np.asarray(distortion_5, dtype=np.float64).flatten()
        k1 = k_params[0] if len(k_params) > 0 else 0.0
        k2 = k_params[1] if len(k_params) > 1 else 0.0
        p1 = k_params[2] if len(k_params) > 2 else 0.0
        p2 = k_params[3] if len(k_params) > 3 else 0.0
        k3 = k_params[4] if len(k_params) > 4 else 0.0

        radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
        pred_x = cx + norm_factor * (x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x))
        pred_y = cy + norm_factor * (y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y)

        err_x = obs[0] - pred_x
        err_y = obs[1] - pred_y
        ideal[0] += err_x
        ideal[1] += err_y

        if err_x * err_x + err_y * err_y < 1e-14:
            break

    # Verification
    x = (ideal[0] - cx) / norm_factor
    y = (ideal[1] - cy) / norm_factor
    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2
    radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
    verif_x = cx + norm_factor * (x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x))
    verif_y = cy + norm_factor * (y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y)
    err2 = (verif_x - obs[0]) ** 2 + (verif_y - obs[1]) ** 2

    success = bool(np.isfinite(ideal[0]) and np.isfinite(ideal[1]) and err2 < 1e-6)
    return success, ideal


def fit_linear_affine(
    points_3d_list: List[np.ndarray],
    camera_points_list: List[np.ndarray],
) -> Tuple[bool, np.ndarray]:
    """
    Fits initial 2x4 affine camera matrix from DLP 3D points and observed camera points via SVD / least squares.
    """
    all_3d = np.vstack(points_3d_list)  # (M, 3)
    all_cam = np.vstack(camera_points_list)  # (M, 2)

    total_pts = len(all_3d)
    if total_pts < 8:
        return False, np.zeros((2, 4), dtype=np.float64)

    # Design matrix: [X, Y, Z, 1]
    design = np.hstack([all_3d, np.ones((total_pts, 1), dtype=np.float64)])

    try:
        # Solve design * row_u^T = u
        row_u, _, _, _ = np.linalg.lstsq(design, all_cam[:, 0], rcond=None)
        # Solve design * row_v^T = v
        row_v, _, _, _ = np.linalg.lstsq(design, all_cam[:, 1], rcond=None)
        affine_2x4 = np.vstack([row_u, row_v])
        return bool(np.all(np.isfinite(affine_2x4))), affine_2x4
    except Exception:
        return False, np.zeros((2, 4), dtype=np.float64)


def decompose_affine_camera(
    affine_2x4: np.ndarray,
    image_size: Tuple[int, int],
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Decomposes 2x4 telecentric affine matrix into:
      scale_x (px/mm), scale_y (px/mm),
      projector_to_camera_rotation (3, 3),
      projector_to_camera_translation (3, 1) with Tz fixed to 0.0 mm gauge.
    """
    row_u = affine_2x4[0, 0:3]
    row_v = affine_2x4[1, 0:3]

    scale_x = float(np.linalg.norm(row_u))
    scale_y = float(np.linalg.norm(row_v))

    if scale_x < 1e-12 or scale_y < 1e-12:
        return 0.0, 0.0, np.eye(3), np.zeros((3, 1))

    r1 = row_u / scale_x
    r2 = row_v / scale_y
    r1 = r1 / max(1e-12, np.linalg.norm(r1))
    r2 = r2 - r1 * np.dot(r1, r2)
    r2 = r2 / max(1e-12, np.linalg.norm(r2))
    r3 = np.cross(r1, r2)
    r3 = r3 / max(1e-12, np.linalg.norm(r3))

    rot_matrix = np.vstack([r1, r2, r3])

    cx = (image_size[0] - 1) * 0.5
    cy = (image_size[1] - 1) * 0.5
    trans = np.array([
        [(affine_2x4[0, 3] - cx) / scale_x],
        [(affine_2x4[1, 3] - cy) / scale_y],
        [0.0],
    ], dtype=np.float64)

    return scale_x, scale_y, rot_matrix, trans

