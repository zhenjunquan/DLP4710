"""
GPU (CUDA / PyTorch) Accelerated 3D Point Cloud Reconstructor.
Telecentric Camera + Scheimpflug DLP Projector.
100% numerically equivalent to MonoSLReconstructor, executed in parallel on NVIDIA GPU.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np


def is_cuda_available() -> bool:
    """Checks whether PyTorch with CUDA support is available."""
    try:
        import torch

        return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
    except Exception:
        return False


def get_cuda_device_name() -> str:
    """Returns the name of the primary CUDA GPU, or empty string."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return ""


def reconstruct_points_cuda(
    camera_uv: np.ndarray,
    projector_v: np.ndarray,
    telecentric_proj: np.ndarray,
    telecentric_dist: np.ndarray,
    camera_size: Tuple[int, int],
    cam_pseudo_u: np.ndarray,
    cam_pseudo_v: np.ndarray,
    cam_line_dir: np.ndarray,
    proj_matrix: np.ndarray,
    proj_dist: np.ndarray,
    proj_tilt: np.ndarray,
    proj_size: Tuple[int, int],
    proj_row_normals: np.ndarray,
    min_z: float,
    max_z: float,
    device_str: str = "cuda:0",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Massively parallel GPU implementation of camera ray - DLP row plane intersection.
    camera_uv: (N, 2) float64
    projector_v: (N,) float64
    Returns:
      points: (N, 3) float64
      valid: (N,) bool
    """
    import torch

    device = torch.device(device_str)

    with torch.no_grad():
        uv_t = torch.from_numpy(camera_uv).to(device=device, dtype=torch.float64)
        v_t = torch.from_numpy(projector_v).to(device=device, dtype=torch.float64)
        count = uv_t.shape[0]

        valid = (torch.isfinite(v_t)) & (v_t >= 0.0) & (v_t < float(proj_size[1]) - 1e-6)
        if not torch.any(valid):
            return np.zeros((count, 3), dtype=np.float64), np.zeros(count, dtype=bool)

        # 1. Telecentric camera undistortion (6 fixed-point iterations)
        observed = uv_t.clone()
        ideal = observed.clone()
        width, height = camera_size
        cx = (width - 1) * 0.5
        cy = (height - 1) * 0.5
        norm_factor = max(1.0, 0.5 * max(width, height))

        k1 = float(telecentric_dist[0])
        k2 = float(telecentric_dist[1])
        p1 = float(telecentric_dist[2])
        p2 = float(telecentric_dist[3])
        k3 = float(telecentric_dist[4]) if len(telecentric_dist) > 4 else 0.0

        for _ in range(6):
            xn = (ideal[:, 0] - cx) / norm_factor
            yn = (ideal[:, 1] - cy) / norm_factor
            r2 = xn * xn + yn * yn
            radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
            pred_x = cx + norm_factor * (
                xn * radial + 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn * xn)
            )
            pred_y = cy + norm_factor * (
                yn * radial + p1 * (r2 + 2.0 * yn * yn) + 2.0 * p2 * xn * yn
            )
            ideal[:, 0] += observed[:, 0] - pred_x
            ideal[:, 1] += observed[:, 1] - pred_y

        valid &= torch.all(torch.isfinite(ideal), dim=1)

        t_proj_03 = float(telecentric_proj[0, 3])
        t_proj_13 = float(telecentric_proj[1, 3])
        bu = ideal[:, 0] - t_proj_03
        bv = ideal[:, 1] - t_proj_13

        pseudo_u = torch.from_numpy(cam_pseudo_u).to(device=device, dtype=torch.float64)
        pseudo_v = torch.from_numpy(cam_pseudo_v).to(device=device, dtype=torch.float64)
        line_dir = torch.from_numpy(cam_line_dir).to(device=device, dtype=torch.float64)

        if abs(float(line_dir[2].item())) < 1e-12:
            return np.zeros((count, 3), dtype=np.float64), np.zeros(count, dtype=bool)

        line_pt = bu.unsqueeze(1) * pseudo_u.unsqueeze(0) + bv.unsqueeze(1) * pseudo_v.unsqueeze(0)

        # 2. Projector row normal interpolation
        proj_h = proj_size[1]
        top = torch.clamp(torch.floor(v_t).to(torch.int64), 0, proj_h - 1)
        bottom = torch.clamp(top + 1, 0, proj_h - 1)
        frac = v_t - top.to(torch.float64)

        row_normals = torch.from_numpy(proj_row_normals).to(device=device, dtype=torch.float64)
        normals = row_normals[top] * (1.0 - frac.unsqueeze(1)) + row_normals[bottom] * frac.unsqueeze(1)
        normal_len = torch.linalg.norm(normals, dim=1)
        valid &= normal_len > 1e-12
        normals = normals / torch.clamp(normal_len.unsqueeze(1), min=1e-12)
        denom = torch.matmul(normals, line_dir)
        valid &= torch.abs(denom) > 1e-12

        # 3. Depth limits along ray
        t_a = (min_z - line_pt[:, 2]) / line_dir[2]
        t_b = (max_z - line_pt[:, 2]) / line_dir[2]
        left = torch.minimum(t_a, t_b)
        right = torch.maximum(t_a, t_b)
        valid &= torch.isfinite(left) & torch.isfinite(right) & (right > left)

        # 4. DLP projection residual calculation
        d = torch.from_numpy(proj_dist).to(device=device, dtype=torch.float64).flatten()
        dk1, dk2, dp1, dp2 = d[0], d[1], d[2], d[3]
        dk3 = d[4] if len(d) > 4 else torch.tensor(0.0, device=device, dtype=torch.float64)
        dk4 = d[5] if len(d) > 5 else torch.tensor(0.0, device=device, dtype=torch.float64)
        dk5 = d[6] if len(d) > 6 else torch.tensor(0.0, device=device, dtype=torch.float64)
        dk6 = d[7] if len(d) > 7 else torch.tensor(0.0, device=device, dtype=torch.float64)
        ds1 = d[8] if len(d) > 8 else torch.tensor(0.0, device=device, dtype=torch.float64)
        ds2 = d[9] if len(d) > 9 else torch.tensor(0.0, device=device, dtype=torch.float64)
        ds3 = d[10] if len(d) > 10 else torch.tensor(0.0, device=device, dtype=torch.float64)
        ds4 = d[11] if len(d) > 11 else torch.tensor(0.0, device=device, dtype=torch.float64)

        K = torch.from_numpy(proj_matrix).to(device=device, dtype=torch.float64)
        R_tilt = torch.from_numpy(proj_tilt).to(device=device, dtype=torch.float64)

        def residual(t_vals: torch.Tensor) -> torch.Tensor:
            pts = line_pt + t_vals.unsqueeze(1) * line_dir.unsqueeze(0)
            z = pts[:, 2]
            z_safe = torch.where(torch.abs(z) < 1e-12, torch.full_like(z, 1e-12), z)
            x = pts[:, 0] / z_safe
            y = pts[:, 1] / z_safe
            r2 = x * x + y * y
            r4 = r2 * r2
            r6 = r4 * r2

            den = 1.0 + dk4 * r2 + dk5 * r4 + dk6 * r6
            den = torch.where(torch.abs(den) < 1e-12, torch.full_like(den, 1e-12), den)
            num = 1.0 + dk1 * r2 + dk2 * r4 + dk3 * r6
            rad = num / den

            dx = x * rad + 2.0 * dp1 * x * y + dp2 * (r2 + 2.0 * x * x) + ds1 * r2 + ds2 * r4
            dy = y * rad + dp1 * (r2 + 2.0 * y * y) + 2.0 * dp2 * x * y + ds3 * r2 + ds4 * r4

            dist_h = torch.stack([dx, dy, torch.ones_like(dx)], dim=1)
            dist_tilted = torch.matmul(dist_h, R_tilt.T)
            inv_z = 1.0 / torch.where(
                torch.abs(dist_tilted[:, 2]) < 1e-12,
                torch.full_like(dist_tilted[:, 2], 1e-12),
                dist_tilted[:, 2],
            )
            px = dist_tilted[:, 0] * inv_z
            py = dist_tilted[:, 1] * inv_z

            v_proj = K[1, 0] * px + K[1, 1] * py + K[1, 2]
            return v_proj - v_t

        res_left = residual(left)
        res_right = residual(right)
        valid &= torch.isfinite(res_left) & torch.isfinite(res_right) & (res_left * res_right <= 0.0)

        # 16 bisection iterations in parallel across CUDA cores
        for _ in range(16):
            middle = 0.5 * (left + right)
            res_mid = residual(middle)
            use_left = (res_left * res_mid) <= 0.0
            right = torch.where(use_left, middle, right)
            res_right = torch.where(use_left, res_mid, res_right)
            left = torch.where(use_left, left, middle)
            res_left = torch.where(use_left, res_left, res_mid)

        t_final = 0.5 * (left + right)
        final_points = line_pt + t_final.unsqueeze(1) * line_dir.unsqueeze(0)

        # Final projection verification
        z_f = final_points[:, 2]
        z_f_safe = torch.where(torch.abs(z_f) < 1e-12, torch.full_like(z_f, 1e-12), z_f)
        xf = final_points[:, 0] / z_f_safe
        yf = final_points[:, 1] / z_f_safe
        r2_f = xf * xf + yf * yf
        r4_f = r2_f * r2_f
        r6_f = r4_f * r2_f
        den_f = 1.0 + dk4 * r2_f + dk5 * r4_f + dk6 * r6_f
        den_f = torch.where(torch.abs(den_f) < 1e-12, torch.full_like(den_f, 1e-12), den_f)
        rad_f = (1.0 + dk1 * r2_f + dk2 * r4_f + dk3 * r6_f) / den_f
        dxf = xf * rad_f + 2.0 * dp1 * xf * yf + dp2 * (r2_f + 2.0 * xf * xf) + ds1 * r2_f + ds2 * r4_f
        dyf = yf * rad_f + dp1 * (r2_f + 2.0 * yf * yf) + 2.0 * dp2 * xf * yf + ds3 * r2_f + ds4 * r4_f
        dist_hf = torch.stack([dxf, dyf, torch.ones_like(dxf)], dim=1)
        dist_tilted_f = torch.matmul(dist_hf, R_tilt.T)
        inv_zf = 1.0 / torch.where(
            torch.abs(dist_tilted_f[:, 2]) < 1e-12,
            torch.full_like(dist_tilted_f[:, 2], 1e-12),
            dist_tilted_f[:, 2],
        )
        px_f = dist_tilted_f[:, 0] * inv_zf
        py_f = dist_tilted_f[:, 1] * inv_zf
        u_proj = K[0, 0] * px_f + K[0, 1] * py_f + K[0, 2]
        v_proj = K[1, 0] * px_f + K[1, 1] * py_f + K[1, 2]

        valid &= (
            torch.all(torch.isfinite(final_points), dim=1)
            & torch.isfinite(u_proj)
            & torch.isfinite(v_proj)
            & (final_points[:, 2] >= min_z)
            & (final_points[:, 2] <= max_z)
            & (torch.abs(v_proj - v_t) <= 0.10)
            & (u_proj >= 0.0)
            & (u_proj < float(proj_size[0]))
        )

        valid_np = valid.cpu().numpy()
        points_np = np.zeros((count, 3), dtype=np.float64)
        if np.any(valid_np):
            points_np[valid_np] = final_points[valid].cpu().numpy()

        return points_np, valid_np

