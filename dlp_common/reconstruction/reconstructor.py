"""
Mono Structured Light 3D Reconstructor.
Telecentric Camera + Scheimpflug DLP Projector.
Exact algorithmic reproduction of C++ MonoSLReconstructor.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
import cv2
import numpy as np

from ..calibration.calib_io import CalibrationIO
from ..calibration.models import CalibrationResult
from ..calibration.scheimpflug_dlp import compute_tilt_matrix, project_to_dlp
from ..calibration.telecentric_camera import (
    project_telecentric_point,
    undistort_telecentric_pixel,
)
from .cuda_reconstructor import (
    get_cuda_device_name,
    is_cuda_available,
    reconstruct_points_cuda,
)


class MonoSLReconstructor:
    """
    Reconstructs metric 3D point clouds from unwrapped horizontal fringe phase maps.
    Horizontal stripes vary along DLP vertical axis Y, so absolute phase maps directly
    to the DLP row coordinate v:
        projector_v = (phase / reference_cycle_count) * projector_height
    """

    def __init__(self, enable_gpu: bool = True):
        self.enable_gpu: bool = enable_gpu
        self.projector_matrix: Optional[np.ndarray] = None
        self.projector_dist_14: Optional[np.ndarray] = None
        self.telecentric_projection: Optional[np.ndarray] = None
        self.telecentric_dist_5: Optional[np.ndarray] = None

        self.camera_image_size: Tuple[int, int] = (0, 0)
        self.projector_image_size: Tuple[int, int] = (0, 0)
        self.fringe_periods: List[float] = []
        self.reference_cycle_count: float = 0.0

        # Affine pseudoinverse columns & line direction
        self.camera_pseudo_col_u: Optional[np.ndarray] = None
        self.camera_pseudo_col_v: Optional[np.ndarray] = None
        self.camera_line_direction: Optional[np.ndarray] = None

        # Scheimpflug tilt matrix
        self.projector_tilt_matrix: Optional[np.ndarray] = None
        self.projector_row_plane_normals: Optional[np.ndarray] = None

        self.min_projector_z: float = 1.0
        self.max_projector_z: float = 1000.0
        self.parameters_loaded: bool = False
        self.last_error: str = ""

    def load_parameters(self, yml_path: str) -> bool:
        """
        Loads the calibration YAML file and prepares internal models.
        """
        self.clear()
        ok, res, meta, msg = CalibrationIO.load_yml(yml_path)
        if not ok or res is None:
            self.last_error = f"加载标定文件失败: {msg}"
            return False

        self.projector_matrix = res.projector_matrix.astype(np.float64)
        self.projector_dist_14 = res.projector_dist_14.astype(np.float64).flatten()
        self.telecentric_projection = res.telecentric_projection_2x4.astype(np.float64)
        self.telecentric_dist_5 = res.telecentric_dist_5.astype(np.float64).flatten()

        self.camera_image_size = (meta["camera_image_width"], meta["camera_image_height"])
        self.projector_image_size = (meta["projector_width"], meta["projector_height"])
        self.fringe_periods = meta["fringe_periods"]
        self.reference_cycle_count = float(self.fringe_periods[0])

        if not self._estimate_depth_range(res, meta):
            return False
        if not self._prepare_camera_line_model():
            return False
        if not self._prepare_projector_model():
            return False

        self.parameters_loaded = True
        return True

    def has_calibration_parameters(self) -> bool:
        """Checks whether valid calibration parameters have been loaded."""
        return (
            self.parameters_loaded
            and self.projector_matrix is not None
            and self.telecentric_projection is not None
            and self.projector_row_plane_normals is not None
        )

    def clear(self):
        self.projector_matrix = None
        self.projector_dist_14 = None
        self.telecentric_projection = None
        self.telecentric_dist_5 = None
        self.camera_image_size = (0, 0)
        self.projector_image_size = (0, 0)
        self.fringe_periods = []
        self.reference_cycle_count = 0.0
        self.projector_row_plane_normals = None
        self.min_projector_z = 1.0
        self.max_projector_z = 1000.0
        self.parameters_loaded = False
        self.last_error = ""

    def _estimate_depth_range(self, res: CalibrationResult, meta: Dict[str, Any]) -> bool:
        board_w = meta.get("board_width", 0)
        board_h = meta.get("board_height", 0)
        sq_size = meta.get("square_size_mm", 0.0)

        if board_w <= 1 or board_h <= 1 or sq_size <= 0.0 or len(res.board_rvecs) == 0:
            self.last_error = "标定文件中缺少有效棋盘格姿态，无法确定重建深度范围"
            return False

        max_bx = (board_w - 1) * sq_size
        max_by = (board_h - 1) * sq_size
        corners = np.array([
            [0.0, 0.0, 0.0],
            [max_bx, 0.0, 0.0],
            [0.0, max_by, 0.0],
            [max_bx, max_by, 0.0],
        ], dtype=np.float64)

        all_z = []
        for i in range(len(res.board_rvecs)):
            r_mat, _ = cv2.Rodrigues(res.board_rvecs[i])
            t_vec = res.board_tvecs[i].flatten()
            pts_dlp = (r_mat @ corners.T).T + t_vec
            all_z.extend(pts_dlp[:, 2])

        min_z = float(np.min(all_z))
        max_z = float(np.max(all_z))

        calib_range = max(0.0, max_z - min_z)
        margin = max(5.0, calib_range * 0.35)
        self.min_projector_z = max(1e-3, min_z - margin)
        self.max_projector_z = max_z + margin

        if self.max_projector_z <= self.min_projector_z:
            self.last_error = "标定深度范围异常"
            return False
        return True

    def _prepare_camera_line_model(self) -> bool:
        row_u = self.telecentric_projection[0, 0:3]
        row_v = self.telecentric_projection[1, 0:3]

        g00 = float(np.dot(row_u, row_u))
        g01 = float(np.dot(row_u, row_v))
        g11 = float(np.dot(row_v, row_v))
        det = g00 * g11 - g01 * g01

        if abs(det) < 1e-12:
            self.last_error = "远心相机2x4投影矩阵退化，无法计算射线基底"
            return False

        self.camera_pseudo_col_u = (row_u * g11 - row_v * g01) / det
        self.camera_pseudo_col_v = (row_v * g00 - row_u * g01) / det

        line_dir = np.cross(row_u, row_v)
        dir_len = float(np.linalg.norm(line_dir))
        if dir_len < 1e-12:
            self.last_error = "远心相机光轴方向退化"
            return False
        self.camera_line_direction = line_dir / dir_len
        return True

    def _prepare_projector_model(self) -> bool:
        tau_x = float(self.projector_dist_14[12]) if len(self.projector_dist_14) > 12 else 0.0
        tau_y = float(self.projector_dist_14[13]) if len(self.projector_dist_14) > 13 else 0.0
        self.projector_tilt_matrix = compute_tilt_matrix(tau_x, tau_y)

        # Precompute DLP row plane unit normals
        proj_w, proj_h = self.projector_image_size
        dist_pts = []
        for y in range(proj_h):
            dist_pts.append([0.0, float(y)])
            dist_pts.append([float(proj_w - 1), float(y)])

        dist_pts_arr = np.array(dist_pts, dtype=np.float32).reshape(-1, 1, 2)
        try:
            norm_pts = cv2.undistortPoints(
                dist_pts_arr,
                self.projector_matrix,
                self.projector_dist_14,
            ).reshape(-1, 2)
        except Exception as e:
            self.last_error = f"DLP行光平面计算失败: {e}"
            return False

        self.projector_row_plane_normals = np.zeros((proj_h, 3), dtype=np.float64)
        for y in range(proj_h):
            p_l = norm_pts[2 * y]
            p_r = norm_pts[2 * y + 1]
            ray_l = np.array([p_l[0], p_l[1], 1.0], dtype=np.float64)
            ray_r = np.array([p_r[0], p_r[1], 1.0], dtype=np.float64)
            normal = np.cross(ray_l, ray_r)
            n_len = np.linalg.norm(normal)
            if n_len > 1e-12:
                normal /= n_len
            self.projector_row_plane_normals[y] = normal

        return True

    def reconstruct_one_point(
        self,
        camera_pixel: Tuple[float, float],
        projector_v: float,
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Reconstructs a single 3D metric point from camera pixel (u_c, v_c) and DLP row v.
        Returns: (success: bool, point_3d: (X, Y, Z) or None)
        """
        if (
            not np.isfinite(projector_v)
            or projector_v < 0.0
            or projector_v >= float(self.projector_image_size[1]) - 1e-6
        ):
            return False, None

        # Build telecentric camera ray: P(t) = line_pt + t * line_dir
        u_c, v_c = camera_pixel
        ok, ideal_uv = undistort_telecentric_pixel(
            np.array([u_c, v_c]),
            self.telecentric_projection,
            self.telecentric_dist_5,
            self.camera_image_size,
        )
        if not ok:
            return False, None

        bu = ideal_uv[0] - self.telecentric_projection[0, 3]
        bv = ideal_uv[1] - self.telecentric_projection[1, 3]
        line_pt = self.camera_pseudo_col_u * bu + self.camera_pseudo_col_v * bv
        line_dir = self.camera_line_direction

        if abs(line_dir[2]) < 1e-12:
            return False, None

        # Row plane interpolation
        proj_h = self.projector_image_size[1]
        top_row = max(0, min(proj_h - 1, int(np.floor(projector_v))))
        bot_row = min(proj_h - 1, top_row + 1)
        frac = projector_v - float(top_row)

        plane_normal = (
            self.projector_row_plane_normals[top_row] * (1.0 - frac)
            + self.projector_row_plane_normals[bot_row] * frac
        )
        p_len = np.linalg.norm(plane_normal)
        if p_len < 1e-12:
            return False, None
        plane_normal /= p_len

        denom = float(np.dot(plane_normal, line_dir))
        if abs(denom) < 1e-12:
            return False, None

        # Depth limits along the ray
        t_min = (self.min_projector_z - line_pt[2]) / line_dir[2]
        t_max = (self.max_projector_z - line_pt[2]) / line_dir[2]
        if t_min > t_max:
            t_min, t_max = t_max, t_min

        if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
            return False, None

        def evaluate_residual(t_val: float) -> Tuple[bool, float]:
            pt3 = line_pt + line_dir * t_val
            proj_uv = project_to_dlp(
                pt3,
                self.projector_matrix,
                self.projector_dist_14,
                self.projector_tilt_matrix,
            )
            if not np.isfinite(proj_uv[1]):
                return False, 0.0
            return True, float(proj_uv[1] - projector_v)

        ok_left, res_left = evaluate_residual(t_min)
        ok_right, res_right = evaluate_residual(t_max)
        if not ok_left or not ok_right:
            return False, None

        # Root must be bracketed
        if res_left * res_right > 0.0:
            return False, None

        # Plane intersection initial guess
        t = -float(np.dot(plane_normal, line_pt)) / denom
        if not np.isfinite(t) or t <= t_min or t >= t_max:
            t = 0.5 * (t_min + t_max)

        left_t, right_t = t_min, t_max
        converged = False
        res = 0.0

        # Safeguarded Newton iteration + Bisection
        for _ in range(8):
            ok_res, res = evaluate_residual(t)
            if not ok_res:
                return False, None

            if abs(res) < 0.02:
                converged = True
                break

            if res_left * res <= 0.0:
                right_t = t
                res_right = res
            else:
                left_t = t
                res_left = res

            h = 0.02
            fwd_t = min(right_t, t + h)
            bwd_t = max(left_t, t - h)

            ok_fwd, fwd_res = evaluate_residual(fwd_t) if fwd_t > t + 1e-12 else (False, 0.0)
            ok_bwd, bwd_res = evaluate_residual(bwd_t) if bwd_t < t - 1e-12 else (False, 0.0)

            deriv = 0.0
            if ok_fwd and ok_bwd:
                deriv = (fwd_res - bwd_res) / (fwd_t - bwd_t)
            elif ok_fwd:
                deriv = (fwd_res - res) / (fwd_t - t)
            elif ok_bwd:
                deriv = (res - bwd_res) / (t - bwd_t)

            next_t = np.nan
            if np.isfinite(deriv) and abs(deriv) > 1e-8:
                next_t = t - res / deriv

            if not np.isfinite(next_t) or next_t <= left_t or next_t >= right_t:
                next_t = 0.5 * (left_t + right_t)

            if abs(next_t - t) < 1e-10:
                t = next_t
                break
            t = next_t

        ok_final, final_res = evaluate_residual(t)
        if not ok_final or (not converged and abs(final_res) > 0.10):
            return False, None

        pt3d = line_pt + line_dir * t
        if (
            not np.all(np.isfinite(pt3d))
            or pt3d[2] < self.min_projector_z
            or pt3d[2] > self.max_projector_z
        ):
            return False, None

        proj_uv = project_to_dlp(
            pt3d,
            self.projector_matrix,
            self.projector_dist_14,
            self.projector_tilt_matrix,
        )
        if (
            abs(proj_uv[1] - projector_v) > 0.10
            or proj_uv[0] < 0.0
            or proj_uv[0] >= float(self.projector_image_size[0])
        ):
            return False, None

        return True, pt3d

    def reconstruct_from_phase(
        self,
        phase_map: np.ndarray,
        mask: Optional[np.ndarray] = None,
        roi: Optional[Tuple[int, int, int, int]] = None,
        sample_step: int = 4,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Tuple[bool, Optional[np.ndarray], str]:
        """
        Reconstructs dense 3D point cloud map from absolute unwrapped phase map.
        phase_map: (H, W) float64
        mask: Optional (H, W) uint8 binary mask (0 = ignore, 255 = reconstruct)
        roi: Optional (x, y, w, h)
        sample_step: Spatial sampling step (1 = full res, 4 = downsampled for speed)

        Returns: (success: bool, xyz_map: Optional[np.ndarray] of shape (out_H, out_W, 3), msg: str)
        """
        if not self.parameters_loaded:
            return False, None, "标定参数尚未加载"

        if phase_map is None or phase_map.ndim != 2:
            return False, None, "相位图必须为二维浮点矩阵"

        cam_w, cam_h = self.camera_image_size
        if roi is not None:
            rx, ry, rw, rh = roi
            rx = max(0, min(rx, cam_w - 1))
            ry = max(0, min(ry, cam_h - 1))
            rw = max(1, min(rw, cam_w - rx))
            rh = max(1, min(rh, cam_h - ry))
        else:
            rx, ry, rw, rh = 0, 0, cam_w, cam_h

        full_size_input = phase_map.shape == (cam_h, cam_w)
        cropped_input = roi is not None and phase_map.shape == (rh, rw)
        if not full_size_input and not cropped_input:
            expected = f"{cam_w}x{cam_h}"
            if roi is not None:
                expected += f"，或ROI尺寸 {rw}x{rh}"
            return (
                False,
                None,
                f"相位图分辨率 ({phase_map.shape[1]}x{phase_map.shape[0]}) 无效，期望 {expected}",
            )

        if mask is not None and mask.shape != phase_map.shape:
            return False, None, "重建掩膜尺寸必须与相位图一致"

        sample_step = max(1, sample_step)
        y_indices = np.arange(ry, ry + rh, sample_step)
        x_indices = np.arange(rx, rx + rw, sample_step)

        out_rows = len(y_indices)
        out_cols = len(x_indices)

        xyz_map = np.zeros((out_rows, out_cols, 3), dtype=np.float64)
        valid_count = 0

        proj_h = float(self.projector_image_size[1])
        ref_cycles = self.reference_cycle_count
        use_gpu = False

        # Keep the scalar path for small inputs and test overrides. Large camera
        # frames use chunked NumPy reconstruction to avoid millions of Python calls.
        if out_rows * out_cols <= 10000:
            for oi, y in enumerate(y_indices):
                for oj, x in enumerate(x_indices):
                    source_y = y if full_size_input else y - ry
                    source_x = x if full_size_input else x - rx
                    if mask is not None and mask[source_y, source_x] == 0:
                        continue
                    phi = phase_map[source_y, source_x]
                    if not np.isfinite(phi) or phi <= 1e-9:
                        continue
                    proj_v = (phi / ref_cycles) * proj_h
                    ok, pt3d = self.reconstruct_one_point((float(x), float(y)), proj_v)
                    if ok and pt3d is not None:
                        xyz_map[oi, oj] = pt3d
                        valid_count += 1
                if progress_callback is not None:
                    progress_callback((oi + 1) / max(1, out_rows))
        else:
            use_gpu = self.enable_gpu and is_cuda_available()
            chunk_rows = 256 if use_gpu else 64
            source_x = x_indices if full_size_input else x_indices - rx
            for row_start in range(0, out_rows, chunk_rows):
                row_end = min(out_rows, row_start + chunk_rows)
                ys = y_indices[row_start:row_end]
                source_y = ys if full_size_input else ys - ry
                phase_chunk = phase_map[np.ix_(source_y, source_x)]
                valid = np.isfinite(phase_chunk) & (phase_chunk > 1e-9)
                if mask is not None:
                    valid &= mask[np.ix_(source_y, source_x)] != 0

                flat_valid = valid.ravel()
                if np.any(flat_valid):
                    grid_x, grid_y = np.meshgrid(x_indices, ys)
                    camera_uv = np.column_stack((grid_x.ravel()[flat_valid], grid_y.ravel()[flat_valid]))
                    projector_v = phase_chunk.ravel()[flat_valid] / ref_cycles * proj_h
                    points, point_valid = self.reconstruct_points(camera_uv, projector_v)

                    chunk_xyz = np.zeros((row_end - row_start, out_cols, 3), dtype=np.float64)
                    valid_positions = np.flatnonzero(flat_valid)
                    chunk_xyz.reshape(-1, 3)[valid_positions[point_valid]] = points[point_valid]
                    xyz_map[row_start:row_end] = chunk_xyz
                    valid_count += int(np.count_nonzero(point_valid))

                if progress_callback is not None:
                    progress_callback(row_end / max(1, out_rows))

        if valid_count == 0:
            return False, None, "没有重建出有效三维点，请检查水平条纹相位范围和标定文件"

        gpu_info = f" (GPU加速: {get_cuda_device_name()})" if (use_gpu and get_cuda_device_name()) else ""
        return True, xyz_map, f"成功重建 {valid_count} 个三维点{gpu_info}"

    def reconstruct_points(
        self,
        camera_uv: np.ndarray,
        projector_v: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reconstructs 3D points from camera pixel coordinates and projector row coordinates.
        Automatically uses GPU (CUDA) if available and enabled, otherwise falls back
        to CPU vectorization.
        """
        if self.enable_gpu and is_cuda_available():
            try:
                return reconstruct_points_cuda(
                    camera_uv,
                    projector_v,
                    self.telecentric_projection,
                    self.telecentric_dist_5,
                    self.camera_image_size,
                    self.camera_pseudo_col_u,
                    self.camera_pseudo_col_v,
                    self.camera_line_direction,
                    self.projector_matrix,
                    self.projector_dist_14,
                    self.projector_tilt_matrix,
                    self.projector_image_size,
                    self.projector_row_plane_normals,
                    self.min_projector_z,
                    self.max_projector_z,
                )
            except Exception:
                pass
        return self._reconstruct_points_vectorized(camera_uv, projector_v)

    def _reconstruct_points_vectorized(
        self,
        camera_uv: np.ndarray,
        projector_v: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized equivalent of reconstruct_one_point for a camera-pixel batch."""
        count = len(camera_uv)
        points = np.zeros((count, 3), dtype=np.float64)
        valid = (
            np.isfinite(projector_v)
            & (projector_v >= 0.0)
            & (projector_v < float(self.projector_image_size[1]) - 1e-6)
        )
        if not np.any(valid):
            return points, valid

        observed = camera_uv.astype(np.float64, copy=False)
        ideal = observed.copy()
        width, height = self.camera_image_size
        cx = (width - 1) * 0.5
        cy = (height - 1) * 0.5
        norm_factor = max(1.0, 0.5 * max(width, height))
        d = self.telecentric_dist_5
        k1, k2, p1, p2 = d[0], d[1], d[2], d[3]
        k3 = d[4] if len(d) > 4 else 0.0

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

        valid &= np.all(np.isfinite(ideal), axis=1)
        bu = ideal[:, 0] - self.telecentric_projection[0, 3]
        bv = ideal[:, 1] - self.telecentric_projection[1, 3]
        line_pt = (
            bu[:, None] * self.camera_pseudo_col_u[None, :]
            + bv[:, None] * self.camera_pseudo_col_v[None, :]
        )
        line_dir = self.camera_line_direction
        if abs(line_dir[2]) < 1e-12:
            return points, np.zeros(count, dtype=bool)

        proj_h = self.projector_image_size[1]
        top = np.clip(np.floor(projector_v).astype(np.int64), 0, proj_h - 1)
        bottom = np.minimum(top + 1, proj_h - 1)
        frac = projector_v - top
        normals = (
            self.projector_row_plane_normals[top] * (1.0 - frac[:, None])
            + self.projector_row_plane_normals[bottom] * frac[:, None]
        )
        normal_len = np.linalg.norm(normals, axis=1)
        valid &= normal_len > 1e-12
        normals /= np.maximum(normal_len[:, None], 1e-12)
        denom = normals @ line_dir
        valid &= np.abs(denom) > 1e-12

        t_a = (self.min_projector_z - line_pt[:, 2]) / line_dir[2]
        t_b = (self.max_projector_z - line_pt[:, 2]) / line_dir[2]
        left = np.minimum(t_a, t_b)
        right = np.maximum(t_a, t_b)
        valid &= np.isfinite(left) & np.isfinite(right) & (right > left)

        def residual(t_values: np.ndarray) -> np.ndarray:
            pts = line_pt + t_values[:, None] * line_dir[None, :]
            return project_to_dlp(
                pts,
                self.projector_matrix,
                self.projector_dist_14,
                self.projector_tilt_matrix,
            )[:, 1] - projector_v

        res_left = residual(left)
        res_right = residual(right)
        valid &= np.isfinite(res_left) & np.isfinite(res_right) & (res_left * res_right <= 0.0)

        # Vectorized bisection is more predictable than per-point Python Newton loops.
        for _ in range(16):
            middle = 0.5 * (left + right)
            res_mid = residual(middle)
            use_left = res_left * res_mid <= 0.0
            right = np.where(use_left, middle, right)
            res_right = np.where(use_left, res_mid, res_right)
            left = np.where(use_left, left, middle)
            res_left = np.where(use_left, res_left, res_mid)

        t_final = 0.5 * (left + right)
        final_points = line_pt + t_final[:, None] * line_dir[None, :]
        projected = project_to_dlp(
            final_points,
            self.projector_matrix,
            self.projector_dist_14,
            self.projector_tilt_matrix,
        )
        valid &= (
            np.all(np.isfinite(final_points), axis=1)
            & np.all(np.isfinite(projected), axis=1)
            & (final_points[:, 2] >= self.min_projector_z)
            & (final_points[:, 2] <= self.max_projector_z)
            & (np.abs(projected[:, 1] - projector_v) <= 0.10)
            & (projected[:, 0] >= 0.0)
            & (projected[:, 0] < float(self.projector_image_size[0]))
        )
        points[valid] = final_points[valid]
        return points, valid
