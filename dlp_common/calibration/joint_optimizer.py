"""
Joint optimization engine for telecentric camera + Scheimpflug DLP projector.
"""

from typing import List, Optional, Tuple
import cv2
import numpy as np
from scipy.optimize import least_squares

from .models import CalibrationPose, CalibrationResult, CalibrationView
from .scheimpflug_dlp import compute_tilt_matrix, project_to_dlp
from .telecentric_camera import (
    decompose_affine_camera,
    fit_linear_affine,
    project_telecentric_point,
)


class JointCalibrationEngine:
    """
    Solves the combined camera + projector calibration problem with Huber-loss joint optimization.
    """

    def __init__(
        self,
        camera_image_size: Tuple[int, int] = (5328, 4608),
        projector_image_size: Tuple[int, int] = (1920, 1080),
        board_size: Tuple[int, int] = (11, 8),
        square_size_mm: float = 3.0,
        # enable_scheimpflug_tilt: bool = True,
        enable_scheimpflug_tilt: bool = False,
        enable_projector_tangential: bool = False,
        enable_camera_distortion: bool = True,
        enable_camera_k3: bool = False,
        run_joint_optimization: bool = True,
        refine_projector_in_joint: bool = True,
        minimum_view_count: int = 8,
        camera_only_iterations: int = 35,
        joint_iterations: int = 25,
        huber_delta: float = 1.5,
        verbose: bool = True,
    ):
        self.camera_image_size = camera_image_size
        self.projector_image_size = projector_image_size
        self.board_size = board_size
        self.square_size_mm = square_size_mm
        self.enable_scheimpflug_tilt = enable_scheimpflug_tilt
        self.enable_projector_tangential = enable_projector_tangential
        self.enable_camera_distortion = enable_camera_distortion
        self.enable_camera_k3 = enable_camera_k3
        self.run_joint_optimization = run_joint_optimization
        self.refine_projector_in_joint = refine_projector_in_joint
        self.minimum_view_count = minimum_view_count
        self.camera_only_iterations = camera_only_iterations
        self.joint_iterations = joint_iterations
        self.huber_delta = huber_delta
        self.verbose = verbose

    def calibrate(
        self,
        views: List[CalibrationView],
    ) -> Tuple[bool, CalibrationResult, str]:
        """
        Runs the complete calibration pipeline.
        Returns: (success: bool, result: CalibrationResult, message: str)
        """
        result = CalibrationResult()
        if len(views) < self.minimum_view_count:
            return (
                False,
                result,
                f"有效位姿数量不足，当前只有 {len(views)} 组，至少需要 {self.minimum_view_count} 组",
            )

        # -------------------------------------------------------------
        # Step 1: Calibrate Projector (Pinhole -> Scheimpflug Tilted Model)
        # -------------------------------------------------------------
        obj_pts_list = [v.object_points.astype(np.float32) for v in views]
        proj_pts_list = [v.projector_points.astype(np.float32) for v in views]
        result.view_names = [v.name for v in views]

        # Initial pinhole calibration for projector
        proj_w, proj_h = self.projector_image_size
        initial_k = np.array(
            [
                [2.0 * proj_w, 0.0, (proj_w - 1) * 0.5],
                [0.0, 2.0 * proj_w, (proj_h - 1) * 0.5],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        initial_dist = np.zeros((5, 1), dtype=np.float64)
        flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_K3
        if not self.enable_projector_tangential:
            flags |= cv2.CALIB_ZERO_TANGENT_DIST
        try:
            (
                rms_p,
                k_p,
                dist_p,
                rvecs_p,
                tvecs_p,
            ) = cv2.calibrateCamera(
                obj_pts_list,
                proj_pts_list,
                self.projector_image_size,
                initial_k,
                initial_dist,
                flags=flags,
                criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 150, 1e-10),
            )
        except Exception as e:
            return False, result, f"投影仪初值标定失败: {e}"

        # Ensure 14 distortion parameters
        dist_14 = np.zeros((14, 1), dtype=np.float64)
        if dist_p is not None:
            dist_flat = dist_p.flatten()
            dist_14[: min(14, len(dist_flat)), 0] = dist_flat[: min(14, len(dist_flat))]

        # If Scheimpflug tilt is enabled, try CALIB_TILTED_MODEL
        if self.enable_scheimpflug_tilt:
            try:
                tilted_flags = (
                    cv2.CALIB_USE_INTRINSIC_GUESS
                    | cv2.CALIB_TILTED_MODEL
                    | cv2.CALIB_FIX_K3
                    | cv2.CALIB_FIX_K4
                    | cv2.CALIB_FIX_K5
                    | cv2.CALIB_FIX_K6
                    | cv2.CALIB_FIX_S1_S2_S3_S4
                )
                if not self.enable_projector_tangential:
                    tilted_flags |= cv2.CALIB_ZERO_TANGENT_DIST
                (
                    tilted_rms,
                    tilted_k,
                    tilted_dist,
                    tilted_rvecs,
                    tilted_tvecs,
                ) = cv2.calibrateCamera(
                    obj_pts_list,
                    proj_pts_list,
                    self.projector_image_size,
                    k_p.copy(),
                    dist_14.copy(),
                    flags=tilted_flags,
                    criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 200, 1e-11),
                )
                tau_x = float(tilted_dist.reshape(-1)[12])
                tau_y = float(tilted_dist.reshape(-1)[13])
                sensible = (
                    np.isfinite(tilted_rms)
                    and tilted_rms <= max(2.0, rms_p * 1.5)
                    and abs(tau_x) < 1.2
                    and abs(tau_y) < 1.2
                    and tilted_k[0, 0] > 10.0
                    and tilted_k[1, 1] > 10.0
                )
                if sensible:
                    k_p = tilted_k
                    dist_14 = tilted_dist
                    rvecs_p = tilted_rvecs
                    tvecs_p = tilted_tvecs
                    rms_p = tilted_rms
                else:
                    result.warnings.append("OpenCV Scheimpflug 倾斜优化未收敛，保留小孔初值模型")
            except Exception as e:
                result.warnings.append(f"Scheimpflug 倾斜优化异常 ({e})，保留初值")

        result.projector_matrix = k_p
        result.projector_dist_14 = dist_14.reshape(14, 1)
        result.board_rvecs = [r.reshape(3, 1) for r in rvecs_p]
        result.board_tvecs = [t.reshape(3, 1) for t in tvecs_p]
        result.projector_rms = float(rms_p)

        # -------------------------------------------------------------
        # Step 2: Calibrate Telecentric Camera
        # -------------------------------------------------------------
        # Compute 3D board points in DLP coordinates for each view
        dlp_3d_points_list = []
        cam_points_list = []
        for i, view in enumerate(views):
            r_mat, _ = cv2.Rodrigues(result.board_rvecs[i])
            pts_dlp = (r_mat @ view.object_points.T).T + result.board_tvecs[i].reshape(1, 3)
            dlp_3d_points_list.append(pts_dlp)
            cam_points_list.append(view.camera_points.astype(np.float64))

        # Fit initial 2x4 affine matrix
        fit_ok, affine_2x4 = fit_linear_affine(dlp_3d_points_list, cam_points_list)
        if not fit_ok:
            return False, result, "远心相机初始仿射矩阵拟合失败"

        dist_cam_5 = np.zeros(5, dtype=np.float64)

        # Nonlinear refinement for camera affine and distortion
        if self.enable_camera_distortion:
            camera_params = np.hstack([affine_2x4.flatten(), dist_cam_5])
            camera_active = list(range(12))
            if self.enable_camera_k3:
                camera_active.append(12)

            def cam_residuals(p):
                aff = p[0:8].reshape(2, 4)
                d5 = p[8:13]
                res = []
                for v_idx in range(len(views)):
                    pred = project_telecentric_point(
                        dlp_3d_points_list[v_idx],
                        aff,
                        d5,
                        self.camera_image_size,
                    )
                    diff = pred - cam_points_list[v_idx]
                    res.append(self._huber_pairs(diff))
                return np.concatenate(res)

            try:
                ok_cam, cam_opt_p = self._run_levenberg_marquardt(
                    camera_params,
                    camera_active,
                    cam_residuals,
                    max_iterations=self.camera_only_iterations,
                )
                if ok_cam:
                    affine_2x4 = cam_opt_p[0:8].reshape(2, 4)
                    dist_cam_5 = cam_opt_p[8:13]
            except Exception as e:
                result.warnings.append(f"远心相机非线性畸变优化异常: {e}")

        result.telecentric_projection_2x4 = affine_2x4
        result.telecentric_dist_5 = dist_cam_5.reshape(5, 1)

        # -------------------------------------------------------------
        # Step 3: Joint Optimization (Camera + Projector + Board Poses)
        # -------------------------------------------------------------
        if self.run_joint_optimization:
            num_views = len(views)
            # Parameter vector packing matching C++:
            # [0..3]: fx, fy, cx, cy
            # [4..9]: k1, k2, p1, p2, tauX, tauY
            # [10..17]: affine 2x4 (8)
            # [18..22]: cam dist 5 (5)
            # [23 + 6*i .. 23 + 6*i + 5]: rvec_i (3), tvec_i (3)
            p_init = [
                k_p[0, 0],
                k_p[1, 1],
                k_p[0, 2],
                k_p[1, 2],
                dist_14[0, 0],
                dist_14[1, 0],
                dist_14[2, 0],
                dist_14[3, 0],
                dist_14[12, 0],
                dist_14[13, 0],
            ]
            aff_init = list(affine_2x4.flatten())
            cam_d_init = list(dist_cam_5.flatten())
            poses_init = []
            for i in range(num_views):
                poses_init.extend(result.board_rvecs[i].flatten())
                poses_init.extend(result.board_tvecs[i].flatten())

            joint_params_0 = np.array(p_init + aff_init + cam_d_init + poses_init, dtype=np.float64)

            joint_active = []
            if self.refine_projector_in_joint:
                joint_active.extend([0, 1, 2, 3, 4, 5])
                if self.enable_projector_tangential:
                    joint_active.extend([6, 7])
                if self.enable_scheimpflug_tilt:
                    joint_active.extend([8, 9])
            joint_active.extend(range(10, 18))
            if self.enable_camera_distortion:
                joint_active.extend([18, 19, 20, 21])
                if self.enable_camera_k3:
                    joint_active.append(22)
            joint_active.extend(range(23, len(joint_params_0)))

            def joint_residuals(p):
                fx, fy, cx, cy = p[0], p[1], p[2], p[3]
                if fx <= 10.0 or fy <= 10.0 or abs(p[8]) > 1.2 or abs(p[9]) > 1.2:
                    return None

                kp_mat = np.array([
                    [fx, 0.0, cx],
                    [0.0, fy, cy],
                    [0.0, 0.0, 1.0],
                ], dtype=np.float64)

                dp_14 = np.zeros(14, dtype=np.float64)
                dp_14[0:4] = p[4:8]
                dp_14[12] = p[8]
                dp_14[13] = p[9]

                c_aff = p[10:18].reshape(2, 4)
                c_dist = p[18:23]

                res = []
                pose_base = 23
                for v in range(num_views):
                    rv = p[pose_base + 6 * v : pose_base + 6 * v + 3]
                    tv = p[pose_base + 6 * v + 3 : pose_base + 6 * v + 6]
                    if tv[2] <= 1e-6:
                        return None

                    r_mat, _ = cv2.Rodrigues(rv)
                    pts_dlp = (r_mat @ views[v].object_points.T).T + tv.reshape(1, 3)

                    # Camera prediction
                    cam_pred = project_telecentric_point(pts_dlp, c_aff, c_dist, self.camera_image_size)
                    cam_diff = cam_pred - views[v].camera_points
                    res.append(self._huber_pairs(cam_diff))

                    # Projector prediction
                    proj_pred = project_to_dlp(pts_dlp, kp_mat, dp_14)
                    proj_diff = proj_pred - views[v].projector_points
                    res.append(self._huber_pairs(proj_diff))

                return np.concatenate(res)

            try:
                ok_joint, opt_p = self._run_levenberg_marquardt(
                    joint_params_0,
                    joint_active,
                    joint_residuals,
                    max_iterations=self.joint_iterations,
                )
                if ok_joint:
                    p = opt_p
                    result.projector_matrix = np.array([
                        [p[0], 0.0, p[2]],
                        [0.0, p[1], p[3]],
                        [0.0, 0.0, 1.0],
                    ], dtype=np.float64)

                    d14 = np.zeros((14, 1), dtype=np.float64)
                    d14[0:4, 0] = p[4:8]
                    d14[12, 0] = p[8]
                    d14[13, 0] = p[9]
                    result.projector_dist_14 = d14

                    result.telecentric_projection_2x4 = p[10:18].reshape(2, 4)
                    result.telecentric_dist_5 = p[18:23].reshape(5, 1)

                    pose_base = 23
                    for v in range(num_views):
                        result.board_rvecs[v] = p[pose_base + 6 * v : pose_base + 6 * v + 3].reshape(3, 1)
                        result.board_tvecs[v] = p[pose_base + 6 * v + 3 : pose_base + 6 * v + 6].reshape(3, 1)
            except Exception as e:
                result.warnings.append(f"联合优化异常 ({e})，保留独立标定结果")

        # -------------------------------------------------------------
        # Step 4: Errors and Extrinsic Decomposition
        # -------------------------------------------------------------
        sx, sy, r_ext, t_ext = decompose_affine_camera(
            result.telecentric_projection_2x4, self.camera_image_size
        )
        result.telecentric_scale_x = sx
        result.telecentric_scale_y = sy
        result.projector_to_camera_rotation = r_ext
        result.projector_to_camera_translation = t_ext

        # Compute individual and global RMS
        total_pts = sum(len(v.object_points) for v in views)
        total_cam_err_sq = 0.0
        total_proj_err_sq = 0.0

        for i, view in enumerate(views):
            r_mat, _ = cv2.Rodrigues(result.board_rvecs[i])
            pts_dlp = (r_mat @ view.object_points.T).T + result.board_tvecs[i].reshape(1, 3)

            cam_pred = project_telecentric_point(
                pts_dlp,
                result.telecentric_projection_2x4,
                result.telecentric_dist_5,
                self.camera_image_size,
            )
            cam_err = np.linalg.norm(cam_pred - view.camera_points, axis=1)
            cam_view_rms = float(np.sqrt(np.mean(cam_err**2)))
            result.camera_view_rms.append(cam_view_rms)
            total_cam_err_sq += np.sum(cam_err**2)

            proj_pred = project_to_dlp(
                pts_dlp,
                result.projector_matrix,
                result.projector_dist_14,
            )
            proj_err = np.linalg.norm(proj_pred - view.projector_points, axis=1)
            proj_view_rms = float(np.sqrt(np.mean(proj_err**2)))
            result.projector_view_rms.append(proj_view_rms)
            total_proj_err_sq += np.sum(proj_err**2)

        result.camera_rms = float(np.sqrt(total_cam_err_sq / total_pts))
        result.projector_rms = float(np.sqrt(total_proj_err_sq / total_pts))
        result.joint_rms = float(np.sqrt((total_cam_err_sq + total_proj_err_sq) / (2.0 * total_pts)))

        return True, result, "系统联合标定成功完成"

    def _huber_pairs(self, residual_pairs: np.ndarray) -> np.ndarray:
        """Apply the same vector-pair Huber weighting as CalibrationEngine.cpp."""
        pairs = np.asarray(residual_pairs, dtype=np.float64).reshape(-1, 2)
        if self.huber_delta <= 0.0:
            return pairs.ravel()
        lengths = np.linalg.norm(pairs, axis=1)
        weights = np.ones_like(lengths)
        robust = lengths > self.huber_delta
        weights[robust] = np.sqrt(self.huber_delta / lengths[robust])
        return (pairs * weights[:, None]).ravel()

    def _run_levenberg_marquardt(
        self,
        parameters: np.ndarray,
        active_indices: List[int],
        residual_fn,
        max_iterations: int = 25,
    ) -> Tuple[bool, np.ndarray]:
        """
        Custom Levenberg-Marquardt optimizer with column-normalized Jacobian.
        Exact numerical port of C++ CalibrationEngine.cpp runLevenbergMarquardt.
        """
        params = np.asarray(parameters, dtype=np.float64).copy()
        res = residual_fn(params)
        if res is None or len(res) == 0:
            return False, params

        cost = float(np.dot(res, res))
        if not np.isfinite(cost):
            return False, params

        n_active = len(active_indices)
        n_res = len(res)
        lam = 1e-3

        for iteration in range(max_iterations):
            # Compute numerical Jacobian for active columns
            J = np.zeros((n_res, n_active), dtype=np.float64)
            for col, idx in enumerate(active_indices):
                val = params[idx]
                step = 1e-7 if abs(val) < 0.1 else 1e-6 * max(1.0, abs(val))
                p_pert = params.copy()
                p_pert[idx] += step
                r_pert = residual_fn(p_pert)
                if r_pert is None or len(r_pert) != n_res:
                    return False, params
                J[:, col] = (r_pert - res) / step

            # Column normalization improves conditioning across diverse units
            col_norms = np.linalg.norm(J, axis=0)
            col_norms[col_norms < 1e-12] = 1.0
            J_scaled = J / col_norms

            JtJ = J_scaled.T @ J_scaled
            grad = J_scaled.T @ res

            step_accepted = False
            cost_drop = 0.0
            for _ in range(10):
                damped = JtJ + lam * np.eye(n_active)
                try:
                    delta = np.linalg.solve(damped, -grad)
                except np.linalg.LinAlgError:
                    lam = min(1e12, lam * 10.0)
                    continue

                p_cand = params.copy()
                p_cand[active_indices] += delta / col_norms
                r_cand = residual_fn(p_cand)
                if r_cand is None or len(r_cand) != n_res:
                    lam = min(1e12, lam * 10.0)
                    continue

                c_cand = float(np.dot(r_cand, r_cand))
                if np.isfinite(c_cand) and c_cand < cost:
                    cost_drop = (cost - c_cand) / max(1.0, cost)
                    params = p_cand
                    res = r_cand
                    cost = c_cand
                    lam = max(1e-12, lam * 0.3)
                    step_accepted = True
                    break
                lam = min(1e12, lam * 10.0)

            if not step_accepted or cost_drop < 1e-7:
                break

        return True, params


