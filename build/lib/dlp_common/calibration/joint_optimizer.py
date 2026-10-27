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
        square_size_mm: float = 5.0,
        enable_scheimpflug_tilt: bool = True,
        enable_projector_tangential: bool = False,
        enable_camera_distortion: bool = True,
        run_joint_optimization: bool = True,
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
        self.run_joint_optimization = run_joint_optimization
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
        if len(views) < 3:
            return False, result, f"有效位姿数量不足，当前只有 {len(views)} 组，至少需要 3 组（推荐 8 组以上）"

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
            # Optimize affine (8) + dist (4 or 5)
            def cam_residuals(params):
                aff = params[0:8].reshape(2, 4)
                d5 = params[8:13]
                res = []
                for v_idx in range(len(views)):
                    pred = project_telecentric_point(
                        dlp_3d_points_list[v_idx],
                        aff,
                        d5,
                        self.camera_image_size,
                    )
                    diff = pred - cam_points_list[v_idx]
                    res.append(diff.flatten())
                return np.concatenate(res)

            init_params = np.hstack([affine_2x4.flatten(), dist_cam_5])
            try:
                res_cam = least_squares(
                    cam_residuals,
                    init_params,
                    loss="huber",
                    f_scale=self.huber_delta,
                    max_nfev=150,
                )
                if res_cam.success:
                    affine_2x4 = res_cam.x[0:8].reshape(2, 4)
                    dist_cam_5 = res_cam.x[8:13]
            except Exception as e:
                result.warnings.append(f"远心相机非线性畸变优化异常: {e}")

        result.telecentric_projection_2x4 = affine_2x4
        result.telecentric_dist_5 = dist_cam_5.reshape(5, 1)

        # -------------------------------------------------------------
        # Step 3: Joint Optimization (Camera + Projector + Board Poses)
        # -------------------------------------------------------------
        if self.run_joint_optimization:
            num_views = len(views)
            # Parameter vector packing:
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

            def joint_residuals(p):
                # Unpack
                kp_mat = np.array([
                    [p[0], 0.0, p[2]],
                    [0.0, p[1], p[3]],
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
                    r_mat, _ = cv2.Rodrigues(rv)

                    # 3D points in DLP coordinates
                    pts_dlp = (r_mat @ views[v].object_points.T).T + tv.reshape(1, 3)

                    # Camera prediction
                    cam_pred = project_telecentric_point(pts_dlp, c_aff, c_dist, self.camera_image_size)
                    cam_diff = cam_pred - views[v].camera_points
                    res.append(cam_diff.flatten())

                    # Projector prediction
                    proj_pred = project_to_dlp(pts_dlp, kp_mat, dp_14)
                    proj_diff = proj_pred - views[v].projector_points
                    res.append(proj_diff.flatten())

                return np.concatenate(res)

            try:
                joint_opt = least_squares(
                    joint_residuals,
                    joint_params_0,
                    loss="huber",
                    f_scale=self.huber_delta,
                    max_nfev=200,
                )
                if joint_opt.success:
                    p = joint_opt.x
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

