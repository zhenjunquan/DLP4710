"""
OpenCV YAML and text report I/O for calibration parameters.
Fully bi-directionally compatible with the C++ calibration files.
"""

from datetime import datetime
import os
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from .models import CalibrationResult


class CalibrationIO:
    """Handles serialization and deserialization of calibration results."""

    @staticmethod
    def save_yml_and_txt(
        yml_path: str,
        txt_path: str,
        result: CalibrationResult,
        camera_image_size: Tuple[int, int],
        projector_image_size: Tuple[int, int],
        board_size: Tuple[int, int],
        square_size_mm: float,
        fringe_periods: List[float],
    ) -> Tuple[bool, str]:
        """
        Saves calibration result to OpenCV YAML and summary TXT files.
        """
        if (
            result.projector_matrix is None
            or result.telecentric_projection_2x4 is None
        ):
            return False, "标定参数为空，无法保存"

        os.makedirs(os.path.dirname(os.path.abspath(yml_path)), exist_ok=True)
        if txt_path:
            os.makedirs(os.path.dirname(os.path.abspath(txt_path)), exist_ok=True)

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            fs = cv2.FileStorage(yml_path, cv2.FILE_STORAGE_WRITE)
            if not fs.isOpened():
                return False, f"无法创建YML文件: {yml_path}"

            fs.write("calibration_time", now_str)
            fs.write("model", "telecentric_affine_camera_plus_scheimpflug_projector")
            fs.write("board_width", int(board_size[0]))
            fs.write("board_height", int(board_size[1]))
            fs.write("square_size_mm", float(square_size_mm))
            fs.write("camera_image_width", int(camera_image_size[0]))
            fs.write("camera_image_height", int(camera_image_size[1]))
            fs.write("projector_width", int(projector_image_size[0]))
            fs.write("projector_height", int(projector_image_size[1]))

            # Fringe periods
            fs.startWriteStruct("fringe_periods", cv2.FileNode_SEQ)
            for p in fringe_periods:
                fs.write("", float(p))
            fs.endWriteStruct()

            # Matrices
            fs.write("projector_matrix", result.projector_matrix.astype(np.float64))
            fs.write("projector_distortion_14", result.projector_dist_14.astype(np.float64))
            fs.write("telecentric_projection_2x4", result.telecentric_projection_2x4.astype(np.float64))
            fs.write("telecentric_distortion_5", result.telecentric_dist_5.astype(np.float64))
            if result.projector_to_camera_rotation is not None:
                fs.write("projector_to_camera_rotation", result.projector_to_camera_rotation.astype(np.float64))
            if result.projector_to_camera_translation is not None:
                fs.write("projector_to_camera_translation_xy_gauge", result.projector_to_camera_translation.astype(np.float64))

            fs.write("telecentric_scale_x_pixel_per_mm", float(result.telecentric_scale_x))
            fs.write("telecentric_scale_y_pixel_per_mm", float(result.telecentric_scale_y))
            fs.write("camera_rms_px", float(result.camera_rms))
            fs.write("projector_rms_px", float(result.projector_rms))
            fs.write("joint_rms_px", float(result.joint_rms))

            # Views
            fs.startWriteStruct("views", cv2.FileNode_SEQ)
            for i, name in enumerate(result.view_names):
                fs.startWriteStruct("", cv2.FileNode_MAP)
                fs.write("name", str(name))
                fs.write("board_rvec_projector", result.board_rvecs[i].astype(np.float64))
                fs.write("board_tvec_projector_mm", result.board_tvecs[i].astype(np.float64))
                cam_err = result.camera_view_rms[i] if i < len(result.camera_view_rms) else -1.0
                proj_err = result.projector_view_rms[i] if i < len(result.projector_view_rms) else -1.0
                fs.write("camera_rms_px", float(cam_err))
                fs.write("projector_rms_px", float(proj_err))
                fs.endWriteStruct()
            fs.endWriteStruct()

            # Warnings
            fs.startWriteStruct("warnings", cv2.FileNode_SEQ)
            for w in result.warnings:
                fs.write("", str(w))
            fs.endWriteStruct()

            fs.release()
        except Exception as e:
            return False, f"保存YML异常: {e}"

        # Save TXT report
        if txt_path:
            try:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write("Telecentric Camera + Scheimpflug DLP Calibration\n")
                    f.write("Model: telecentric affine camera + tilted-image-plane projector\n")
                    f.write(f"Calibration time: {now_str}\n\n")
                    f.write(f"Camera RMS: {result.camera_rms:.6f} px\n")
                    f.write(f"Projector RMS: {result.projector_rms:.6f} px\n")
                    f.write(f"Joint RMS: {result.joint_rms:.6f} px\n")
                    f.write(f"Telecentric scale X: {result.telecentric_scale_x:.6f} px/mm\n")
                    f.write(f"Telecentric scale Y: {result.telecentric_scale_y:.6f} px/mm\n\n")
                    f.write("Telecentric projection 2x4 (projector XYZ -> camera uv):\n")
                    f.write(f"{result.telecentric_projection_2x4}\n\n")
                    f.write("Telecentric distortion 5:\n")
                    f.write(f"{result.telecentric_dist_5.flatten()}\n\n")
                    f.write("Projector matrix:\n")
                    f.write(f"{result.projector_matrix}\n\n")
                    f.write("Projector distortion 14:\n")
                    f.write(f"{result.projector_dist_14.flatten()}\n\n")
                    if result.warnings:
                        f.write("Warnings:\n")
                        for w in result.warnings:
                            f.write(f"- {w}\n")
            except Exception as e:
                return False, f"保存TXT异常: {e}"

        return True, "标定文件保存成功"

    @staticmethod
    def load_yml(yml_path: str) -> Tuple[bool, Optional[CalibrationResult], Dict[str, Any], str]:
        """
        Loads calibration result from OpenCV YAML file.
        Returns: (success: bool, result: Optional[CalibrationResult], meta: dict, message: str)
        """
        if not os.path.isfile(yml_path):
            return False, None, {}, f"文件不存在: {yml_path}"

        try:
            fs = cv2.FileStorage(yml_path, cv2.FILE_STORAGE_READ)
            if not fs.isOpened():
                return False, None, {}, f"无法以OpenCV FileStorage格式打开: {yml_path}"

            model_node = fs.getNode("model")
            model_name = model_node.string() if not model_node.empty() else ""
            if "telecentric_affine_camera_plus_scheimpflug_projector" not in model_name:
                return (
                    False,
                    None,
                    {},
                    f"标定模型不匹配: '{model_name}'，必须使用远心相机+Scheimpflug DLP标定模型",
                )

            res = CalibrationResult()
            res.projector_matrix = fs.getNode("projector_matrix").mat()
            res.projector_dist_14 = fs.getNode("projector_distortion_14").mat()
            res.telecentric_projection_2x4 = fs.getNode("telecentric_projection_2x4").mat()
            res.telecentric_dist_5 = fs.getNode("telecentric_distortion_5").mat()

            rot_node = fs.getNode("projector_to_camera_rotation")
            if not rot_node.empty():
                res.projector_to_camera_rotation = rot_node.mat()

            trans_node = fs.getNode("projector_to_camera_translation_xy_gauge")
            if not trans_node.empty():
                res.projector_to_camera_translation = trans_node.mat()

            res.telecentric_scale_x = float(fs.getNode("telecentric_scale_x_pixel_per_mm").real())
            res.telecentric_scale_y = float(fs.getNode("telecentric_scale_y_pixel_per_mm").real())
            res.camera_rms = float(fs.getNode("camera_rms_px").real())
            res.projector_rms = float(fs.getNode("projector_rms_px").real())
            res.joint_rms = float(fs.getNode("joint_rms_px").real())

            # Metadata
            meta = {
                "calibration_time": fs.getNode("calibration_time").string(),
                "board_width": int(fs.getNode("board_width").real()),
                "board_height": int(fs.getNode("board_height").real()),
                "square_size_mm": float(fs.getNode("square_size_mm").real()),
                "camera_image_width": int(fs.getNode("camera_image_width").real()),
                "camera_image_height": int(fs.getNode("camera_image_height").real()),
                "projector_width": int(fs.getNode("projector_width").real()),
                "projector_height": int(fs.getNode("projector_height").real()),
            }

            fringe_periods_node = fs.getNode("fringe_periods")
            fringe_periods = []
            for i in range(fringe_periods_node.size()):
                fringe_periods.append(float(fringe_periods_node.at(i).real()))
            meta["fringe_periods"] = fringe_periods

            views_node = fs.getNode("views")
            res.view_names = []
            res.board_rvecs = []
            res.board_tvecs = []
            res.camera_view_rms = []
            res.projector_view_rms = []

            for i in range(views_node.size()):
                vn = views_node.at(i)
                res.view_names.append(vn.getNode("name").string())
                res.board_rvecs.append(vn.getNode("board_rvec_projector").mat())
                res.board_tvecs.append(vn.getNode("board_tvec_projector_mm").mat())
                res.camera_view_rms.append(float(vn.getNode("camera_rms_px").real()))
                res.projector_view_rms.append(float(vn.getNode("projector_rms_px").real()))

            fs.release()
            return True, res, meta, "标定文件加载成功"
        except Exception as e:
            return False, None, {}, f"读取标定文件异常: {e}"

