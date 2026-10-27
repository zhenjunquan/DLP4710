"""
Controller orchestrating business logic and state for calibration software.
"""

import os
import shutil
from datetime import datetime
from typing import List, Optional, Tuple
import cv2
import numpy as np

from dlp_common.calibration.calib_io import CalibrationIO
from dlp_common.calibration.corner_detector import CornerDetector
from dlp_common.calibration.joint_optimizer import JointCalibrationEngine
from dlp_common.calibration.models import CalibrationPose, CalibrationResult, CalibrationView
from dlp_common.config import SystemConfig
from dlp_common.fringe.phase_unwrapping import PhaseProcessor
from dlp_common.hardware.daheng_camera import DahengCameraDriver
from dlp_common.hardware.dlp_controller import DLP4710Controller


class CalibrationController:
    """
    Coordinates calibration poses, hardware interactions, and calibration calculations.
    """

    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config if config is not None else SystemConfig()

        self.camera = DahengCameraDriver(
            width=self.config.camera_width,
            height=self.config.camera_height,
        )
        self.dlp = DLP4710Controller()

        self.detector = CornerDetector(
            board_cols=self.config.board_cols,
            board_rows=self.config.board_rows,
            square_size_mm=self.config.square_size_mm,
        )
        self.phase_processor = PhaseProcessor(
            periods=self.config.fringe_periods,
            step_num=self.config.phase_step_count,
        )
        self.joint_engine = JointCalibrationEngine(
            camera_image_size=(self.config.camera_width, self.config.camera_height),
            projector_image_size=(self.config.projector_width, self.config.projector_height),
            board_size=(self.config.board_cols, self.config.board_rows),
            square_size_mm=self.config.square_size_mm,
            enable_scheimpflug_tilt=self.config.enable_scheimpflug_tilt,
            enable_projector_tangential=self.config.enable_projector_tangential,
            enable_camera_distortion=self.config.enable_camera_distortion,
            enable_camera_k3=self.config.enable_camera_k3,
            run_joint_optimization=self.config.run_joint_optimization,
            refine_projector_in_joint=self.config.refine_projector_in_joint,
            minimum_view_count=self.config.minimum_view_count,
            camera_only_iterations=self.config.camera_only_iterations,
            joint_iterations=self.config.joint_iterations,
            huber_delta=self.config.huber_delta_pixels,
        )

        self.poses: List[CalibrationPose] = []
        self.calibration_result: Optional[CalibrationResult] = None
        self.bin_path: str = ""
        self.session_output_folder: str = ""

    def update_board_params(self, cols: int, rows: int, square_size_mm: float):
        self.config.board_cols = cols
        self.config.board_rows = rows
        self.config.square_size_mm = square_size_mm
        self.detector.board_cols = cols
        self.detector.board_rows = rows
        self.detector.square_size_mm = square_size_mm
        self.detector.pattern_size = (cols, rows)
        self.joint_engine.board_size = (cols, rows)
        self.joint_engine.square_size_mm = square_size_mm
        self.calibration_result = None

    def add_pose_from_folder(self, folder_path: str) -> Tuple[bool, Optional[CalibrationPose], str]:
        """
        Loads a pose directory containing:
          - 0.bmp / board.bmp (chessboard image without fringe)
          - 12 horizontal fringe images (H_Fringe_1..12 or 1..12)
          - 12 vertical fringe images (V_Fringe_1..12 or 13..24)
        """
        if not os.path.isdir(folder_path):
            return False, None, f"目录不存在: {folder_path}"

        pose_name = os.path.basename(os.path.normpath(folder_path))
        pose = CalibrationPose(name=pose_name, folder_path=folder_path)

        # Check for board image
        candidates_board = ["0.bmp", "board.bmp", "chess.bmp"]
        board_img = None
        for b in candidates_board:
            bp = os.path.join(folder_path, b)
            if os.path.isfile(bp):
                board_img = cv2.imread(bp, cv2.IMREAD_GRAYSCALE)
                if board_img is not None:
                    break

        if board_img is None:
            return False, None, f"未在目录中找到棋盘格基底图像 (0.bmp 或 board.bmp): {folder_path}"

        pose.board_image = board_img
        pose.camera_image_size = (board_img.shape[1], board_img.shape[0])

        self.poses.append(pose)
        self.calibration_result = None
        return True, pose, f"成功导入姿态: {pose_name}"

    def delete_pose(self, index: int) -> bool:
        if 0 <= index < len(self.poses):
            self.poses.pop(index)
            self.calibration_result = None
            return True
        return False

    def get_valid_views(self) -> List[CalibrationView]:
        """
        Assembles CalibrationView objects for all poses with both valid corners and valid phase.
        """
        obj_pts = self.detector.create_object_points()
        views: List[CalibrationView] = []

        expected_image_size = None
        for p in self.poses:
            if (
                p.corner_valid
                and p.phase_valid
                and p.camera_corners is not None
                and p.projector_corners is not None
                and len(p.camera_corners) == len(obj_pts)
                and len(p.projector_corners) == len(obj_pts)
            ):
                if expected_image_size is None:
                    expected_image_size = p.camera_image_size
                    self.config.camera_width, self.config.camera_height = expected_image_size
                    self.joint_engine.camera_image_size = expected_image_size
                if p.camera_image_size != expected_image_size:
                    continue
                view = CalibrationView(
                    name=p.name,
                    object_points=obj_pts.copy(),
                    camera_points=p.camera_corners.copy(),
                    projector_points=p.projector_corners.copy(),
                )
                views.append(view)

        return views

    def save_calibration(self, yml_path: str, txt_path: str) -> Tuple[bool, str]:
        if self.calibration_result is None:
            return False, "尚未完成系统联合标定，无可保存参数"

        return CalibrationIO.save_yml_and_txt(
            yml_path=yml_path,
            txt_path=txt_path,
            result=self.calibration_result,
            camera_image_size=(self.config.camera_width, self.config.camera_height),
            projector_image_size=(self.config.projector_width, self.config.projector_height),
            board_size=(self.config.board_cols, self.config.board_rows),
            square_size_mm=self.config.square_size_mm,
            fringe_periods=self.config.fringe_periods,
        )

    @staticmethod
    def _write_image(file_path: str, image: np.ndarray):
        if not cv2.imwrite(file_path, np.asarray(image)):
            raise OSError(f"图像写入失败: {file_path}")

    @staticmethod
    def _phase_preview(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        finite = np.isfinite(values)
        normalized = np.zeros(values.shape, dtype=np.uint8)
        if np.any(finite):
            low = float(np.min(values[finite]))
            high = float(np.max(values[finite]))
            if high > low:
                normalized[finite] = np.clip(
                    (values[finite] - low) * 255.0 / (high - low), 0, 255
                ).astype(np.uint8)
        preview = cv2.applyColorMap(normalized, cv2.COLORMAP_VIRIDIS)
        preview[~finite] = 0
        return preview

    def _save_float_result(self, folder: str, name: str, values: np.ndarray):
        values32 = np.asarray(values, dtype=np.float32)
        np.save(os.path.join(folder, f"{name}.npy"), values32)
        self._write_image(os.path.join(folder, f"{name}.tiff"), values32)
        self._write_image(
            os.path.join(folder, f"{name}_preview.png"),
            self._phase_preview(values32),
        )

    def start_incremental_session(self, output_folder: str) -> Tuple[bool, str]:
        """Create a new disk-backed calibration session before acquisition."""
        if os.path.exists(output_folder):
            return False, f"目标文件夹已存在，请使用新的文件夹名称: {output_folder}"
        parent = os.path.dirname(os.path.abspath(output_folder))
        if not os.path.isdir(parent):
            return False, f"父目录不存在: {parent}"
        try:
            os.mkdir(output_folder)
            os.mkdir(os.path.join(output_folder, "poses"))
            self.session_output_folder = output_folder
            self._write_session_info(final_saved=False)
        except Exception as exc:
            shutil.rmtree(output_folder, ignore_errors=True)
            self.session_output_folder = ""
            return False, f"新建标定会话失败: {exc}"
        return True, f"标定会话已建立，采集结果将实时保存到: {output_folder}"

    def prepare_pose_storage(self, pose_index: int, pose_name: str) -> Tuple[bool, str, str]:
        if not self.session_output_folder:
            return False, "", "尚未建立标定会话文件夹"
        pose_dir = os.path.join(
            self.session_output_folder,
            "poses",
            f"{pose_index:02d}_{pose_name}",
        )
        if os.path.exists(pose_dir):
            return False, "", f"姿态目录已存在: {pose_dir}"
        try:
            os.makedirs(os.path.join(pose_dir, "horizontal"))
            os.makedirs(os.path.join(pose_dir, "vertical"))
        except Exception as exc:
            shutil.rmtree(pose_dir, ignore_errors=True)
            return False, "", f"创建姿态目录失败: {exc}"
        return True, pose_dir, ""

    def persist_decoded_pose(
        self,
        pose: CalibrationPose,
        horizontal_wrapped: List[np.ndarray],
        horizontal_modulation: np.ndarray,
        vertical_wrapped: List[np.ndarray],
        vertical_modulation: np.ndarray,
    ):
        """Persist decoded products from the decode worker, then release large arrays."""
        if not pose.folder_path:
            return
        board_path = os.path.join(pose.folder_path, "board.bmp")
        board = pose.board_image
        if board is None:
            board = cv2.imread(board_path, cv2.IMREAD_GRAYSCALE)
        if board is None:
            raise OSError(f"无法读取姿态基底图: {board_path}")

        if pose.camera_corners is not None:
            np.save(
                os.path.join(pose.folder_path, "camera_corners.npy"),
                np.asarray(pose.camera_corners, dtype=np.float32),
            )
            overlay = cv2.cvtColor(board, cv2.COLOR_GRAY2BGR) if board.ndim == 2 else board.copy()
            cv2.drawChessboardCorners(
                overlay,
                (self.config.board_cols, self.config.board_rows),
                np.asarray(pose.camera_corners, dtype=np.float32).reshape(-1, 1, 2),
                bool(pose.corner_valid),
            )
            self._write_image(
                os.path.join(pose.folder_path, "camera_corners_preview.png"), overlay
            )
        if pose.projector_corners is not None:
            np.save(
                os.path.join(pose.folder_path, "projector_corners.npy"),
                np.asarray(pose.projector_corners, dtype=np.float32),
            )

        for folder_name, wrapped, unwrapped, modulation in (
            ("horizontal", horizontal_wrapped, pose.horizontal_phase, horizontal_modulation),
            ("vertical", vertical_wrapped, pose.vertical_phase, vertical_modulation),
        ):
            folder = os.path.join(pose.folder_path, folder_name)
            for index, phase in enumerate(wrapped, start=1):
                self._save_float_result(folder, f"wrapped_frequency_{index:02d}", phase)
            self._save_float_result(folder, "unwrapped_phase", unwrapped)
            self._save_float_result(folder, "modulation", modulation)

        self._write_pose_info(pose)
        pose.board_image = None
        pose.horizontal_frames.clear()
        pose.vertical_frames.clear()
        pose.horizontal_phase = None
        pose.vertical_phase = None
        self._write_session_info(final_saved=False)

    @staticmethod
    def pose_phase_path(pose: CalibrationPose, direction: str = "horizontal") -> str:
        if not pose.folder_path:
            return ""
        return os.path.join(pose.folder_path, direction, "unwrapped_phase.npy")

    def _write_pose_info(self, pose: CalibrationPose):
        roi_text = "无" if pose.phase_roi is None else ", ".join(map(str, pose.phase_roi))
        lines = [
            f"姿态名称: {pose.name}",
            f"相机分辨率: {pose.camera_image_size[0]} x {pose.camera_image_size[1]}",
            f"角点有效: {pose.corner_valid}",
            f"相位有效: {pose.phase_valid}",
            f"相位ROI (x, y, w, h): {roi_text}",
            f"相机RMS: {pose.camera_view_error:.6f} px",
            f"投影仪RMS: {pose.projector_view_error:.6f} px",
        ]
        with open(os.path.join(pose.folder_path, "pose_info.txt"), "w", encoding="utf-8") as stream:
            stream.write("\n".join(lines) + "\n")

    def _write_session_info(self, final_saved: bool):
        if not self.session_output_folder:
            return
        lines = [
            "DLP4710 单目结构光增量标定结果",
            f"更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"已采集姿态数量: {len(self.poses)}",
            f"棋盘格内角点: {self.config.board_cols} x {self.config.board_rows}",
            f"棋盘格间距: {self.config.square_size_mm:.6f} mm",
            f"最终标定参数: {'已保存' if final_saved else '尚未保存或尚未完成标定'}",
            "说明: 原始图像和相位中间结果在每个姿态处理完成后立即写入硬盘。",
        ]
        with open(
            os.path.join(self.session_output_folder, "calibration_session.txt"),
            "w",
            encoding="utf-8",
        ) as stream:
            stream.write("\n".join(lines) + "\n")

    def finalize_incremental_session(self) -> Tuple[bool, str]:
        if not self.session_output_folder:
            return False, "尚未建立标定会话文件夹"
        try:
            final_saved = False
            if self.calibration_result is not None:
                calibration_dir = os.path.join(self.session_output_folder, "calibration")
                os.makedirs(calibration_dir, exist_ok=True)
                ok, message = self.save_calibration(
                    os.path.join(calibration_dir, "MonoSLCalibration.yml"),
                    os.path.join(calibration_dir, "MonoSLCalibration.txt"),
                )
                if not ok:
                    return False, message
                final_saved = True
            self._write_session_info(final_saved=final_saved)
        except Exception as exc:
            return False, f"更新标定会话失败: {exc}"
        if final_saved:
            return True, f"最终标定参数已更新到: {self.session_output_folder}"
        return True, f"中间结果已实时保存；当前尚无最终标定参数: {self.session_output_folder}"

    def save_complete_session(self, output_folder: str) -> Tuple[bool, str]:
        """Save raw captures, decoded phase products, correspondences and calibration."""
        if not self.poses:
            return False, "当前没有标定姿态，无法保存完整结果"
        if os.path.exists(output_folder):
            return False, f"目标文件夹已存在，请使用新的文件夹名称: {output_folder}"

        parent = os.path.dirname(os.path.abspath(output_folder))
        if not os.path.isdir(parent):
            return False, f"父目录不存在: {parent}"

        created = False
        try:
            os.mkdir(output_folder)
            created = True
            poses_dir = os.path.join(output_folder, "poses")
            os.mkdir(poses_dir)

            for pose_index, pose in enumerate(self.poses, start=1):
                pose_name = pose.name or f"Pose_{pose_index:02d}"
                pose_dir = os.path.join(poses_dir, f"{pose_index:02d}_{pose_name}")
                horizontal_dir = os.path.join(pose_dir, "horizontal")
                vertical_dir = os.path.join(pose_dir, "vertical")
                os.makedirs(horizontal_dir)
                os.makedirs(vertical_dir)

                if pose.board_image is None:
                    raise ValueError(f"{pose_name} 缺少棋盘格基底图")
                self._write_image(os.path.join(pose_dir, "board.bmp"), pose.board_image)

                if pose.board_image.ndim == 2:
                    overlay = cv2.cvtColor(pose.board_image, cv2.COLOR_GRAY2BGR)
                else:
                    overlay = np.asarray(pose.board_image).copy()
                if pose.camera_corners is not None:
                    cv2.drawChessboardCorners(
                        overlay,
                        (self.config.board_cols, self.config.board_rows),
                        np.asarray(pose.camera_corners, dtype=np.float32).reshape(-1, 1, 2),
                        bool(pose.corner_valid),
                    )
                    np.save(
                        os.path.join(pose_dir, "camera_corners.npy"),
                        np.asarray(pose.camera_corners, dtype=np.float32),
                    )
                self._write_image(os.path.join(pose_dir, "camera_corners_preview.png"), overlay)

                if len(pose.horizontal_frames) != 12 or len(pose.vertical_frames) != 12:
                    raise ValueError(
                        f"{pose_name} 条纹数量不完整: "
                        f"水平{len(pose.horizontal_frames)}张，竖直{len(pose.vertical_frames)}张"
                    )

                for direction, frames, phase, folder, prefix in (
                    ("水平", pose.horizontal_frames, pose.horizontal_phase, horizontal_dir, "H"),
                    ("竖直", pose.vertical_frames, pose.vertical_phase, vertical_dir, "V"),
                ):
                    for frame_index, frame in enumerate(frames, start=1):
                        self._write_image(
                            os.path.join(folder, f"{prefix}_Fringe_{frame_index:02d}.bmp"),
                            frame,
                        )
                    wrapped, modulation = self.phase_processor.solve_wrapped_phases(
                        frames, roi=pose.phase_roi
                    )
                    for frequency_index, wrapped_phase in enumerate(wrapped, start=1):
                        self._save_float_result(
                            folder,
                            f"wrapped_frequency_{frequency_index:02d}",
                            wrapped_phase,
                        )
                    unwrapped = phase
                    if unwrapped is None:
                        unwrapped = self.phase_processor.unwrap_wrapped_phases(wrapped)
                    self._save_float_result(folder, "unwrapped_phase", unwrapped)
                    self._save_float_result(folder, "modulation", modulation)

                if pose.projector_corners is not None:
                    np.save(
                        os.path.join(pose_dir, "projector_corners.npy"),
                        np.asarray(pose.projector_corners, dtype=np.float32),
                    )

                roi_text = "无" if pose.phase_roi is None else ", ".join(map(str, pose.phase_roi))
                pose_info = [
                    f"姿态名称: {pose_name}",
                    f"相机分辨率: {pose.camera_image_size[0]} x {pose.camera_image_size[1]}",
                    f"角点有效: {pose.corner_valid}",
                    f"相位有效: {pose.phase_valid}",
                    f"相位ROI (x, y, w, h): {roi_text}",
                    f"相机RMS: {pose.camera_view_error:.6f} px",
                    f"投影仪RMS: {pose.projector_view_error:.6f} px",
                ]
                with open(os.path.join(pose_dir, "pose_info.txt"), "w", encoding="utf-8") as stream:
                    stream.write("\n".join(pose_info) + "\n")

            calibration_saved = False
            if self.calibration_result is not None:
                calibration_dir = os.path.join(output_folder, "calibration")
                os.mkdir(calibration_dir)
                ok, message = self.save_calibration(
                    os.path.join(calibration_dir, "MonoSLCalibration.yml"),
                    os.path.join(calibration_dir, "MonoSLCalibration.txt"),
                )
                if not ok:
                    raise OSError(message)
                calibration_saved = True

            session_info = [
                "DLP4710 单目结构光完整标定结果",
                f"保存时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"姿态数量: {len(self.poses)}",
                f"棋盘格内角点: {self.config.board_cols} x {self.config.board_rows}",
                f"棋盘格间距: {self.config.square_size_mm:.6f} mm",
                f"投影分辨率: {self.config.projector_width} x {self.config.projector_height}",
                f"三频周期: {', '.join(str(v) for v in self.config.fringe_periods)}",
                f"最终标定参数: {'已保存' if calibration_saved else '尚未完成标定，本次仅保存采集与相位结果'}",
                "说明: NPY保存原始浮点值，TIFF保存浮点图，PNG保存归一化伪彩预览。",
            ]
            with open(os.path.join(output_folder, "calibration_session.txt"), "w", encoding="utf-8") as stream:
                stream.write("\n".join(session_info) + "\n")
        except Exception as exc:
            if created:
                shutil.rmtree(output_folder, ignore_errors=True)
            return False, f"保存完整标定结果失败: {exc}"

        suffix = "，包含最终YML/TXT参数" if self.calibration_result is not None else "（尚无最终标定参数）"
        return True, f"完整标定结果已保存到: {output_folder}{suffix}"

