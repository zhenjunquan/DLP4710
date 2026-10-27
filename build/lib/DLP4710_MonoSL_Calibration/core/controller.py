"""
Controller orchestrating business logic and state for calibration software.
"""

import os
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
            run_joint_optimization=self.config.run_joint_optimization,
            huber_delta=self.config.huber_delta_pixels,
        )

        self.poses: List[CalibrationPose] = []
        self.calibration_result: Optional[CalibrationResult] = None
        self.bin_path: str = ""

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
        return True, pose, f"成功导入姿态: {pose_name}"

    def delete_pose(self, index: int) -> bool:
        if 0 <= index < len(self.poses):
            self.poses.pop(index)
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

