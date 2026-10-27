"""
Controller managing hardware, calibration parameters, and measurement pipeline.
"""

import os
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from dlp_common.config import SystemConfig
from dlp_common.fringe.phase_unwrapping import PhaseProcessor
from dlp_common.hardware.daheng_camera import DahengCameraDriver
from dlp_common.hardware.dlp_controller import DLP4710Controller
from dlp_common.reconstruction.point_cloud_io import save_point_cloud_ply, save_point_cloud_txt
from dlp_common.reconstruction.reconstructor import MonoSLReconstructor


class MeasurementController:
    """
    Business logic and state management for 3D measurement application.
    """

    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config if config is not None else SystemConfig()

        self.reconstructor = MonoSLReconstructor()
        self.camera = DahengCameraDriver(
            width=self.config.camera_width,
            height=self.config.camera_height,
        )
        self.dlp = DLP4710Controller()
        self.phase_processor = PhaseProcessor(
            periods=self.config.fringe_periods,
            step_num=self.config.phase_step_count,
        )

        self.calib_file_path: str = ""
        self.latest_result: Optional[Dict] = None

    def load_calibration(self, yml_path: str) -> Tuple[bool, str]:
        """
        Loads the system calibration parameters from YAML.
        """
        ok = self.reconstructor.load_parameters(yml_path)
        if ok:
            self.calib_file_path = yml_path
            # Update camera and projector specs from calibration
            cw, ch = self.reconstructor.camera_image_size
            self.camera.target_width = cw
            self.camera.target_height = ch
            if self.camera.is_mock or not self.camera.is_opened:
                self.camera.actual_width = cw
                self.camera.actual_height = ch
            self.phase_processor.periods = self.reconstructor.fringe_periods
            return True, f"成功加载标定文件: {yml_path}\n相机分辨率: {cw}x{ch}, 投影分辨率: {self.reconstructor.projector_image_size[0]}x{self.reconstructor.projector_image_size[1]}"
        else:
            return False, self.reconstructor.last_error

    def load_offline_frames(self, folder_path: str) -> Tuple[bool, List[np.ndarray], str]:
        """
        Loads 12 horizontal fringe frames from an offline directory.
        """
        if not os.path.isdir(folder_path):
            return False, [], f"目录不存在: {folder_path}"

        frames = []
        for i in range(1, 13):
            candidates = [
                os.path.join(folder_path, f"H_Fringe_{i}.bmp"),
                os.path.join(folder_path, f"Fringe_{i}.bmp"),
                os.path.join(folder_path, f"{i}.bmp"),
            ]
            found = False
            for c in candidates:
                if os.path.isfile(c):
                    img = cv2.imread(c, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        frames.append(img)
                        found = True
                        break
            if not found:
                break

        if len(frames) != 12:
            return False, [], f"在目录中未集齐12张条纹图像 (需 1..12 或 H_Fringe_1..12)，当前找到 {len(frames)} 张"

        return True, frames, f"成功载入12张条纹图像: {folder_path}"

    def save_ply(self, file_path: str) -> Tuple[bool, str]:
        if self.latest_result is None or "xyz_map" not in self.latest_result:
            return False, "当前尚无测量结果，无法导出"

        xyz_map = self.latest_result["xyz_map"]
        ok, count, msg = save_point_cloud_ply(file_path, xyz_map)
        return ok, msg

    def save_txt(self, file_path: str) -> Tuple[bool, str]:
        if self.latest_result is None or "xyz_map" not in self.latest_result:
            return False, "当前尚无测量结果，无法导出"

        xyz_map = self.latest_result["xyz_map"]
        ok, count, msg = save_point_cloud_txt(file_path, xyz_map)
        return ok, msg

