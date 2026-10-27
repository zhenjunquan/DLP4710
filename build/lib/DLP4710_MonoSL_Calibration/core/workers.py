"""
Asynchronous QThread workers for responsive UI during calibration and image processing.
"""

import os
from typing import List, Tuple
import cv2
import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal

from dlp_common.calibration.corner_detector import CornerDetector
from dlp_common.calibration.joint_optimizer import JointCalibrationEngine
from dlp_common.calibration.models import CalibrationPose, CalibrationResult, CalibrationView
from dlp_common.fringe.phase_unwrapping import PhaseProcessor
from dlp_common.fringe.utils import bilinear_sample
from dlp_common.hardware.daheng_camera import DahengCameraDriver
from dlp_common.hardware.dlp_controller import DLP4710Controller


class DetectCornersWorker(QThread):
    """Batch chessboard corner detection worker."""
    progress = pyqtSignal(int, int, str)  # current, total, pose_name
    finished = pyqtSignal(bool, str)

    def __init__(self, poses: List[CalibrationPose], detector: CornerDetector):
        super().__init__()
        self.poses = poses
        self.detector = detector

    def run(self):
        total = len(self.poses)
        success_count = 0
        for i, pose in enumerate(self.poses):
            self.progress.emit(i + 1, total, f"检测角点: {pose.name}")
            if pose.board_image is None and pose.folder_path:
                # Load board image from folder
                board_path = os.path.join(pose.folder_path, "0.bmp")
                if not os.path.isfile(board_path):
                    board_path = os.path.join(pose.folder_path, "board.bmp")
                if os.path.isfile(board_path):
                    pose.board_image = cv2.imread(board_path, cv2.IMREAD_GRAYSCALE)

            if pose.board_image is not None:
                pose.camera_image_size = (pose.board_image.shape[1], pose.board_image.shape[0])
                ok, corners, msg = self.detector.detect_corners(pose.board_image)
                if ok and corners is not None:
                    pose.camera_corners = corners
                    pose.corner_valid = True
                    success_count += 1
                else:
                    pose.corner_valid = False
            else:
                pose.corner_valid = False

        self.finished.emit(True, f"批量角点检测完成: {success_count}/{total} 个姿态成功检测")


class DecodePhasesWorker(QThread):
    """Batch phase unwrapping and DLP coordinate mapping worker."""
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(bool, str)

    def __init__(
        self,
        poses: List[CalibrationPose],
        processor: PhaseProcessor,
        projector_size: Tuple[int, int],
        roi_margin: int = 32,
    ):
        super().__init__()
        self.poses = poses
        self.processor = processor
        self.projector_size = projector_size
        self.roi_margin = roi_margin

    def run(self):
        total = len(self.poses)
        success_count = 0
        ref_cycle = self.processor.periods[0]
        proj_w, proj_h = self.projector_size

        for i, pose in enumerate(self.poses):
            self.progress.emit(i + 1, total, f"相位解算: {pose.name}")
            if not pose.corner_valid or pose.camera_corners is None:
                continue

            # Load frames if needed
            if len(pose.horizontal_frames) != 12 or len(pose.vertical_frames) != 12:
                self._load_pose_frames(pose)

            if len(pose.horizontal_frames) != 12 or len(pose.vertical_frames) != 12:
                continue

            # Compute ROI around corners
            pts = pose.camera_corners
            min_x = max(0, int(np.floor(np.min(pts[:, 0])) - self.roi_margin))
            max_x = min(pose.camera_image_size[0], int(np.ceil(np.max(pts[:, 0])) + self.roi_margin))
            min_y = max(0, int(np.floor(np.min(pts[:, 1])) - self.roi_margin))
            max_y = min(pose.camera_image_size[1], int(np.ceil(np.max(pts[:, 1])) + self.roi_margin))
            roi = (min_x, min_y, max_x - min_x, max_y - min_y)
            pose.phase_roi = roi

            try:
                # Unwrap horizontal and vertical
                h_unwrapped, _ = self.processor.unwrap_phase(pose.horizontal_frames, roi=roi)
                v_unwrapped, _ = self.processor.unwrap_phase(pose.vertical_frames, roi=roi)

                pose.horizontal_phase = h_unwrapped
                pose.vertical_phase = v_unwrapped

                # Map camera corners to DLP corners using bilinear interpolation
                proj_corners = []
                valid_all = True
                for c_pt in pose.camera_corners:
                    local_x = c_pt[0] - roi[0]
                    local_y = c_pt[1] - roi[1]

                    phi_h = bilinear_sample(h_unwrapped, local_x, local_y)
                    phi_v = bilinear_sample(v_unwrapped, local_x, local_y)

                    if not np.isfinite(phi_h) or not np.isfinite(phi_v):
                        valid_all = False
                        break

                    proj_y = (phi_h / ref_cycle) * proj_h
                    proj_x = (phi_v / ref_cycle) * proj_w

                    if not (0.0 <= proj_x < proj_w and 0.0 <= proj_y < proj_h):
                        valid_all = False
                        break

                    proj_corners.append([proj_x, proj_y])

                if valid_all and len(proj_corners) == len(pose.camera_corners):
                    pose.projector_corners = np.array(proj_corners, dtype=np.float32)
                    pose.phase_valid = True
                    success_count += 1
                else:
                    pose.phase_valid = False
            except Exception:
                pose.phase_valid = False

        self.finished.emit(True, f"批量相位解算完成: {success_count}/{total} 个姿态成功解算")

    def _load_pose_frames(self, pose: CalibrationPose):
        if not pose.folder_path or not os.path.isdir(pose.folder_path):
            return
        h_frames = []
        v_frames = []

        # Try standard filenames: H_Fringe_1..12 or 1..12, 13..24
        for idx in range(1, 13):
            candidates = [
                os.path.join(pose.folder_path, f"H_Fringe_{idx}.bmp"),
                os.path.join(pose.folder_path, f"H_{idx}.bmp"),
                os.path.join(pose.folder_path, f"{idx}.bmp"),
            ]
            for c in candidates:
                if os.path.isfile(c):
                    img = cv2.imread(c, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        h_frames.append(img)
                        break

        for idx in range(1, 13):
            candidates = [
                os.path.join(pose.folder_path, f"V_Fringe_{idx}.bmp"),
                os.path.join(pose.folder_path, f"V_{idx}.bmp"),
                os.path.join(pose.folder_path, f"{idx + 12}.bmp"),
            ]
            for c in candidates:
                if os.path.isfile(c):
                    img = cv2.imread(c, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        v_frames.append(img)
                        break

        pose.horizontal_frames = h_frames
        pose.vertical_frames = v_frames


class CalibrateWorker(QThread):
    """Joint system calibration solver worker."""
    finished = pyqtSignal(bool, CalibrationResult, str)

    def __init__(self, engine: JointCalibrationEngine, views: List[CalibrationView]):
        super().__init__()
        self.engine = engine
        self.views = views

    def run(self):
        try:
            ok, res, msg = self.engine.calibrate(self.views)
            self.finished.emit(ok, res, msg)
        except Exception as e:
            self.finished.emit(False, CalibrationResult(), f"联合标定崩溃: {e}")


class BurnBinWorker(QThread):
    """DLP BIN burning worker."""
    finished = pyqtSignal(bool, str)

    def __init__(self, controller: DLP4710Controller, bin_path: str):
        super().__init__()
        self.controller = controller
        self.bin_path = bin_path

    def run(self):
        ok, msg = self.controller.burn_bin(self.bin_path)
        self.finished.emit(ok, msg)

