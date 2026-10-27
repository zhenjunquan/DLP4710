"""
Asynchronous QThread workers for responsive UI during calibration and image processing.
"""

import os
import shutil
from typing import Callable, List, Optional, Tuple
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


class CapturePoseWorker(QThread):
    """Capture one board image and one 24-frame pattern sequence off the UI thread."""

    progress = pyqtSignal(int, str)
    completed = pyqtSignal(bool, object, str)

    def __init__(
        self,
        camera: DahengCameraDriver,
        dlp: DLP4710Controller,
        detector: CornerDetector,
        board_exposure_us: float,
        pattern_exposure_us: float,
        pose_folder: str = "",
    ):
        super().__init__()
        self.camera = camera
        self.dlp = dlp
        self.detector = detector
        self.board_exposure_us = board_exposure_us
        self.pattern_exposure_us = pattern_exposure_us
        self.pose_folder = pose_folder

    def _require(self, ok: bool, fallback: str):
        if not ok:
            raise RuntimeError(self.camera.last_error or fallback)

    def run(self):
        pose = None
        error = ""
        try:
            self.progress.emit(5, "正在切换相机到基底图采集模式...")
            self.camera.stop_grabbing()
            self._require(self.camera.set_trigger_mode(False), "关闭相机触发模式失败")
            self._require(self.camera.set_exposure(self.board_exposure_us), "设置基底曝光失败")
            self._require(
                self.camera.set_trigger_mode(True, source="Software"),
                "设置相机软触发失败",
            )
            self._require(self.camera.start_grabbing(), "开启相机采集流失败")

            self.msleep(20)
            ok, board_frame = self.camera.grab_frame()
            if not ok or board_frame is None:
                raise RuntimeError(self.camera.last_error or "无法读取相机当前基底图像")

            if board_frame.ndim == 3:
                board_gray = cv2.cvtColor(board_frame, cv2.COLOR_BGR2GRAY)
            else:
                board_gray = board_frame
            if self.pose_folder and not cv2.imwrite(
                os.path.join(self.pose_folder, "board.bmp"), board_gray
            ):
                raise OSError("棋盘格基底图写入硬盘失败")

            def save_pattern_frame(frame_index: int, frame: np.ndarray):
                if frame_index < 12:
                    folder = os.path.join(self.pose_folder, "horizontal")
                    filename = f"H_Fringe_{frame_index + 1:02d}.bmp"
                else:
                    folder = os.path.join(self.pose_folder, "vertical")
                    filename = f"V_Fringe_{frame_index - 11:02d}.bmp"
                if not cv2.imwrite(os.path.join(folder, filename), frame):
                    raise OSError(f"条纹图像写入硬盘失败: {filename}")

            self.progress.emit(20, "基底图采集完成，正在准备24帧条纹...")
            self.camera.stop_grabbing()
            self._require(self.camera.set_trigger_mode(False), "关闭相机触发模式失败")

            self._require(self.camera.set_exposure(self.pattern_exposure_us), "设置条纹曝光失败")
            self.progress.emit(30, "等待条纹曝光稳定...")
            self.msleep(1500)
            self._require(
                self.camera.set_trigger_mode(True, source="Line2"),
                "设置Line2硬触发失败",
            )
            self._require(self.camera.start_grabbing(), "开启相机硬触发采集流失败")

            self.msleep(20)
            self.progress.emit(40, "DLP正在投影，等待24帧硬触发图像...")
            if not self.dlp.start_work(0):
                raise RuntimeError(self.dlp.last_error or "DLP启动投影失败")

            if self.pose_folder:
                ok, pattern_frames = self.camera.grab_triggered_sequence(
                    24,
                    frame_timeout_ms=1000,
                    overall_timeout_ms=30000,
                    frame_callback=save_pattern_frame,
                    retain_frames=False,
                )
            else:
                ok, pattern_frames = self.camera.grab_triggered_sequence(
                    24,
                    frame_timeout_ms=1000,
                    overall_timeout_ms=30000,
                )
            if not ok:
                raise RuntimeError("硬触发条纹图像不足24张，请检查DLP触发和相机Line2")
            horizontal_frames = pattern_frames[:12] if pattern_frames else []
            vertical_frames = pattern_frames[12:] if pattern_frames else []

            self.progress.emit(90, "图像采集完成，正在检测棋盘格角点...")
            ok_corner, corners, corner_msg = self.detector.detect_corners(board_gray)
            pose = CalibrationPose(
                folder_path=self.pose_folder,
                board_image=board_gray,
                camera_image_size=(board_gray.shape[1], board_gray.shape[0]),
                horizontal_frames=horizontal_frames,
                vertical_frames=vertical_frames,
                camera_corners=corners if ok_corner else None,
                corner_valid=bool(ok_corner and corners is not None),
            )
            self.progress.emit(100, "当前姿态采集完成")
            if pose.corner_valid:
                message = f"采集完成，检测到 {len(corners)} 个棋盘格角点"
            else:
                message = f"采集完成，但未检测到完整棋盘格角点: {corner_msg}"
        except Exception as exc:
            error = str(exc)
            message = f"姿态采集失败: {error}"
            if self.pose_folder:
                shutil.rmtree(self.pose_folder, ignore_errors=True)
        finally:
            try:
                self.camera.stop_grabbing()
                self.camera.set_trigger_mode(False)
                self.camera.set_exposure(self.board_exposure_us)
                if self.camera.is_opened:
                    self.camera.start_grabbing()
            except Exception as exc:
                if not error:
                    pose = None
                    message = f"恢复相机预览模式失败: {exc}"

        self.completed.emit(pose is not None, pose, message)


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
        result_saver: Optional[Callable] = None,
    ):
        super().__init__()
        self.poses = poses
        self.processor = processor
        self.projector_size = projector_size
        self.roi_margin = roi_margin
        self.result_saver = result_saver

    def run(self):
        total = len(self.poses)
        success_count = 0
        phase_map_count = 0
        failures = []
        ref_cycle = self.processor.periods[0]
        proj_w, proj_h = self.projector_size

        for i, pose in enumerate(self.poses):
            self.progress.emit(i + 1, total, f"相位解算: {pose.name}")
            if not pose.corner_valid or pose.camera_corners is None:
                failures.append(f"{pose.name}: 缺少有效棋盘格角点")
                continue

            horizontal_paths, vertical_paths = self._find_pose_frame_paths(pose)
            in_memory = len(pose.horizontal_frames) == 12 and len(pose.vertical_frames) == 12
            on_disk = len(horizontal_paths) == 12 and len(vertical_paths) == 12
            if not in_memory and not on_disk:
                failures.append(
                    f"{pose.name}: 条纹数量不正确 "
                    f"(内存水平{len(pose.horizontal_frames)}张/竖直{len(pose.vertical_frames)}张，"
                    f"硬盘水平{len(horizontal_paths)}张/竖直{len(vertical_paths)}张)"
                )
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
                if in_memory:
                    h_wrapped, h_modulation = self.processor.solve_wrapped_phases(
                        pose.horizontal_frames, roi=roi
                    )
                    v_wrapped, v_modulation = self.processor.solve_wrapped_phases(
                        pose.vertical_frames, roi=roi
                    )
                else:
                    h_wrapped, h_modulation = self.processor.solve_wrapped_phases_from_files(
                        horizontal_paths, roi=roi
                    )
                    v_wrapped, v_modulation = self.processor.solve_wrapped_phases_from_files(
                        vertical_paths, roi=roi
                    )
                h_unwrapped = self.processor.unwrap_wrapped_phases(h_wrapped)
                v_unwrapped = self.processor.unwrap_wrapped_phases(v_wrapped)

                pose.horizontal_phase = h_unwrapped
                pose.vertical_phase = v_unwrapped
                phase_map_count += 1

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
                    failures.append(f"{pose.name}: 相位图已生成，但部分投影坐标无效或越界")
                if self.result_saver is not None:
                    self.result_saver(
                        pose,
                        h_wrapped,
                        h_modulation,
                        v_wrapped,
                        v_modulation,
                    )
            except Exception as exc:
                pose.phase_valid = False
                failures.append(f"{pose.name}: {exc}")

        message = (
            f"相位解算完成: {phase_map_count}/{total} 个姿态生成相位图，"
            f"{success_count}/{total} 个姿态获得有效DLP坐标"
        )
        if failures:
            message += "；" + "；".join(failures)
        self.finished.emit(True, message)

    def _find_pose_frame_paths(self, pose: CalibrationPose):
        if not pose.folder_path or not os.path.isdir(pose.folder_path):
            return [], []
        h_paths = []
        v_paths = []

        # Try standard filenames: H_Fringe_1..12 or 1..12, 13..24
        for idx in range(1, 13):
            candidates = [
                os.path.join(pose.folder_path, "horizontal", f"H_Fringe_{idx:02d}.bmp"),
                os.path.join(pose.folder_path, "horizontal", f"H_{idx:02d}.bmp"),
                os.path.join(pose.folder_path, f"H_Fringe_{idx}.bmp"),
                os.path.join(pose.folder_path, f"H_{idx}.bmp"),
                os.path.join(pose.folder_path, f"{idx}.bmp"),
            ]
            for c in candidates:
                if os.path.isfile(c):
                    h_paths.append(c)
                    break

        for idx in range(1, 13):
            candidates = [
                os.path.join(pose.folder_path, "vertical", f"V_Fringe_{idx:02d}.bmp"),
                os.path.join(pose.folder_path, "vertical", f"V_{idx:02d}.bmp"),
                os.path.join(pose.folder_path, f"V_Fringe_{idx}.bmp"),
                os.path.join(pose.folder_path, f"V_{idx}.bmp"),
                os.path.join(pose.folder_path, f"{idx + 12}.bmp"),
            ]
            for c in candidates:
                if os.path.isfile(c):
                    v_paths.append(c)
                    break
        return h_paths, v_paths


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

