"""
Asynchronous workers for live camera preview and 3D measurement reconstruction.
"""

import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal

from dlp_common.fringe.phase_unwrapping import PhaseProcessor
from dlp_common.hardware.daheng_camera import DahengCameraDriver
from dlp_common.hardware.dlp_controller import DLP4710Controller
from dlp_common.reconstruction.point_cloud_io import extract_valid_points
from dlp_common.reconstruction.reconstructor import MonoSLReconstructor


class LivePreviewWorker(QThread):
    """Continuous camera frame capture thread for live preview."""
    frame_received = pyqtSignal(np.ndarray)

    def __init__(self, camera: DahengCameraDriver):
        super().__init__()
        self.camera = camera
        self._running = False

    def run(self):
        self._running = True
        while self._running:
            ok, frame = self.camera.grab_frame(timeout_ms=500)
            if ok and frame is not None:
                self.frame_received.emit(frame)
            time.sleep(0.04)  # ~25 FPS

    def stop(self):
        self._running = False
        self.wait(1000)


class CaptureMeasurementWorker(QThread):
    """Capture base gray frame and the 12-frame measurement sequence without blocking the UI."""

    progress = pyqtSignal(int, str)
    completed = pyqtSignal(bool, object, str)
    gray_frame_captured = pyqtSignal(np.ndarray)

    def __init__(
        self,
        camera: DahengCameraDriver,
        dlp: DLP4710Controller,
        exposure_us: float,
        gain_db: float,
        gray_exposure_us: float = 150000.0,
    ):
        super().__init__()
        self.camera = camera
        self.dlp = dlp
        self.exposure_us = exposure_us
        self.gain_db = gain_db
        self.gray_exposure_us = gray_exposure_us
        self.gray_frame: Optional[np.ndarray] = None

    def _require(self, ok: bool, fallback: str):
        if not ok:
            raise RuntimeError(self.camera.last_error or fallback)

    def run(self):
        frames = []
        message = ""
        try:
            # 1. 首先先采集曝光为设置曝光时间的相机灰度图
            self.progress.emit(5, f"正在采集基准灰度图 (曝光 {int(self.gray_exposure_us)} μs)...")
            self.camera.stop_grabbing()
            self._require(self.camera.set_trigger_mode(False), "关闭相机触发模式失败")
            self._require(self.camera.set_exposure(self.gray_exposure_us), "设置灰度图曝光失败")
            self._require(self.camera.set_gain(self.gain_db), "设置相机增益失败")
            self._require(
                self.camera.set_trigger_mode(True, source="Software"),
                "设置相机软触发失败",
            )
            self._require(self.camera.start_grabbing(), "启动灰度图采集流失败")

            self.msleep(50)
            if hasattr(self.camera, "grab_frame"):
                ok_gray, gray_frame = self.camera.grab_frame(timeout_ms=3000)
                if not ok_gray or gray_frame is None:
                    raise RuntimeError(self.camera.last_error or "采集基准灰度图失败")
            else:
                gray_frame = np.zeros((48, 64), dtype=np.uint8)

            if gray_frame.ndim == 3:
                gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_BGR2GRAY)

            self.gray_frame = gray_frame
            self.gray_frame_captured.emit(gray_frame)
            self.progress.emit(25, "基准灰度图采集完成，正在配置12帧条纹投影...")

            # 2. 然后执行设置曝光时间的 投影采集12帧
            self.camera.stop_grabbing()
            self._require(self.camera.set_trigger_mode(False), "关闭相机触发模式失败")
            self._require(self.camera.set_exposure(self.exposure_us), "设置条纹投影曝光失败")
            self._require(self.camera.set_gain(self.gain_db), "设置相机增益失败")
            self.progress.emit(35, f"等待投影曝光稳定 ({int(self.exposure_us)} μs)...")
            self.msleep(150)

            self._require(
                self.camera.set_trigger_mode(True, source="Line2"),
                "设置Line2硬触发失败",
            )
            self._require(self.camera.start_grabbing(), "启动大恒相机采集流失败")

            if hasattr(self.camera, "flush_buffer"):
                self.camera.flush_buffer()

            self.msleep(30)
            self.progress.emit(50, "正在启动DLP并等待12帧硬触发图像...")
            if not self.dlp.start_work(0):
                raise RuntimeError(self.dlp.last_error or "DLP启动失败")

            ok, frames = self.camera.grab_triggered_sequence(
                12,
                frame_timeout_ms=1500,
                overall_timeout_ms=15000,
            )
            if not ok or len(frames) != 12:
                # 异常时若条纹序列被打断，重置模式以便下次采集
                if hasattr(self.dlp, "prepare_pattern_mode"):
                    try:
                        self.dlp.prepare_pattern_mode()
                    except Exception:
                        pass
                raise RuntimeError(f"硬触发图像不足12张，实际收到 {len(frames)} 张")
            self.progress.emit(100, "12帧条纹采集完成")
            message = "成功采集基准灰度图与12帧硬触发条纹图像"
        except Exception as exc:
            message = f"在线采集失败: {exc}"
        finally:
            # 原版C++参考设计：正常结束严禁调用 stop_work()，DLP单次序列自然结束，
            # 保持I2C总线和Pattern模式就绪，供下一次采集直接触发。
            try:
                self.camera.stop_grabbing()
                self.camera.set_trigger_mode(False)
            except Exception:
                pass

        self.completed.emit(len(frames) == 12, frames, message)


class BurnMeasurementBinWorker(QThread):
    """Burn the measurement pattern BIN without blocking the UI."""

    completed = pyqtSignal(bool, str)

    def __init__(self, dlp: DLP4710Controller, bin_path: str):
        super().__init__()
        self.dlp = dlp
        self.bin_path = bin_path

    def run(self):
        ok, message = self.dlp.burn_bin(self.bin_path)
        self.completed.emit(ok, message)


class MeasureWorker(QThread):
    """
    Executes full 3D measurement pipeline in background thread:
    1. Phase unwrapping (3-freq 4-step)
    2. Modulation mask calculation
    3. Mono structured light 3D point cloud reconstruction
    """
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, dict, str)

    def __init__(
        self,
        reconstructor: MonoSLReconstructor,
        processor: PhaseProcessor,
        frames: List[np.ndarray],
        roi: Optional[Tuple[int, int, int, int]] = None,
        sample_step: int = 4,
        min_modulation: float = 5.0,
        gray_frame: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.reconstructor = reconstructor
        self.processor = processor
        self.frames = frames
        self.roi = roi
        self.sample_step = sample_step
        self.min_modulation = min_modulation
        self.gray_frame = gray_frame

    def run(self):
        start_time = time.time()
        try:
            self.progress.emit(20, "正在计算包裹相位与三频外差解包...")
            wrapped_list, avg_mod = self.processor.solve_wrapped_phases(self.frames, roi=self.roi)
            unwrapped_phase = self.processor.unwrap_wrapped_phases(wrapped_list)
            modulation = avg_mod

            self.progress.emit(50, "正在计算表面反射调制掩膜...")
            mask = self.processor.compute_mask(modulation, min_modulation=self.min_modulation)

            self.progress.emit(70, "正在利用远心-Scheimpflug模型进行空间光线相交重构...")
            ok, xyz_map, msg = self.reconstructor.reconstruct_from_phase(
                phase_map=unwrapped_phase,
                mask=mask,
                roi=self.roi,
                sample_step=self.sample_step,
                progress_callback=lambda fraction: self.progress.emit(
                    70 + min(19, int(fraction * 19)),
                    f"正在进行空间光线相交重构... {fraction * 100:.0f}%",
                ),
            )

            if not ok or xyz_map is None:
                self.finished.emit(False, {}, f"三维重构失败: {msg}")
                return

            self.progress.emit(90, "正在提取三维点云与边界度量...")
            valid_pts, _ = extract_valid_points(xyz_map)
            elapsed = time.time() - start_time

            raw_frame = self.gray_frame if self.gray_frame is not None else self.frames[0]
            result = {
                "frames": list(self.frames),
                "raw_frame": raw_frame,
                "gray_frame": self.gray_frame,
                "wrapped_phases": wrapped_list,
                "wrapped_phase": wrapped_list[0],
                "unwrapped_phase": unwrapped_phase,
                "modulation": modulation,
                "mask": mask,
                "xyz_map": xyz_map,
                "valid_points": valid_pts,
                "point_count": len(valid_pts),
                "elapsed_seconds": elapsed,
            }

            self.finished.emit(True, result, f"三维重构完成，提取 {len(valid_pts):,} 个有效三维点，耗时 {elapsed:.2f} 秒")
        except Exception as e:
            self.finished.emit(False, {}, f"测量重构异常: {e}")

