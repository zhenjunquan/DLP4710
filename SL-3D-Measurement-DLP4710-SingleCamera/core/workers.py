"""
Asynchronous workers for live camera preview and 3D measurement reconstruction.
"""

import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal

from dlp_common.fringe.multi_exposure import (
    fuse_multi_exposure_fringe_patterns,
    fuse_multi_exposure_phases,
)
from dlp_common.fringe.phase_unwrapping import PhaseProcessor
from dlp_common.hardware.daheng_camera import DahengCameraDriver
from dlp_common.hardware.dlp_controller import DLP4710Controller
from dlp_common.reconstruction.cuda_reconstructor import (
    get_cuda_device_name,
    is_cuda_available,
)
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

            frames = []
            max_attempts = 5
            for attempt in range(max_attempts):
                if attempt > 0:
                    self.msleep(200)
                    if hasattr(self.dlp, "prepare_pattern_mode"):
                        try:
                            self.dlp.prepare_pattern_mode()
                        except Exception:
                            pass
                    self.msleep(100)

                if hasattr(self.camera, "flush_buffer"):
                    self.camera.flush_buffer()

                self.msleep(30)
                self.progress.emit(
                    50,
                    "正在启动DLP并等待12帧硬触发图像..."
                    if attempt == 0
                    else f"硬触发图像不足，正在自动重试启动DLP (第 {attempt+1}/{max_attempts} 次)...",
                )
                if not self.dlp.start_work(0):
                    if hasattr(self.dlp, "prepare_pattern_mode"):
                        self.dlp.prepare_pattern_mode()
                        self.msleep(50)
                        if not self.dlp.start_work(0):
                            raise RuntimeError(self.dlp.last_error or "DLP启动失败")
                    else:
                        raise RuntimeError(self.dlp.last_error or "DLP启动失败")

                try:
                    ok, frames = self.camera.grab_triggered_sequence(
                        12,
                        frame_timeout_ms=1500,
                        overall_timeout_ms=15000,
                        flush_before=False,
                    )
                except TypeError:
                    ok, frames = self.camera.grab_triggered_sequence(
                        12,
                        frame_timeout_ms=1500,
                        overall_timeout_ms=15000,
                    )
                if ok and len(frames) == 12:
                    break

                if attempt < max_attempts - 1:
                    self.msleep(300)
                    if hasattr(self.camera, "flush_buffer"):
                        self.camera.flush_buffer()
                    self.msleep(50)

            if len(frames) != 12:
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
    """Burn and verify the measurement pattern BIN without blocking the UI."""

    progress = pyqtSignal(int, str)
    verified = pyqtSignal(bool, dict, str)
    completed = pyqtSignal(bool, str)

    def __init__(self, dlp: DLP4710Controller, bin_path: str, verify_projection: bool = True):
        super().__init__()
        self.dlp = dlp
        self.bin_path = bin_path
        self.verify_projection = verify_projection

    def run(self):
        def _cb(pct: int, msg: str):
            self.progress.emit(pct, msg)

        try:
            ok, report, message = self.dlp.burn_and_verify_bin(
                self.bin_path,
                verify_projection=self.verify_projection,
                progress_callback=_cb
            )
            self.verified.emit(ok, report, message)
            self.completed.emit(ok, message)
        except Exception as exc:
            err_msg = f"烧录验证异常: {exc}"
            self.verified.emit(False, {"error": str(exc)}, err_msg)
            self.completed.emit(False, err_msg)


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

            gpu_active = bool(getattr(self.reconstructor, "enable_gpu", True) and is_cuda_available())
            dev_name = get_cuda_device_name() if gpu_active else ""
            accel_text = f" (GPU加速: {dev_name})" if dev_name else " (CPU计算)"
            self.progress.emit(70, f"正在进行空间光线相交重构{accel_text}...")
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


class MultiExposureCaptureWorker(QThread):
    """
    Captures base gray frame and multi-exposure (HDR) 12-frame fringe sequences
    across multiple camera exposure and/or DLP light intensity settings without blocking UI.
    """

    progress = pyqtSignal(int, str)
    completed = pyqtSignal(bool, object, str)  # (ok, brackets_frames, msg)
    gray_frame_captured = pyqtSignal(np.ndarray)

    def __init__(
        self,
        camera: DahengCameraDriver,
        dlp: DLP4710Controller,
        brackets: List[Dict[str, float]],
        gain_db: float = 0.0,
        gray_exposure_us: float = 150000.0,
    ):
        super().__init__()
        self.camera = camera
        self.dlp = dlp
        self.brackets = brackets
        self.gain_db = gain_db
        self.gray_exposure_us = gray_exposure_us
        self.gray_frame: Optional[np.ndarray] = None

    def _require(self, ok: bool, fallback: str):
        if not ok:
            raise RuntimeError(self.camera.last_error or fallback)

    def run(self):
        all_brackets_frames: List[List[np.ndarray]] = []
        message = ""
        num_brackets = len(self.brackets)
        if num_brackets == 0:
            self.completed.emit(False, [], "多曝光档位配置为空")
            return

        try:
            # 1. 采集基准灰度图
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

            # 2. 循环采集各档曝光条纹序列
            # 先停止当前采集，将曝光时间预设为第1档的曝光时间，给传感器留出稳定时间（避免基准灰度图800ms等超长曝光残留）
            self.camera.stop_grabbing()
            self._require(self.camera.set_trigger_mode(False), "关闭相机触发模式失败")
            first_exp = float(self.brackets[0].get("exposure_us", 3000.0))
            first_gain = float(self.brackets[0].get("gain_db", self.gain_db))
            self._require(self.camera.set_exposure(first_exp), "预设第1档曝光失败")
            self._require(self.camera.set_gain(first_gain), "预设第1档增益失败")
            current_cam_exp = first_exp
            current_cam_gain = first_gain
            self.msleep(150)  # 关键：彻底等待基准灰度图超长曝光（如800ms）在传感器内部排空！

            # 配置为 Line2 硬触发模式并启动采集流
            self._require(
                self.camera.set_trigger_mode(True, source="Line2"),
                "设置 Line2 硬触发失败",
            )
            self._require(self.camera.start_grabbing(), "启动相机采集流失败")

            for k, bracket in enumerate(self.brackets):
                exp_us = float(bracket.get("exposure_us", 3000.0))
                led_pct = float(bracket.get("led_percent", 100.0))
                b_gain = float(bracket.get("gain_db", self.gain_db))

                prog_base = 15 + int(80 * k / num_brackets)
                self.progress.emit(
                    prog_base,
                    f"正在配置曝光档位 {k+1}/{num_brackets} (曝光: {int(exp_us)} μs, 投影光强: {int(led_pct)}%)...",
                )

                # 设置投影仪光强 (仅在数值变化时发送I2C命令，避免总线冲突)
                if hasattr(self.dlp, "set_brightness_percent"):
                    curr_pct = getattr(self.dlp, "brightness_percent", -1.0)
                    if abs(curr_pct - led_pct) > 0.05:
                        self.dlp.set_brightness_percent(led_pct)
                        self.msleep(60)  # 等待驱动电流稳定

                # 设置相机曝光与增益 (仅在数值变化时更新寄存器)
                if abs(exp_us - current_cam_exp) > 0.5:
                    self._require(self.camera.set_exposure(exp_us), f"设置档位{k+1}曝光时间失败")
                    current_cam_exp = exp_us
                    self.msleep(80)  # 等待相机传感器曝光时间生效稳定
                if abs(b_gain - current_cam_gain) > 0.05:
                    self._require(self.camera.set_gain(b_gain), f"设置档位{k+1}增益失败")
                    current_cam_gain = b_gain
                    self.msleep(40)

                # 带有自动重试保护的条纹序列采集 (最多采集5次)
                frames = []
                max_attempts = 5
                for attempt in range(max_attempts):
                    # 若重试，重新载入DLP序列表并重整触发信号
                    if attempt > 0:
                        self.msleep(200)
                        if hasattr(self.dlp, "prepare_pattern_mode"):
                            try:
                                self.dlp.prepare_pattern_mode()
                            except Exception:
                                pass
                        self.msleep(100)

                    # 采集前清空旧缓冲区并保证布防时间
                    if hasattr(self.camera, "flush_buffer"):
                        self.camera.flush_buffer()
                    self.msleep(30)

                    self.progress.emit(
                        prog_base + 8,
                        f"正在启动DLP投射档位 {k+1}/{num_brackets} (12帧条纹)..."
                        if attempt == 0
                        else f"档位 {k+1}/{num_brackets} 图像不足，正在自动重试第 {k+1} 档 (第 {attempt+1}/{max_attempts} 次)...",
                    )

                    if not self.dlp.start_work(0):
                        if hasattr(self.dlp, "prepare_pattern_mode"):
                            self.dlp.prepare_pattern_mode()
                            self.msleep(50)
                            if not self.dlp.start_work(0):
                                raise RuntimeError(self.dlp.last_error or f"DLP启动档位{k+1}投影失败")
                        else:
                            raise RuntimeError(self.dlp.last_error or f"DLP启动档位{k+1}投影失败")

                    try:
                        ok_seq, frames = self.camera.grab_triggered_sequence(
                            12,
                            frame_timeout_ms=1500,
                            overall_timeout_ms=15000,
                            flush_before=False,  # 绝不在 start_work 之后二次清空缓冲区！
                        )
                    except TypeError:
                        ok_seq, frames = self.camera.grab_triggered_sequence(
                            12,
                            frame_timeout_ms=1500,
                            overall_timeout_ms=15000,
                        )

                    if ok_seq and len(frames) == 12:
                        break

                    # 重试前等待 DLP 投影完全停歇
                    if attempt < max_attempts - 1:
                        self.msleep(300)
                        if hasattr(self.camera, "flush_buffer"):
                            self.camera.flush_buffer()
                        self.msleep(50)

                if len(frames) != 12:
                    if hasattr(self.dlp, "prepare_pattern_mode"):
                        try:
                            self.dlp.prepare_pattern_mode()
                        except Exception:
                            pass
                    raise RuntimeError(
                        f"档位 {k+1}/{num_brackets} 硬触发图像不足12张，实际收到 {len(frames)} 张"
                    )

                all_brackets_frames.append(frames)
                # 留出 150ms 供 DLP 单次序列完全结束并让硬件就绪
                self.msleep(150)

            self.progress.emit(100, f"全部 {num_brackets} 档多曝光条纹采集完成")
            message = f"成功采集基准灰度图与 {num_brackets} 档多曝光条纹图像 (共 {num_brackets * 12} 帧)"
        except Exception as exc:
            message = f"在线多曝光采集失败: {exc}"
        finally:
            try:
                self.camera.stop_grabbing()
                self.camera.set_trigger_mode(False)
            except Exception:
                pass

        success = len(all_brackets_frames) == num_brackets and num_brackets > 0
        self.completed.emit(success, all_brackets_frames, message)


class MultiExposureMeasureWorker(QThread):
    """
    Executes full Multi-Exposure (HDR) 3D measurement pipeline:
    1. Phase unwrapping (3-freq 4-step) for each exposure bracket
    2. Multi-exposure HDR phase fusion (best modulation, saturation rejection)
    3. Mono structured light 3D point cloud reconstruction (GPU accelerated)
    """

    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, dict, str)

    def __init__(
        self,
        reconstructor: MonoSLReconstructor,
        processor: PhaseProcessor,
        brackets_frames: List[List[np.ndarray]],
        roi: Optional[Tuple[int, int, int, int]] = None,
        sample_step: int = 4,
        min_modulation: float = 5.0,
        sat_threshold: float = 250.0,
        gray_frame: Optional[np.ndarray] = None,
        brackets_info: Optional[List[Dict[str, float]]] = None,
        compute_per_bracket_phases: bool = False,
    ):
        super().__init__()
        self.reconstructor = reconstructor
        self.processor = processor
        self.brackets_frames = brackets_frames
        self.roi = roi
        self.sample_step = sample_step
        self.min_modulation = min_modulation
        self.sat_threshold = sat_threshold
        self.gray_frame = gray_frame
        self.brackets_info = brackets_info or []
        self.compute_per_bracket_phases = compute_per_bracket_phases

    def run(self):
        start_time = time.time()
        num_brackets = len(self.brackets_frames)
        if num_brackets == 0:
            self.finished.emit(False, {}, "多曝光条纹帧组为空")
            return

        try:
            # 论文多曝光基准算法: 饱和像素剔除 + 最大调制度选取 + 条纹融合 (Jiang [9] & Zhu [10])
            # 核心优化: 三维点云重构仅需要对融合后的1组高动态(HDR)条纹进行一次相位解算与解包。
            # 默认跳过前置的13档冗余独立解包，测量总耗时从 90+ 秒暴降至 2~3 秒！
            if self.compute_per_bracket_phases:
                phases = []
                modulations = []
                all_wrapped_phases = []
                for k, frames in enumerate(self.brackets_frames):
                    pct = 5 + int(35 * k / num_brackets)
                    self.progress.emit(pct, f"正在计算档位 {k+1}/{num_brackets} 包裹相位与解包...")
                    wrapped_list, avg_mod = self.processor.solve_wrapped_phases(frames, roi=self.roi)
                    unwrapped = self.processor.unwrap_wrapped_phases(wrapped_list)
                    phases.append(unwrapped)
                    modulations.append(avg_mod)
                    all_wrapped_phases.append(wrapped_list)
            else:
                phases = [None] * num_brackets
                modulations = [None] * num_brackets
                all_wrapped_phases = [None] * num_brackets

            # 论文多曝光基准算法: 饱和像素剔除 + 最大调制度选取 + 条纹融合 (Jiang [9] & Zhu [10])
            self.progress.emit(25, f"正在执行 {num_brackets} 档条纹饱和剔除与最优调制度流式融合 (GPU)...")
            fused_frames, fused_mod, fused_mask, exp_indices = fuse_multi_exposure_fringe_patterns(
                brackets_frames=self.brackets_frames,
                sat_threshold=self.sat_threshold,
                min_modulation=self.min_modulation,
            )

            # 从融合的高动态范围条纹图直接解算包裹相位与绝对相位 (仅需1次，GPU极速求解)
            self.progress.emit(55, "正在从融合条纹解算绝对相位与调制掩膜 (GPU极速)...")
            fused_wrapped, _ = self.processor.solve_wrapped_phases(fused_frames, roi=self.roi)
            fused_phase = self.processor.unwrap_wrapped_phases(fused_wrapped)
            mask = self.processor.compute_mask(fused_mod, min_modulation=self.min_modulation)

            gpu_active = bool(getattr(self.reconstructor, "enable_gpu", True) and is_cuda_available())
            dev_name = get_cuda_device_name() if gpu_active else ""
            accel_text = f" (GPU加速: {dev_name})" if dev_name else " (CPU计算)"
            self.progress.emit(70, f"正在进行空间光线相交重构{accel_text}...")
            ok, xyz_map, msg = self.reconstructor.reconstruct_from_phase(
                phase_map=fused_phase,
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

            self.progress.emit(92, "正在提取三维点云与多曝光度量统计...")
            valid_pts, _ = extract_valid_points(xyz_map)
            elapsed = time.time() - start_time

            raw_frame = self.gray_frame if self.gray_frame is not None else fused_frames[0]
            bracket_counts = {}
            for k in range(num_brackets):
                bracket_counts[k] = int(np.sum((exp_indices == k) & (mask > 0)))

            result = {
                "is_multi_exposure": True,
                "brackets_frames": self.brackets_frames,
                "brackets_info": self.brackets_info,
                "fused_frames": fused_frames,
                "frames": fused_frames,  # 融合后的12帧高动态范围条纹图 (供2D图谱和导出使用)
                "raw_frame": raw_frame,
                "gray_frame": self.gray_frame,
                "wrapped_phases": fused_wrapped,
                "wrapped_phase": fused_wrapped[0],
                "all_wrapped_phases": all_wrapped_phases,
                "all_unwrapped_phases": phases,
                "all_modulations": modulations,
                "unwrapped_phase": fused_phase,
                "modulation": fused_mod,
                "mask": mask,
                "exposure_indices": exp_indices,
                "bracket_counts": bracket_counts,
                "xyz_map": xyz_map,
                "valid_points": valid_pts,
                "point_count": len(valid_pts),
                "elapsed_seconds": elapsed,
            }

            self.finished.emit(
                True,
                result,
                f"多曝光三维重构完成 ({num_brackets}档融合)，提取 {len(valid_pts):,} 个有效三维点，耗时 {elapsed:.2f} 秒",
            )
        except Exception as e:
            self.finished.emit(False, {}, f"多曝光测量重构异常: {e}")

