"""
Asynchronous workers for live camera preview and 3D measurement reconstruction.
"""

import time
from typing import Dict, List, Optional, Tuple
import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal

from dlp_common.fringe.phase_unwrapping import PhaseProcessor
from dlp_common.hardware.daheng_camera import DahengCameraDriver
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
    ):
        super().__init__()
        self.reconstructor = reconstructor
        self.processor = processor
        self.frames = frames
        self.roi = roi
        self.sample_step = sample_step
        self.min_modulation = min_modulation

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
            )

            if not ok or xyz_map is None:
                self.finished.emit(False, {}, f"三维重构失败: {msg}")
                return

            self.progress.emit(90, "正在提取三维点云与边界度量...")
            valid_pts, _ = extract_valid_points(xyz_map)
            elapsed = time.time() - start_time

            result = {
                "raw_frame": self.frames[0],
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

