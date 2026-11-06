"""
Focus Assistant business controller managing camera acquisition, focus evaluation,
and DLP projection assistance.
"""

import os
import sys
import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from dlp_common.config import SystemConfig
from dlp_common.focus.focus_metric import (
    FocusMetricType,
    compute_focus_peaking_mask,
    compute_focus_score,
)
from dlp_common.focus.focus_tracker import FocusState, FocusTracker
from dlp_common.hardware.daheng_camera import DahengCameraDriver
from dlp_common.hardware.dlp_controller import DLP4710Controller


class FocusController:
    """
    Controller coordinating real-time camera frames, sharpness evaluation,
    dynamic peak tracking, and optional DLP structured pattern projection.
    """

    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config if config is not None else SystemConfig()

        self.camera = DahengCameraDriver(
            width=self.config.camera_width,
            height=self.config.camera_height,
        )
        self.dlp = DLP4710Controller()

        # Focus evaluation settings
        self.metric_type: FocusMetricType = FocusMetricType.LAPLACIAN_VAR
        self.roi: Optional[Tuple[int, int, int, int]] = None
        self.peaking_enabled: bool = True
        self.peaking_threshold: float = 35.0
        self.sound_enabled: bool = True

        # Dynamic focus tracker
        self.tracker = FocusTracker(
            history_len=150,
            in_focus_ratio=0.92,
            overshot_threshold=0.85,
        )

        # Simulation / mock fallback state
        self._is_simulated: bool = False
        self._sim_z_step: float = 0.0

    @property
    def is_camera_open(self) -> bool:
        return self.camera.is_opened or self._is_simulated

    def detect_cameras(self) -> List[Dict]:
        return self.camera.detect_camera_devices()

    def connect_camera(self, device_id=0) -> Tuple[bool, str]:
        """Connects to physical camera or enables mock simulation if unavailable."""
        ok = self.camera.open(device_id)
        if ok:
            self._is_simulated = False
            self.camera.set_trigger_mode(False)
            self.camera.start_grabbing()
            return True, "已成功连接大恒工业相机"
        else:
            # Enable simulation mode so user/tests can preview without hardware
            self._is_simulated = True
            return True, "未检测到物理相机，已启动模拟对焦模式 (可使用测试画面)"

    def disconnect_camera(self) -> Tuple[bool, str]:
        if self.camera.is_opened:
            self.camera.stop_grabbing()
            self.camera.close()
        self._is_simulated = False
        return True, "相机已断开"

    def set_exposure(self, exposure_us: float) -> bool:
        return self.camera.set_exposure(exposure_us)

    def set_gain(self, gain_db: float) -> bool:
        return self.camera.set_gain(gain_db)

    def set_roi(self, x: int, y: int, w: int, h: int):
        if w > 10 and h > 10:
            self.roi = (int(x), int(y), int(w), int(h))
            self.tracker.reset_peak()
        else:
            self.roi = None

    def clear_roi(self):
        self.roi = None
        self.tracker.reset_peak()

    def reset_peak(self):
        self.tracker.reset_peak()

    def grab_single_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Grabs a frame from physical camera or generates a simulated target frame."""
        if self._is_simulated:
            return True, self._generate_simulated_frame()

        ok, frame = self.camera.grab_frame(timeout_ms=500)
        return ok, frame

    def process_frame(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, float, FocusState, float, float, bool]:
        """
        Processes a single frame:
        1. Computes sharpness score on ROI.
        2. Updates tracker state.
        3. Optionally overlays focus peaking highlight.
        Returns:
            (display_frame_bgr, score, state, ratio, peak, just_entered_in_focus)
        """
        now = time.monotonic()
        score = compute_focus_score(frame, self.metric_type, self.roi)
        prev_state = self.tracker.state
        state, ratio, peak = self.tracker.update(score, now)

        # Detect transition into IN_FOCUS
        just_entered_focus = (state == FocusState.IN_FOCUS and prev_state != FocusState.IN_FOCUS)
        if just_entered_focus and self.sound_enabled:
            self._play_beep()

        # Prepare display frame (BGR)
        if len(frame.shape) == 2:
            display_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            display_bgr = frame.copy()

        # Peaking overlay
        if self.peaking_enabled and state in (FocusState.IN_FOCUS, FocusState.APPROACHING):
            mask = compute_focus_peaking_mask(frame, self.peaking_threshold, self.roi)
            # Overlay bright green on edges
            display_bgr[mask > 0] = [0, 255, 60]

        return display_bgr, score, state, ratio, peak, just_entered_focus

    def _play_beep(self):
        """Asynchronously triggers sound alert on Windows."""
        try:
            import winsound
            winsound.MessageBeep(winsound.MB_ICONASTERISK)
        except Exception:
            pass

    def _generate_simulated_frame(self) -> np.ndarray:
        """Generates a simulated test chart with synthetic blur for offline evaluation."""
        self._sim_z_step += 0.05
        # Sine wave simulating moving in and out of depth of field
        blur_amount = int(abs(np.sin(self._sim_z_step)) * 14) * 2 + 1
        h, w = 900, 1200
        img = np.zeros((h, w), dtype=np.uint8)

        # Draw checkerboard pattern
        cell_size = 50
        for i in range(0, h, cell_size):
            for j in range(0, w, cell_size):
                if ((i // cell_size) + (j // cell_size)) % 2 == 0:
                    img[i : i + cell_size, j : j + cell_size] = 220
                else:
                    img[i : i + cell_size, j : j + cell_size] = 40

        # Draw high-frequency Siemens-like concentric rings in center
        center = (w // 2, h // 2)
        for r in range(10, 220, 15):
            cv2.circle(img, center, r, 255 if (r // 15) % 2 == 0 else 20, 2)

        # Text indicating simulation
        cv2.putText(
            img,
            "SIMULATED TARGET (Auto-focus test)",
            (40, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            255,
            2,
        )

        if blur_amount > 1:
            img = cv2.GaussianBlur(img, (blur_amount, blur_amount), 0)

        return img

