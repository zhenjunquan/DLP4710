"""
Background worker thread for continuous camera capture and real-time focus processing.
"""

import time
from typing import Optional
import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal

from DLP4710_Focus_Assistant.core.focus_controller import FocusController


class FocusStreamWorker(QThread):
    """
    Continuous worker thread capturing camera frames, calculating focus sharpness,
    and pushing processed results to the UI thread.
    """

    frame_ready = pyqtSignal(
        np.ndarray,  # Processed display frame (BGR)
        float,       # Current focus score
        str,         # Focus state description
        float,       # Ratio relative to peak (0 ~ 1.0)
        float,       # Peak score
        list,        # Score history for plotting
    )
    fps_updated = pyqtSignal(float)
    status_message = pyqtSignal(str)

    def __init__(self, controller: FocusController):
        super().__init__()
        self.controller = controller
        self._running = False
        self._interval_s = 0.033  # Target ~30 FPS

    def run(self):
        self._running = True
        frame_count = 0
        fps_start = time.monotonic()

        while self._running:
            loop_start = time.monotonic()
            ok, frame = self.controller.grab_single_frame()

            if ok and frame is not None:
                display_bgr, score, state, ratio, peak, _ = self.controller.process_frame(frame)
                history = self.controller.tracker.get_history()

                self.frame_ready.emit(
                    display_bgr,
                    score,
                    state.value,
                    ratio,
                    peak,
                    history,
                )

                frame_count += 1
                now = time.monotonic()
                if now - fps_start >= 1.0:
                    fps = frame_count / (now - fps_start)
                    self.fps_updated.emit(fps)
                    frame_count = 0
                    fps_start = now

            # Throttle to desired FPS
            elapsed = time.monotonic() - loop_start
            sleep_time = max(0.005, self._interval_s - elapsed)
            time.sleep(sleep_time)

    def stop(self):
        self._running = False
        self.wait(1000)

