"""
Unit tests for focus evaluation algorithms, focus tracker, and focus UI.
"""

import os
import sys
import unittest
import cv2
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PyQt6.QtWidgets import QApplication

from dlp_common.focus.focus_metric import (
    FocusMetricType,
    compute_focus_peaking_mask,
    compute_focus_score,
    extract_roi,
)
from dlp_common.focus.focus_tracker import FocusState, FocusTracker
from DLP4710_Focus_Assistant.core.focus_controller import FocusController
from DLP4710_Focus_Assistant.ui.main_window import FocusMainWindow


class TestFocusModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        # Create a synthetic sharp image (step edges and checkerboard)
        self.sharp_img = np.zeros((300, 300), dtype=np.uint8)
        for i in range(0, 300, 30):
            for j in range(0, 300, 30):
                if ((i // 30) + (j // 30)) % 2 == 0:
                    self.sharp_img[i : i + 30, j : j + 30] = 255

        # Create a blurred version (out of focus)
        self.blurred_img = cv2.GaussianBlur(self.sharp_img, (15, 15), 0)

    def test_focus_metric_monotonicity(self):
        """Verify that all focus operators yield significantly higher score on sharp vs blurred image."""
        for metric in FocusMetricType:
            score_sharp = compute_focus_score(self.sharp_img, metric=metric)
            score_blurred = compute_focus_score(self.blurred_img, metric=metric)
            self.assertGreater(
                score_sharp,
                score_blurred * 3.0,
                f"Metric {metric.value} failed to distinguish sharp from blurred image",
            )

    def test_focus_roi_extraction(self):
        """Verify ROI extraction and score calculation within ROI."""
        roi = (50, 50, 100, 100)
        cropped = extract_roi(self.sharp_img, roi)
        self.assertEqual(cropped.shape, (100, 100))

        score_roi = compute_focus_score(self.sharp_img, roi=roi)
        self.assertGreater(score_roi, 0.0)

    def test_focus_peaking_mask(self):
        """Verify peaking mask highlights sharp edges and suppresses blurred ones."""
        mask_sharp = compute_focus_peaking_mask(self.sharp_img, highpass_threshold=60.0)
        mask_blurred = compute_focus_peaking_mask(self.blurred_img, highpass_threshold=60.0)
        self.assertGreater(np.count_nonzero(mask_sharp), 0)
        self.assertEqual(np.count_nonzero(mask_blurred), 0)

    def test_focus_tracker_state_transitions(self):
        """Verify full lifecycle: Out-of-focus -> Approaching -> In-Focus -> Overshot -> Reset."""
        tracker = FocusTracker(in_focus_ratio=0.92, overshot_threshold=0.85, min_peak_valid=5.0)

        # 1. Starting at low scores
        state, ratio, peak = tracker.update(10.0, 1.0)
        self.assertEqual(state, FocusState.OUT_OF_FOCUS)

        # 2. Approaching focal plane (sharpness rising steadily)
        tracker.update(30.0, 2.0)
        tracker.update(60.0, 3.0)
        state, ratio, peak = tracker.update(85.0, 4.0)
        self.assertEqual(state, FocusState.APPROACHING)

        # 3. Peak focal plane (best sharpness reached, plateauing)
        tracker.update(100.0, 5.0)
        # Next frame near peak (trend plateaus)
        state, ratio, peak = tracker.update(98.0, 6.0)
        self.assertEqual(state, FocusState.IN_FOCUS)
        self.assertEqual(peak, 100.0)
        self.assertAlmostEqual(ratio, 0.98, places=2)

        # 4. Small movement within depth of field (e.g. 94% of peak >= 92%)
        state, ratio, peak = tracker.update(94.0, 7.0)
        self.assertEqual(state, FocusState.IN_FOCUS)

        # 5. Overshot (moving past focal plane, sharpness drops below 85%)
        tracker.update(80.0, 8.0)
        state, ratio, peak = tracker.update(70.0, 9.0)
        self.assertEqual(state, FocusState.OVERSHOT)

        # 6. Re-entering focus when moving back
        state, ratio, peak = tracker.update(95.0, 10.0)
        self.assertEqual(state, FocusState.IN_FOCUS)

        # 7. Reset peak
        tracker.reset_peak()
        self.assertEqual(tracker.peak_score, 0.0)
        self.assertEqual(tracker.state, FocusState.IDLE)

    def test_focus_main_window_startup(self):
        """Verify UI window initializes and processes frames cleanly."""
        controller = FocusController()
        window = FocusMainWindow(controller=controller)
        try:
            window.show()
            self.app.processEvents()

            # Process a frame
            frame = controller._generate_simulated_frame()
            display_bgr, score, state, ratio, peak, _ = controller.process_frame(frame)
            self.assertIsNotNone(display_bgr)
            self.assertGreater(score, 0.0)

            # Test ROI selection signal
            window.canvas.roi_selected.emit(10, 10, 80, 80)
            self.assertIsNotNone(controller.roi)
            self.assertEqual(controller.roi, (10, 10, 80, 80))

            # Test ROI clear signal
            window.canvas.roi_cleared.emit()
            self.assertIsNone(controller.roi)
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
