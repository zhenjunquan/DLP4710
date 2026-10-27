import os
import sys
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PyQt6.QtWidgets import QApplication

from DLP4710_MonoSL_Calibration.ui.main_window import CalibrationMainWindow
from dlp_common.calibration.models import CalibrationPose


class TestCalibrationUi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_phase_view_survives_pose_table_refresh(self):
        window = CalibrationMainWindow()
        try:
            pose = CalibrationPose(
                name="Pose_01",
                board_image=np.zeros((24, 32), dtype=np.uint8),
                camera_image_size=(32, 24),
                horizontal_phase=np.arange(24 * 32, dtype=np.float64).reshape(24, 32),
                phase_valid=True,
            )
            window.controller.poses = [pose]
            window._refresh_pose_table()
            window.table_poses.selectRow(0)
            window._on_pose_selection_changed()
            self.assertIsNotNone(window.view_phase._current_pixmap)

            window._refresh_pose_table()
            window._on_pose_selection_changed()
            self.assertEqual(window.table_poses.currentRow(), 0)
            self.assertIsNotNone(window.view_phase._current_pixmap)
            self.assertFalse(window.view_phase._current_pixmap.isNull())
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
