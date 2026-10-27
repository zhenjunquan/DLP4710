import os
import sys
import tempfile
import unittest

import cv2
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

    def test_batch_import_multiple_poses(self):
        window = CalibrationMainWindow()
        try:
            with tempfile.TemporaryDirectory() as root:
                p1 = os.path.join(root, "Pose_01")
                p2 = os.path.join(root, "Pose_02")
                os.makedirs(p1)
                os.makedirs(p2)
                dummy_img = np.zeros((24, 32), dtype=np.uint8)
                cv2.imwrite(os.path.join(p1, "board.bmp"), dummy_img)
                cv2.imwrite(os.path.join(p2, "board.bmp"), dummy_img)

                # Test find_pose_folders from parent root
                found = window.controller.find_pose_folders(root)
                self.assertEqual(len(found), 2)
                self.assertEqual(os.path.basename(found[0]), "Pose_01")
                self.assertEqual(os.path.basename(found[1]), "Pose_02")

                # Test batch import
                count, poses, msgs = window.controller.add_poses_from_folders([p1, p2])
                self.assertEqual(count, 2)
                self.assertEqual(len(window.controller.poses), 2)
                window._refresh_pose_table()
                self.assertEqual(window.table_poses.rowCount(), 2)

                # Re-importing should skip duplicates
                count_dup, _, _ = window.controller.add_poses_from_folders([p1])
                self.assertEqual(count_dup, 0)
                self.assertEqual(len(window.controller.poses), 2)
        finally:
            window.close()

    def test_spinbox_up_and_down_buttons_clickable(self):
        from PyQt6.QtCore import QPointF, Qt
        from PyQt6.QtGui import QMouseEvent
        from PyQt6.QtWidgets import QStyle, QStyleOptionSpinBox

        window = CalibrationMainWindow()
        try:
            window.show()
            self.app.processEvents()

            spin = window.spin_cols
            initial_val = spin.value()

            opt = QStyleOptionSpinBox()
            spin.initStyleOption(opt)
            up_rect = spin.style().subControlRect(
                QStyle.ComplexControl.CC_SpinBox, opt, QStyle.SubControl.SC_SpinBoxUp, spin
            )
            down_rect = spin.style().subControlRect(
                QStyle.ComplexControl.CC_SpinBox, opt, QStyle.SubControl.SC_SpinBoxDown, spin
            )

            # Ensure child QLineEdit does not intercept the click on up_button
            self.assertIsNone(spin.childAt(up_rect.center()))
            self.assertIsNone(spin.childAt(down_rect.center()))

            # Click Up (+) button
            press = QMouseEvent(QMouseEvent.Type.MouseButtonPress, QPointF(up_rect.center()), Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
            release = QMouseEvent(QMouseEvent.Type.MouseButtonRelease, QPointF(up_rect.center()), Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
            self.app.sendEvent(spin, press)
            self.app.sendEvent(spin, release)
            self.assertEqual(spin.value(), initial_val + 1)

            # Click Down (-) button
            press = QMouseEvent(QMouseEvent.Type.MouseButtonPress, QPointF(down_rect.center()), Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
            release = QMouseEvent(QMouseEvent.Type.MouseButtonRelease, QPointF(down_rect.center()), Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
            self.app.sendEvent(spin, press)
            self.app.sendEvent(spin, release)
            self.assertEqual(spin.value(), initial_val)
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
