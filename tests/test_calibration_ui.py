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

                # Put old results in p1 to ensure import does NOT load them
                np.save(os.path.join(p1, "camera_corners.npy"), np.ones((4, 2), dtype=np.float32))
                np.save(os.path.join(p1, "projector_corners.npy"), np.ones((4, 2), dtype=np.float32))

                # Test find_pose_folders from parent root
                found = window.controller.find_pose_folders(root)
                self.assertEqual(len(found), 2)
                self.assertEqual(os.path.basename(found[0]), "Pose_01")
                self.assertEqual(os.path.basename(found[1]), "Pose_02")

                # Test batch import
                count, poses, msgs = window.controller.add_poses_from_folders([p1, p2])
                self.assertEqual(count, 2)
                self.assertEqual(len(window.controller.poses), 2)

                # Ensure imported poses are clean (no stale corners or phases)
                p1_imported = window.controller.poses[0]
                self.assertFalse(p1_imported.corner_valid)
                self.assertFalse(p1_imported.phase_valid)
                self.assertIsNone(p1_imported.camera_corners)
                self.assertIsNone(p1_imported.projector_corners)
                self.assertIsNone(p1_imported.horizontal_phase)

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

    def test_multithreaded_decode_phases_worker(self):
        from DLP4710_MonoSL_Calibration.core.workers import DecodePhasesWorker
        from dlp_common.fringe.phase_unwrapping import PhaseProcessor

        # Create two minimal poses with in-memory frames
        processor = PhaseProcessor()
        poses = []
        for idx in range(2):
            h_frames = [np.full((32, 32), 128 + 50 * np.cos(i), dtype=np.uint8) for i in range(12)]
            v_frames = [np.full((32, 32), 128 + 50 * np.cos(i), dtype=np.uint8) for i in range(12)]
            corners = np.array([[8.0, 8.0], [24.0, 8.0], [8.0, 24.0], [24.0, 24.0]], dtype=np.float32)
            pose = CalibrationPose(
                name=f"Pose_{idx:02d}",
                camera_corners=corners,
                camera_image_size=(32, 32),
                horizontal_frames=h_frames,
                vertical_frames=v_frames,
                corner_valid=True,
            )
            poses.append(pose)

        worker = DecodePhasesWorker(
            poses=poses,
            processor=processor,
            projector_size=(1920, 1080),
            roi_margin=4,
            max_workers=2,
        )

        progress_calls = []
        worker.progress.connect(lambda cur, tot, msg: progress_calls.append((cur, tot, msg)))
        worker.run()

        self.assertEqual(len(progress_calls), 3)  # initial 0, then 1, then 2
        for pose in poses:
            self.assertTrue(pose.phase_valid)
            self.assertIsNotNone(pose.horizontal_phase)
            self.assertIsNotNone(pose.vertical_phase)
            self.assertIsNotNone(pose.projector_corners)
            self.assertEqual(len(pose.projector_corners), 4)


if __name__ == "__main__":
    unittest.main()
