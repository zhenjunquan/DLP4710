import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from DLP4710_MonoSL_Calibration.core.controller import CalibrationController
from dlp_common.calibration.models import CalibrationPose


class TestCalibrationExport(unittest.TestCase):
    @staticmethod
    def _make_frames():
        frames = []
        height, width = 16, 20
        x = np.arange(width, dtype=np.float64)[None, :]
        for cycles in (70.0, 64.0, 59.0):
            for step in range(4):
                phase = 2.0 * np.pi * cycles * x / width + step * np.pi / 2.0
                image = 127.5 + 100.0 * np.cos(phase)
                frames.append(np.repeat(image, height, axis=0).astype(np.uint8))
        return frames

    def test_complete_session_saves_raw_and_phase_products(self):
        controller = CalibrationController()
        horizontal = self._make_frames()
        vertical = [frame.T.copy() for frame in horizontal]
        h_phase, _ = controller.phase_processor.unwrap_phase(horizontal)
        v_phase, _ = controller.phase_processor.unwrap_phase(vertical)
        pose = CalibrationPose(
            name="Pose_01",
            camera_image_size=(20, 16),
            board_image=np.zeros((16, 20), dtype=np.uint8),
            horizontal_frames=horizontal,
            vertical_frames=vertical,
            horizontal_phase=h_phase,
            vertical_phase=v_phase,
            phase_roi=None,
        )
        controller.poses.append(pose)

        with tempfile.TemporaryDirectory() as parent:
            output = os.path.join(parent, "session")
            ok, message = controller.save_complete_session(output)
            self.assertTrue(ok, message)
            pose_dir = os.path.join(output, "poses", "01_Pose_01")
            self.assertTrue(os.path.isfile(os.path.join(pose_dir, "board.bmp")))
            self.assertTrue(os.path.isfile(os.path.join(pose_dir, "horizontal", "H_Fringe_01.bmp")))
            self.assertTrue(os.path.isfile(os.path.join(pose_dir, "vertical", "V_Fringe_12.bmp")))
            phase_path = os.path.join(
                pose_dir, "horizontal", "wrapped_frequency_01.npy"
            )
            self.assertTrue(os.path.isfile(phase_path))
            self.assertEqual(np.load(phase_path).shape, (16, 20))
            self.assertTrue(os.path.isfile(os.path.join(output, "calibration_session.txt")))

            ok_again, _ = controller.save_complete_session(output)
            self.assertFalse(ok_again)


if __name__ == "__main__":
    unittest.main()
