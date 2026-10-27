import os
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from DLP4710_MonoSL_Calibration.core.controller import CalibrationController
from dlp_common.calibration.models import CalibrationPose
from dlp_common.calibration.corner_detector import CornerDetector
from dlp_common.fringe.phase_unwrapping import PhaseProcessor
from DLP4710_MonoSL_Calibration.core.workers import CapturePoseWorker


class TestCalibrationIncremental(unittest.TestCase):
    @staticmethod
    def _make_frames():
        height, width = 16, 20
        x = np.arange(width, dtype=np.float64)[None, :]
        frames = []
        for cycles in (70.0, 64.0, 59.0):
            for step in range(4):
                phase = 2.0 * np.pi * cycles * x / width + step * np.pi / 2.0
                image = 127.5 + 100.0 * np.cos(phase)
                frames.append(np.repeat(image, height, axis=0).astype(np.uint8))
        return frames

    def test_incremental_session_creates_and_rejects_existing(self):
        controller = CalibrationController()
        with tempfile.TemporaryDirectory() as parent:
            output = os.path.join(parent, "session")
            ok, message = controller.start_incremental_session(output)
            self.assertTrue(ok, message)
            self.assertEqual(controller.session_output_folder, output)
            self.assertTrue(os.path.isdir(os.path.join(output, "poses")))
            ok_again, _ = controller.start_incremental_session(output)
            self.assertFalse(ok_again)

    def test_prepare_pose_storage(self):
        controller = CalibrationController()
        with tempfile.TemporaryDirectory() as parent:
            output = os.path.join(parent, "session")
            controller.start_incremental_session(output)
            ok, pose_dir, message = controller.prepare_pose_storage(1, "Pose_01")
            self.assertTrue(ok, message)
            self.assertTrue(os.path.isdir(os.path.join(pose_dir, "horizontal")))
            self.assertTrue(os.path.isdir(os.path.join(pose_dir, "vertical")))

    def test_persist_decoded_pose_releases_large_arrays(self):
        controller = CalibrationController()
        horizontal = self._make_frames()
        vertical = [frame.T.copy() for frame in horizontal]
        h_wrapped, h_modulation = controller.phase_processor.solve_wrapped_phases(horizontal)
        v_wrapped, v_modulation = controller.phase_processor.solve_wrapped_phases(vertical)
        h_unwrapped = controller.phase_processor.unwrap_wrapped_phases(h_wrapped)
        v_unwrapped = controller.phase_processor.unwrap_wrapped_phases(v_wrapped)

        with tempfile.TemporaryDirectory() as parent:
            output = os.path.join(parent, "session")
            controller.start_incremental_session(output)
            _, pose_dir, _ = controller.prepare_pose_storage(1, "Pose_01")
            pose = CalibrationPose(
                name="Pose_01",
                folder_path=pose_dir,
                camera_image_size=(20, 16),
                board_image=np.zeros((16, 20), dtype=np.uint8),
                horizontal_frames=horizontal,
                vertical_frames=vertical,
                horizontal_phase=h_unwrapped,
                vertical_phase=v_unwrapped,
                phase_roi=None,
                camera_corners=np.zeros((1, 2), dtype=np.float32),
                projector_corners=np.zeros((1, 2), dtype=np.float32),
                corner_valid=True,
                phase_valid=True,
            )
            cv2.imwrite(os.path.join(pose_dir, "board.bmp"), pose.board_image)
            controller.persist_decoded_pose(
                pose, h_wrapped, h_modulation, v_wrapped, v_modulation
            )

            self.assertTrue(os.path.isfile(os.path.join(pose_dir, "camera_corners.npy")))
            self.assertTrue(os.path.isfile(os.path.join(pose_dir, "projector_corners.npy")))
            self.assertTrue(
                os.path.isfile(
                    os.path.join(pose_dir, "horizontal", "unwrapped_phase.npy")
                )
            )
            self.assertIsNone(pose.board_image)
            self.assertEqual(pose.horizontal_frames, [])
            self.assertEqual(pose.vertical_frames, [])
            self.assertIsNone(pose.horizontal_phase)
            self.assertIsNone(pose.vertical_phase)

    def test_disk_phase_matches_memory(self):
        processor = PhaseProcessor([70.0, 64.0, 59.0], step_num=4)
        frames = self._make_frames()
        with tempfile.TemporaryDirectory() as folder:
            paths = []
            for index, frame in enumerate(frames, start=1):
                path = os.path.join(folder, f"{index:02d}.bmp")
                cv2.imwrite(path, frame)
                paths.append(path)

            memory_wrapped, memory_modulation = processor.solve_wrapped_phases(frames)
            disk_wrapped, disk_modulation = processor.solve_wrapped_phases_from_files(paths)

            self.assertEqual(len(disk_wrapped), 3)
            for memory, disk in zip(memory_wrapped, disk_wrapped):
                np.testing.assert_allclose(disk, memory, atol=1e-6)
            np.testing.assert_allclose(disk_modulation, memory_modulation, atol=1e-6)


    def test_disk_backed_capture_worker_writes_frames_incrementally(self):
        board = np.full((480, 640), 127, dtype=np.uint8)
        square = 36
        start_x, start_y = 104, 78
        for row in range(9):
            for col in range(12):
                color = 245 if (row + col) % 2 == 0 else 10
                cv2.rectangle(
                    board,
                    (start_x + col * square, start_y + row * square),
                    (start_x + (col + 1) * square, start_y + (row + 1) * square),
                    color,
                    -1,
                )

        patterns = [
            np.full((480, 640), 20 + (i % 200), dtype=np.uint8) for i in range(24)
        ]

        class FakeGalaxyCamera:
            def __init__(self):
                self.is_opened = True
                self.is_grabbing = True
                self.last_error = ""
                self._trigger_mode = False
                self._exposure_us = 0.0
                self._frames = [board] + patterns

            def stop_grabbing(self):
                self.is_grabbing = False
                return True

            def start_grabbing(self):
                self.is_grabbing = True
                return True

            def set_trigger_mode(self, enabled, source="Software"):
                self._trigger_mode = enabled
                return True

            def set_exposure(self, exposure_us):
                self._exposure_us = exposure_us
                return True

            def grab_frame(self, timeout_ms=1500):
                return (True, self._frames.pop(0)) if self._frames else (False, None)

            def grab_triggered_sequence(
                self,
                count,
                frame_timeout_ms=1000,
                overall_timeout_ms=30000,
                frame_callback=None,
                retain_frames=True,
            ):
                frames = self._frames[:count]
                self._frames = self._frames[count:]
                for index, frame in enumerate(frames):
                    if frame_callback is not None:
                        frame_callback(index, frame)
                if retain_frames:
                    return len(frames) == count, frames
                return len(frames) == count, []

        class FakeDlp:
            last_error = ""

            def start_work(self, repeat_num=0):
                return True

        with tempfile.TemporaryDirectory() as pose_folder:
            os.makedirs(os.path.join(pose_folder, "horizontal"))
            os.makedirs(os.path.join(pose_folder, "vertical"))
            camera = FakeGalaxyCamera()
            results = []
            worker = CapturePoseWorker(
                camera=camera,
                dlp=FakeDlp(),
                detector=CornerDetector(board_cols=11, board_rows=8),
                board_exposure_us=10000.0,
                pattern_exposure_us=20000.0,
                pose_folder=pose_folder,
            )
            worker.completed.connect(lambda ok, pose, msg: results.append((ok, pose, msg)))
            worker.run()

            self.assertEqual(len(results), 1)
            ok, pose, _ = results[0]
            self.assertTrue(ok)
            self.assertIsNotNone(pose)
            self.assertTrue(os.path.isfile(os.path.join(pose_folder, "board.bmp")))
            self.assertTrue(
                os.path.isfile(os.path.join(pose_folder, "horizontal", "H_Fringe_01.bmp"))
            )
            self.assertTrue(
                os.path.isfile(os.path.join(pose_folder, "vertical", "V_Fringe_12.bmp"))
            )
            self.assertEqual(pose.horizontal_frames, [])
            self.assertEqual(pose.vertical_frames, [])
if __name__ == "__main__":
    unittest.main()
