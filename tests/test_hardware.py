"""
Unit tests for hardware drivers and mock fallbacks.
"""

import os
import sys
import unittest
import ctypes
import cv2
import importlib
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dlp_common.hardware.daheng_camera import DahengCameraDriver
from dlp_common.hardware.dlp_controller import DLP4710Controller
from dlp_common.calibration.corner_detector import CornerDetector
from dlp_common.fringe.phase_unwrapping import PhaseProcessor
from dlp_common.fringe.generator import FringeGenerator
from DLP4710_MonoSL_Calibration.core.workers import CapturePoseWorker, DecodePhasesWorker

CaptureMeasurementWorker = importlib.import_module(
    "SL-3D-Measurement-DLP4710-SingleCamera.core.workers"
).CaptureMeasurementWorker


class TestHardware(unittest.TestCase):
    def test_dlp_controller_init(self):
        controller = DLP4710Controller()
        self.assertIsNotNone(controller)

        # Unit tests must not call the physical USB/I2C DLL. Hardware probing is
        # covered manually on the measurement workstation.
        mock_controller = DLP4710Controller(dll_path="")
        self.assertTrue(mock_controller.is_mock)
        self.assertTrue(mock_controller.connect())
        self.assertTrue(mock_controller.is_connected)
        mode_ok, _ = mock_controller.prepare_pattern_mode()
        self.assertTrue(mode_ok)
        self.assertTrue(mock_controller.pattern_ready)
        self.assertTrue(mock_controller.start_work(0))
        self.assertTrue(mock_controller.stop_work())
        self.assertTrue(mock_controller.is_connected, "stop_work must not disconnect the DLP")
        self.assertFalse(mock_controller.is_projecting)
        self.assertTrue(mock_controller.start_work(0), "DLP should be able to restart projection after stop_work")
        self.assertTrue(mock_controller.disconnect())
        self.assertFalse(mock_controller.is_connected)
        self.assertFalse(mock_controller.pattern_ready)

    def test_dlp_start_work_passes_repeat_count_by_reference(self):
        class FakeDll:
            repeat_value = None

            def startWork(self, repeat_ptr):
                self.repeat_value = ctypes.cast(
                    repeat_ptr,
                    ctypes.POINTER(ctypes.c_uint8),
                ).contents.value

        controller = DLP4710Controller(dll_path="")
        controller._is_mock = False
        controller._dll = FakeDll()
        controller._is_connected = True
        controller._pattern_ready = True

        self.assertTrue(controller.start_work(7))
        self.assertEqual(controller._dll.repeat_value, 7)

    def test_camera_driver_init(self):
        camera = DahengCameraDriver(width=1000, height=800)
        self.assertEqual(camera.resolution, (1000, 800))
        self.assertFalse(camera.is_opened)
        self.assertNotIn("DirectShow", camera.last_error)

    def test_capture_pose_worker_with_fake_galaxy_hardware(self):
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

        generator = FringeGenerator(640, 480, periods=[70.0, 64.0, 59.0], step_num=4)
        pattern_frames = (
            generator.generate_directional_fringes(0)
            + generator.generate_directional_fringes(1)
        )

        class FakeGalaxyCamera:
            def __init__(self):
                self.is_opened = True
                self.is_grabbing = True
                self.last_error = ""
                self._trigger_mode = False
                self._exposure_us = 0.0
                self._frames = [board] + pattern_frames

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

            def grab_triggered_sequence(self, count, frame_timeout_ms=1000, overall_timeout_ms=30000):
                frames = self._frames[:count]
                self._frames = self._frames[count:]
                return len(frames) == count, frames

        class FakeDlp:
            last_error = ""

            def start_work(self, repeat_num=0):
                return True

            def stop_work(self):
                return True

        camera = FakeGalaxyCamera()
        dlp = FakeDlp()

        results = []
        worker = CapturePoseWorker(
            camera=camera,
            dlp=dlp,
            detector=CornerDetector(board_cols=11, board_rows=8),
            board_exposure_us=10000.0,
            pattern_exposure_us=20000.0,
        )
        worker.completed.connect(lambda ok, pose, msg: results.append((ok, pose, msg)))
        worker.run()

        self.assertEqual(len(results), 1)
        ok, pose, _ = results[0]
        self.assertTrue(ok)
        self.assertIsNotNone(pose)
        self.assertEqual(len(pose.horizontal_frames), 12)
        self.assertEqual(len(pose.vertical_frames), 12)
        self.assertTrue(camera.is_grabbing)
        self.assertFalse(camera._trigger_mode)
        self.assertEqual(camera._exposure_us, 10000.0)

        decode_results = []
        decode_worker = DecodePhasesWorker(
            poses=[pose],
            processor=PhaseProcessor(periods=[70.0, 64.0, 59.0], step_num=4),
            projector_size=(1920, 1080),
            roi_margin=32,
        )
        decode_worker.finished.connect(
            lambda decode_ok, message: decode_results.append((decode_ok, message))
        )
        decode_worker.run()
        self.assertEqual(len(decode_results), 1)
        self.assertIsNotNone(pose.horizontal_phase)
        self.assertIsNotNone(pose.vertical_phase)

    def test_measurement_capture_worker_collects_12_hardware_frames(self):
        frames = [np.full((48, 64), i, dtype=np.uint8) for i in range(12)]

        class FakeGalaxyCamera:
            is_opened = True
            is_grabbing = False
            last_error = ""
            trigger_source = ""
            exposure = 0.0
            gain = 0.0

            def __init__(self):
                self.exposure_history = []

            def stop_grabbing(self):
                self.is_grabbing = False
                return True

            def start_grabbing(self):
                self.is_grabbing = True
                return True

            def set_trigger_mode(self, enabled, source="Software"):
                self.trigger_source = source if enabled else "Off"
                return True

            def set_exposure(self, value):
                self.exposure = value
                self.exposure_history.append(value)
                return True

            def set_gain(self, value):
                self.gain = value
                return True

            def grab_frame(self, timeout_ms=1500):
                return True, frames[0]

            def grab_triggered_sequence(self, count, frame_timeout_ms=1000, overall_timeout_ms=15000):
                return True, frames[:count]

        class FakeDlp:
            last_error = ""

            def start_work(self, repeat_num=0):
                return True

            def stop_work(self):
                return True

        camera = FakeGalaxyCamera()
        results = []
        worker = CaptureMeasurementWorker(camera, FakeDlp(), 12000.0, 2.5)
        worker.completed.connect(lambda ok, images, msg: results.append((ok, images, msg)))
        worker.run()

        self.assertEqual(len(results), 1)
        self.assertTrue(results[0][0])
        self.assertEqual(len(results[0][1]), 12)
        self.assertIsNotNone(worker.gray_frame)
        self.assertEqual(camera.exposure_history, [150000.0, 12000.0])
        self.assertEqual(camera.exposure, 12000.0)
        self.assertEqual(camera.gain, 2.5)
        self.assertEqual(camera.trigger_source, "Off")


if __name__ == "__main__":
    unittest.main()
