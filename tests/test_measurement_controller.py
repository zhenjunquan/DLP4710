"""Tests for measurement controller device lifecycle."""

import importlib
import os
import sys
import unittest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
MeasurementController = importlib.import_module(
    "SL-3D-Measurement-DLP4710-SingleCamera.core.controller"
).MeasurementController


class TestMeasurementController(unittest.TestCase):
    def test_disconnect_devices_closes_camera_and_dlp(self):
        class FakeCamera:
            is_opened = True
            closed = False

            def close(self):
                self.closed = True
                self.is_opened = False

        class FakeDlp:
            is_connected = True
            last_error = ""
            disconnected = False

            def disconnect(self):
                self.disconnected = True
                self.is_connected = False
                return True

        controller = MeasurementController()
        controller.camera = FakeCamera()
        controller.dlp = FakeDlp()

        ok, message = controller.disconnect_devices()

        self.assertTrue(ok, message)
        self.assertTrue(controller.camera.closed)
        self.assertTrue(controller.dlp.disconnected)
        self.assertIn("已断开", message)


if __name__ == "__main__":
    unittest.main()
