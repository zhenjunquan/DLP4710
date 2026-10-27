"""
Industrial Daheng Galaxy camera driver using Daheng official Python SDK (gxipy).
Provides high-performance frame acquisition, parameter control (exposure, gain),
and software/hardware trigger management matching the C++ DahengCamera implementation.
"""

from datetime import datetime
import os
import sys
import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


def _setup_galaxy_environment():
    """
    Ensures Daheng Galaxy SDK environment variables and DLL paths are properly registered.
    """
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
        ) as key:
            i = 0
            while True:
                try:
                    name, val, _ = winreg.EnumValue(key, i)
                    if "GALAXY" in name or "GENICAM" in name:
                        os.environ[name] = val
                    i += 1
                except OSError:
                    break
    except Exception:
        pass

    # Fallback to standard installation paths if missing
    sdk_root = os.environ.get("GALAXY_SDK_DEVELOPMENT")
    if sdk_root:
        sdk_root = os.path.abspath(os.path.join(sdk_root, ".."))
    else:
        sdk_root = r"D:\ZhenSoft\GalaxySDK"

    os.environ.setdefault("GALAXY_GENICAM_ROOT", os.path.join(sdk_root, "GenICam"))
    os.environ.setdefault("GENICAM_GENTL64_PATH", os.path.join(sdk_root, "GenTL", "Win64"))
    os.environ.setdefault("GENICAM_GENTL32_PATH", os.path.join(sdk_root, "GenTL", "Win32"))

    dll_dirs = [
        os.path.join(sdk_root, "GenICam", "bin", "Win64_x64"),
        os.path.join(sdk_root, "APIDll", "Win64"),
        os.path.join(sdk_root, "GenTL", "Win64"),
    ]
    for d in dll_dirs:
        if os.path.isdir(d):
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(d)
                except Exception:
                    pass
            os.environ["PATH"] = d + ";" + os.environ.get("PATH", "")


# Initialize environment on module load
_setup_galaxy_environment()

# Attempt importing Daheng official Python SDK
_GXIPY_AVAILABLE = False
try:
    import gxipy as gx
    from gxipy.gxidef import DxValidBit, GxPixelFormatEntry, GxSwitchEntry, GxTriggerSourceEntry
    _GXIPY_AVAILABLE = True
except Exception as _e:
    _GXIPY_AVAILABLE = False


class DahengCameraDriver:
    """
    Daheng Galaxy industrial camera driver.
    Matches the C++ DahengCamera class API and behaviors.
    """

    def __init__(
        self,
        width: int = 5328,
        height: int = 4608,
    ):
        self.target_width = width
        self.target_height = height
        self.actual_width = width
        self.actual_height = height

        # gxipy device handles
        self._dev_manager: Optional[object] = None
        self._cam: Optional[object] = None
        self._remote_feature = None
        self._format_convert = None
        self._device_sn: Optional[str] = None
        self._is_gx_opened: bool = False

        # Fallback / DirectShow handle
        self._cap: Optional[cv2.VideoCapture] = None
        self._is_opened: bool = False
        self._is_grabbing: bool = False
        self._is_mock: bool = False

        self._exposure_us: float = 150000.0
        self._gain_db: float = 0.0
        self._trigger_mode: bool = False
        self._trigger_source: str = "Software"
        self._last_error: str = ""

        # Simulated frame generation
        self._sim_folder: Optional[str] = None
        self._sim_images: List[str] = []
        self._sim_index: int = 0
        self._sim_frame_count: int = 0

    @property
    def is_opened(self) -> bool:
        return self._is_opened or self._is_gx_opened

    @property
    def is_grabbing(self) -> bool:
        return self._is_grabbing

    @property
    def is_mock(self) -> bool:
        return self._is_mock

    @property
    def last_error(self) -> str:
        return self._last_error

    @property
    def resolution(self) -> Tuple[int, int]:
        return (self.actual_width, self.actual_height)

    def detect_cameras(self) -> List[str]:
        """Backward compatible helper returning list of camera display names."""
        devs = self.detect_camera_devices()
        return [d["name"] for d in devs]

    def detect_camera_devices(self) -> List[Dict]:
        """
        Scans for available Daheng industrial cameras via gxipy.
        Returns a list of dicts: [{'index': int, 'sn': str, 'name': str, 'type': str}]
        """
        devices = []

        # 1. Daheng Galaxy cameras via official gxipy SDK
        if _GXIPY_AVAILABLE:
            try:
                dm = gx.DeviceManager()
                count, dev_info_list = dm.update_all_device_list()
                for i, info in enumerate(dev_info_list):
                    sn = info.get("sn", f"SN_{i}")
                    model = info.get("model_name", "Daheng Galaxy Camera")
                    name = f"大恒工业相机: {model} (SN: {sn})"
                    devices.append({
                        "index": i,
                        "sn": sn,
                        "name": name,
                        "type": "daheng_gxipy",
                        "is_mock": False,
                    })
            except Exception as e:
                self._last_error = f"Daheng gxipy 枚举异常: {e}"

        # 2. DirectShow USB cameras (for physical USB webcams / secondary video inputs)
        for idx in range(2):
            try:
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                if cap.isOpened():
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()
                    devices.append({
                        "index": idx,
                        "sn": f"USB_{idx}",
                        "name": f"USB/DirectShow 视频设备 {idx} ({w}x{h})",
                        "type": "directshow",
                        "is_mock": False,
                    })
            except Exception:
                pass

        # 3. Virtual simulated industrial camera (always available for development/testing)
        devices.append({
            "index": -1,
            "sn": "MOCK_SIMULATOR",
            "name": f"虚拟仿真大恒相机 ({self.target_width}x{self.target_height} 标定模式)",
            "type": "mock",
            "is_mock": True,
        })

        return devices

    def open(self, device_identifier=0) -> bool:
        """
        Opens a camera by device index, serial number string, or mock selector (-1).
        """
        self.close()

        # Check if identifier is mock
        if device_identifier == -1 or device_identifier == "MOCK_SIMULATOR":
            return self._open_mock()

        # 1. Try Daheng gxipy SDK first
        if _GXIPY_AVAILABLE:
            try:
                dm = gx.DeviceManager()
                count, dev_info_list = dm.update_all_device_list()
                if count > 0:
                    if isinstance(device_identifier, str) and not device_identifier.isdigit():
                        cam = dm.open_device_by_sn(device_identifier)
                        self._device_sn = device_identifier
                    else:
                        idx = int(device_identifier) if str(device_identifier).isdigit() else 0
                        idx = max(0, min(idx, count - 1))
                        sn = dev_info_list[idx].get("sn")
                        cam = dm.open_device_by_sn(sn)
                        self._device_sn = sn

                    self._dev_manager = dm
                    self._cam = cam
                    self._remote_feature = cam.get_remote_device_feature_control()
                    self._format_convert = dm.create_image_format_convert()

                    # Configure camera parameters like C++ DahengCamera
                    self._configure_daheng_camera()
                    self._is_gx_opened = True
                    self._is_opened = True
                    self._is_mock = False
                    return True
            except Exception as e:
                self._last_error = f"打开大恒相机失败: {e}"

        # 2. Try DirectShow camera if requested
        if isinstance(device_identifier, int) and device_identifier >= 0:
            try:
                cap = cv2.VideoCapture(device_identifier, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(device_identifier)
                if cap.isOpened():
                    self._cap = cap
                    self.actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    self.actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    self._is_opened = True
                    self._is_mock = False
                    return True
            except Exception as e:
                self._last_error = f"打开DirectShow相机失败: {e}"

        # 3. Fallback to virtual simulated camera
        return self._open_mock()

    def _open_mock(self) -> bool:
        self._is_mock = True
        self._is_opened = True
        self.actual_width = self.target_width
        self.actual_height = self.target_height
        return True

    def _configure_daheng_camera(self):
        """
        Configures Daheng camera features: Mono8 format, continuous mode, exposure, gain.
        """
        if self._remote_feature is None:
            return

        try:
            # Set continuous mode
            if self._remote_feature.is_implemented("AcquisitionMode"):
                self._remote_feature.get_enum_feature("AcquisitionMode").set("Continuous")

            # Set TriggerMode Off for live preview
            if self._remote_feature.is_implemented("TriggerMode"):
                self._remote_feature.get_enum_feature("TriggerMode").set("Off")

            # Try setting Mono8 pixel format
            if self._remote_feature.is_implemented("PixelFormat"):
                try:
                    self._remote_feature.get_enum_feature("PixelFormat").set("Mono8")
                except Exception:
                    pass

            # Read camera resolution
            if self._remote_feature.is_implemented("Width"):
                self.actual_width = self._remote_feature.get_int_feature("Width").get()
            if self._remote_feature.is_implemented("Height"):
                self.actual_height = self._remote_feature.get_int_feature("Height").get()

            # Set initial exposure and gain
            self.set_exposure(self._exposure_us)
            self.set_gain(self._gain_db)
        except Exception as e:
            self._last_error = f"配置大恒相机参数提示: {e}"

    def close(self):
        """
        Stops acquisition, releases camera device, and resets state.
        """
        self.stop_grabbing()

        if self._cam is not None:
            try:
                self._cam.close_device()
            except Exception:
                pass
            self._cam = None
            self._remote_feature = None
            self._format_convert = None

        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

        self._is_gx_opened = False
        self._is_opened = False
        self._is_grabbing = False

    def start_grabbing(self) -> bool:
        if not self.is_opened:
            if not self.open(0):
                return False

        if self._is_grabbing:
            return True

        if self._cam is not None:
            try:
                self._cam.stream_on()
                self._is_grabbing = True
                return True
            except Exception as e:
                self._last_error = f"大恒相机开启流失败: {e}"
                return False

        self._is_grabbing = True
        return True

    def stop_grabbing(self) -> bool:
        if not self._is_grabbing:
            return True

        if self._cam is not None:
            try:
                self._cam.stream_off()
            except Exception:
                pass

        self._is_grabbing = False
        return True

    def set_exposure(self, exposure_us: float) -> bool:
        self._exposure_us = float(exposure_us)

        # 1. Daheng Galaxy camera
        if self._remote_feature is not None:
            try:
                if self._remote_feature.is_implemented("ExposureAuto"):
                    self._remote_feature.get_enum_feature("ExposureAuto").set("Off")
                if self._remote_feature.is_implemented("ExposureTime"):
                    feat = self._remote_feature.get_float_feature("ExposureTime")
                    val_range = feat.get_range()
                    clamped = max(val_range["min"], min(self._exposure_us, val_range["max"]))
                    feat.set(clamped)
                    return True
            except Exception as e:
                self._last_error = f"设置曝光失败: {e}"

        # 2. DirectShow camera
        if self._cap is not None and self._cap.isOpened():
            try:
                exp_val = -round(np.log2(max(1.0, 1000000.0 / self._exposure_us)))
                self._cap.set(cv2.CAP_PROP_EXPOSURE, exp_val)
            except Exception:
                pass

        return True

    def set_gain(self, gain_db: float) -> bool:
        self._gain_db = float(gain_db)

        # 1. Daheng Galaxy camera
        if self._remote_feature is not None:
            try:
                if self._remote_feature.is_implemented("GainAuto"):
                    self._remote_feature.get_enum_feature("GainAuto").set("Off")
                if self._remote_feature.is_implemented("Gain"):
                    feat = self._remote_feature.get_float_feature("Gain")
                    val_range = feat.get_range()
                    clamped = max(val_range["min"], min(self._gain_db, val_range["max"]))
                    feat.set(clamped)
                    return True
            except Exception as e:
                self._last_error = f"设置增益失败: {e}"

        # 2. DirectShow camera
        if self._cap is not None and self._cap.isOpened():
            try:
                self._cap.set(cv2.CAP_PROP_GAIN, float(gain_db))
            except Exception:
                pass

        return True

    def set_trigger_mode(self, enable: bool, source: str = "Software") -> bool:
        """
        Sets TriggerMode to On/Off, and selects TriggerSource ('Software', 'Line0', 'Line2').
        """
        self._trigger_mode = enable
        self._trigger_source = source if enable else "Software"
        if self._remote_feature is not None:
            try:
                if self._remote_feature.is_implemented("TriggerMode"):
                    self._remote_feature.get_enum_feature("TriggerMode").set("On" if enable else "Off")
                if enable and self._remote_feature.is_implemented("TriggerSource"):
                    self._remote_feature.get_enum_feature("TriggerSource").set(source)
                return True
            except Exception as e:
                self._last_error = f"设置触发模式失败: {e}"
                return False
        return True

    def trigger_software(self) -> bool:
        """
        Sends a single software trigger pulse to the camera.
        """
        if self._remote_feature is not None:
            try:
                if self._remote_feature.is_implemented("TriggerSoftware"):
                    self._remote_feature.get_command_feature("TriggerSoftware").send_command()
                    return True
            except Exception as e:
                self._last_error = f"软触发执行失败: {e}"
                return False
        return True

    def grab_frame(self, timeout_ms: int = 1500) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Grabs a single image frame (Grayscale uint8 numpy array).
        """
        if not self.is_opened:
            if not self.open(0):
                return False, None

        if not self._is_grabbing:
            self.start_grabbing()

        # 1. Physical Daheng Galaxy Camera via gxipy
        if self._cam is not None and self._is_gx_opened:
            try:
                if self._trigger_mode and self._trigger_source == "Software":
                    self.trigger_software()

                raw_image = self._cam.data_stream[0].get_image(timeout=int(timeout_ms))
                if raw_image is None or raw_image.get_status() != 0:
                    return False, None

                # Convert image to numpy array
                fmt = raw_image.get_pixel_format()
                if fmt == GxPixelFormatEntry.MONO8:
                    numpy_img = raw_image.get_numpy_array()
                else:
                    # Convert to RGB then Grayscale
                    rgb_image = raw_image.convert("RGB")
                    if rgb_image is not None:
                        rgb_arr = rgb_image.get_numpy_array()
                        numpy_img = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2GRAY)
                    else:
                        numpy_img = raw_image.get_numpy_array()

                if numpy_img is not None:
                    return True, numpy_img
            except Exception as e:
                self._last_error = f"大恒相机采集图像失败: {e}"
                return False, None

        # 2. Physical DirectShow Camera
        if self._cap is not None and self._cap.isOpened() and not self._is_mock:
            ret, frame = self._cap.read()
            if ret and frame is not None:
                if frame.ndim == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                return True, frame

        # 3. Offline sequence playback
        if self._sim_images:
            img_path = self._sim_images[self._sim_index % len(self._sim_images)]
            self._sim_index += 1
            frame = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if frame is not None:
                return True, frame

        # 4. Virtual simulated live frame
        frame = self._generate_simulated_live_frame()
        return True, frame

    def grab_sequence(self, count: int, timeout_ms: int = 5000) -> Tuple[bool, List[np.ndarray]]:
        """
        Synchronously captures a sequence of frames.
        """
        frames: List[np.ndarray] = []
        for _ in range(count):
            ok, frame = self.grab_frame(timeout_ms=timeout_ms)
            if not ok or frame is None:
                return False, frames
            frames.append(frame)
            time.sleep(0.01)
        return True, frames

    def set_simulation_folder(self, folder_path: str):
        if not os.path.isdir(folder_path):
            return
        self._sim_folder = folder_path
        exts = (".bmp", ".png", ".jpg", ".jpeg", ".tif")
        self._sim_images = sorted([
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(exts)
        ])
        self._sim_index = 0

    def _generate_simulated_live_frame(self) -> np.ndarray:
        """
        Generates simulated live frame with moving chessboard target.
        """
        self._sim_frame_count += 1
        w, h = self.actual_width, self.actual_height
        img = np.full((h, w), 35, dtype=np.uint8)

        offset_x = int(np.sin(self._sim_frame_count * 0.05) * 15)
        offset_y = int(np.cos(self._sim_frame_count * 0.04) * 10)

        cols, rows = 11, 8
        sq = 42
        start_x = (w - cols * sq) // 2 + offset_x
        start_y = (h - rows * sq) // 2 + offset_y

        for r in range(rows + 1):
            for c in range(cols + 1):
                x0 = start_x + c * sq
                y0 = start_y + r * sq
                x1 = x0 + sq
                y1 = y0 + sq
                color = 230 if (r + c) % 2 == 0 else 25
                cv2.rectangle(img, (x0, y0), (x1, y1), color, -1)

        t_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        cv2.putText(
            img,
            f"DAHENG GALAXY SIMULATION | {t_str} | Frame #{self._sim_frame_count}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            240,
            2,
        )
        return img
