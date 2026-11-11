"""
Industrial Daheng Galaxy camera driver using Daheng official Python SDK (gxipy).
Provides high-performance frame acquisition, parameter control (exposure, gain),
and software/hardware trigger management matching the C++ DahengCamera implementation.
"""

import os
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple
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

        self._is_opened: bool = False
        self._is_grabbing: bool = False

        self._exposure_us: float = 150000.0
        self._gain_db: float = 0.0
        self._trigger_mode: bool = False
        self._trigger_source: str = "Software"
        self._last_error: str = ""

    @property
    def is_opened(self) -> bool:
        return self._is_opened or self._is_gx_opened

    @property
    def is_grabbing(self) -> bool:
        return self._is_grabbing

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
                    })
            except Exception as e:
                self._last_error = f"Daheng gxipy 枚举异常: {e}"

        return devices

    def open(self, device_identifier=0) -> bool:
        """
        Opens a Daheng Galaxy camera by device index or serial number.
        """
        self.close()

        if not _GXIPY_AVAILABLE:
            self._last_error = "未加载大恒Galaxy SDK Python模块 gxipy"
            return False

        try:
            dm = gx.DeviceManager()
            count, dev_info_list = dm.update_all_device_list()
            if count <= 0:
                self._last_error = "未发现大恒Galaxy相机"
                return False

            if isinstance(device_identifier, str) and not device_identifier.isdigit():
                sn = device_identifier
            else:
                idx = int(device_identifier) if str(device_identifier).isdigit() else 0
                if not 0 <= idx < count:
                    self._last_error = f"大恒相机索引无效: {idx}"
                    return False
                sn = dev_info_list[idx].get("sn")

            self._cam = dm.open_device_by_sn(sn)
            self._device_sn = sn
            self._dev_manager = dm
            self._remote_feature = self._cam.get_remote_device_feature_control()
            self._format_convert = dm.create_image_format_convert()
            self._configure_daheng_camera()
            self._is_gx_opened = True
            self._is_opened = True
            self._last_error = ""
            return True
        except Exception as e:
            self._last_error = f"打开大恒相机失败: {e}"
            return False

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
                # 增大流通道缓冲区数量（默认仅1~2帧），避免高速硬件触发连续多帧时由于缓冲区溢出丢帧
                if hasattr(self._cam, "data_stream") and len(self._cam.data_stream) > 0:
                    try:
                        self._cam.data_stream[0].set_acquisition_buffer_number(32)
                    except Exception:
                        pass
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

        return self._remote_feature is None or not self.is_opened

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

        return self._remote_feature is None or not self.is_opened

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
                    if source == "Line2":
                        if self._remote_feature.is_implemented("LineSelector"):
                            self._remote_feature.get_enum_feature("LineSelector").set("Line2")
                        if self._remote_feature.is_implemented("LineMode"):
                            self._remote_feature.get_enum_feature("LineMode").set("Input")
                        if self._remote_feature.is_implemented("TriggerSelector"):
                            self._remote_feature.get_enum_feature("TriggerSelector").set("FrameStart")
                    self._remote_feature.get_enum_feature("TriggerSource").set(source)
                    if source == "Line2" and self._remote_feature.is_implemented("TriggerActivation"):
                        self._remote_feature.get_enum_feature("TriggerActivation").set("RisingEdge")
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

        self._last_error = "大恒相机未返回有效图像"
        return False, None

    def flush_buffer(self) -> bool:
        """Flushes the stream queue to clear any pending or stale frames."""
        if self._cam is not None and self._is_gx_opened:
            try:
                if hasattr(self._cam, "data_stream") and len(self._cam.data_stream) > 0:
                    self._cam.data_stream[0].flush_queue()
                    return True
            except Exception as e:
                self._last_error = f"清空相机缓冲区失败: {e}"
                return False
        return True

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

    def grab_triggered_sequence(
        self,
        count: int,
        frame_timeout_ms: int = 1000,
        overall_timeout_ms: int = 30000,
        frame_callback: Optional[Callable[[int, np.ndarray], None]] = None,
        retain_frames: bool = True,
        flush_before: bool = False,
    ) -> Tuple[bool, List[np.ndarray]]:
        """Collect a hardware-triggered sequence while tolerating temporary timeouts."""
        if flush_before:
            self.flush_buffer()
        frames: List[np.ndarray] = []
        deadline = time.monotonic() + max(1, overall_timeout_ms) / 1000.0
        empty_timeouts = 0
        received_count = 0
        while received_count < count and time.monotonic() < deadline:
            ok, frame = self.grab_frame(timeout_ms=frame_timeout_ms)
            if ok and frame is not None:
                if frame_callback is not None:
                    frame_callback(received_count, frame)
                if retain_frames:
                    frames.append(frame)
                received_count += 1
                empty_timeouts = 0
            else:
                empty_timeouts += 1
                if received_count == 0 and empty_timeouts >= 5:
                    break
                if received_count > 0 and empty_timeouts >= 3:
                    # Burst sequence has definitely finished; break early to allow retry rather than hanging
                    break
        return received_count == count, frames

