"""
DLP4710 hardware controller wrapping DLP4710Control.dll via ctypes.
Includes a fully featured Mock / Simulated mode for offline use.
"""

import ctypes
import os
import sys
import time
from typing import Optional, Tuple


class DLP4710Controller:
    """
    Controller for Texas Instruments DLP4710 EVM via Cypress USB-Serial bridge.
    """

    def __init__(self, dll_path: Optional[str] = None):
        self._dll: Optional[ctypes.CDLL] = None
        self._is_mock: bool = False
        self._is_connected: bool = False
        self._pattern_ready: bool = False
        self._is_projecting: bool = False
        self._last_error: str = ""

        # Locate DLL
        if dll_path is None:
            # Default search in workspace libs
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            candidates = [
                os.path.join(base_dir, "libs", "DLP4710Control.dll"),
                os.path.join(
                    base_dir,
                    "..",
                    "zhenjunquan",
                    "DLP4710_MonoSL_Calibration",
                    "MonoSLSystemCalibration",
                    "MonoSLSystemCalibration",
                    "Release",
                    "DLP4710Control.dll",
                ),
            ]
            for cand in candidates:
                if os.path.isfile(cand):
                    dll_path = cand
                    break

        if dll_path and os.path.isfile(dll_path):
            self._load_dll(dll_path)
        else:
            self._is_mock = True

    def _load_dll(self, dll_path: str):
        dll_dir = os.path.dirname(os.path.abspath(dll_path))
        try:
            if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
                os.add_dll_directory(dll_dir)
            self._dll = ctypes.cdll.LoadLibrary(dll_path)

            # Set argtypes and restype
            self._dll.dlpConnect.argtypes = []
            self._dll.dlpConnect.restype = ctypes.c_bool

            self._dll.loadPreBuildPattern.argtypes = [ctypes.c_char_p]
            self._dll.loadPreBuildPattern.restype = ctypes.c_bool

            self._dll.setTriggerOut2.argtypes = []
            self._dll.setTriggerOut2.restype = None

            self._dll.loadPatternOrderTableEntryfromFlash.argtypes = []
            self._dll.loadPatternOrderTableEntryfromFlash.restype = None

            self._dll.writeInternalOperatingMode.argtypes = []
            self._dll.writeInternalOperatingMode.restype = None

            # The exported C++ signature is startWork(const uint8_t&), so the
            # ABI receives a pointer rather than an uint8 value.
            self._dll.startWork.argtypes = [ctypes.POINTER(ctypes.c_uint8)]
            self._dll.startWork.restype = None

            self._dll.stopWork.argtypes = []
            self._dll.stopWork.restype = ctypes.c_bool
        except Exception as e:
            self._last_error = f"加载DLP动态链接库失败: {e}"
            self._dll = None
            self._is_mock = True

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_mock(self) -> bool:
        return self._is_mock

    @property
    def pattern_ready(self) -> bool:
        return self._pattern_ready

    @property
    def last_error(self) -> str:
        return self._last_error

    def connect(self) -> bool:
        """
        Initializes DLP controller and connects to I2C via Cypress bridge.
        """
        if self._is_connected:
            return True
        if self._is_mock:
            self._is_connected = True
            return True

        try:
            ok = bool(self._dll.dlpConnect())
            self._is_connected = ok
            if not ok:
                self._last_error = "DLP连接失败，未找到Cypress USB-Serial桥接设备或I2C未应答"
            return ok
        except Exception as e:
            self._last_error = f"DLP连接异常: {e}"
            self._is_connected = False
            return False

    def burn_bin(self, bin_path: str) -> Tuple[bool, str]:
        """
        Burns the pre-built pattern BIN file to DLP flash and configures internal mode.
        """
        if not os.path.isfile(bin_path):
            self._last_error = f"BIN文件不存在: {bin_path}"
            return False, self._last_error

        if not self._is_connected and not self.connect():
            return False, self._last_error or "DLP尚未连接，无法烧录BIN"

        if self._is_mock:
            time.sleep(0.5)
            self._pattern_ready = True
            return True, "模拟模式: BIN配置就绪"

        try:
            bin_path_bytes = bin_path.encode("gbk", errors="ignore")
            # 1. Load patterns into flash
            burn_ok = bool(self._dll.loadPreBuildPattern(bin_path_bytes))
            if not burn_ok:
                self._last_error = "loadPreBuildPattern返回false，请检查BIN路径与权限"
                return False, self._last_error

            # 2. Configure Trigger Out 2
            self._dll.setTriggerOut2()

            # 3. Load pattern order table from flash
            self._dll.loadPatternOrderTableEntryfromFlash()

            # 4. Set operating mode
            self._dll.writeInternalOperatingMode()

            self._pattern_ready = True
            return True, "BIN烧录及工作模式配置成功"
        except Exception as e:
            self._last_error = f"BIN烧录异常: {e}"
            return False, self._last_error

    def prepare_pattern_mode(self) -> Tuple[bool, str]:
        """Load the pattern order already stored in Flash and enter internal mode."""
        if not self._is_connected and not self.connect():
            return False, self._last_error or "DLP尚未连接"

        if self._is_mock:
            self._pattern_ready = True
            return True, "模拟DLP条纹模式已就绪"

        try:
            self._dll.setTriggerOut2()
            self._dll.loadPatternOrderTableEntryfromFlash()
            self._dll.writeInternalOperatingMode()
            self._pattern_ready = True
            self._last_error = ""
            return True, "DLP已加载Flash条纹序列并进入内部投影模式"
        except Exception as e:
            self._pattern_ready = False
            self._last_error = f"DLP条纹模式初始化异常: {e}"
            return False, self._last_error

    def start_work(self, repeat_num: int = 0) -> bool:
        """
        Starts pattern projection. repeat_num = 0 for continuous/infinite repeat.
        """
        if self._is_mock:
            self._is_projecting = True
            return True

        if not self._is_connected:
            self._last_error = "DLP尚未连接，不能启动投影"
            return False
        if not self._pattern_ready:
            self._last_error = "DLP条纹序列尚未就绪，请先烧录24帧标定BIN"
            return False

        try:
            repeat_value = ctypes.c_uint8(repeat_num)
            self._dll.startWork(ctypes.byref(repeat_value))
            self._is_projecting = True
            self._last_error = ""
            return True
        except Exception as e:
            self._last_error = f"启动DLP投影异常: {e}"
            self._is_projecting = False
            return False

    def stop_work(self) -> bool:
        """
        Stops pattern projection.
        """
        if self._is_mock:
            self._is_projecting = False
            self._is_connected = False
            return True

        try:
            ok = bool(self._dll.stopWork())
            self._is_projecting = False
            self._is_connected = False
            return ok
        except Exception as e:
            self._last_error = f"停止DLP投影异常: {e}"
            return False

    def disconnect(self) -> bool:
        """Stop projection and release the logical DLP connection state."""
        if not self._is_connected and not self._is_projecting:
            self._pattern_ready = False
            return True

        ok = self.stop_work()
        self._is_projecting = False
        self._is_connected = False
        self._pattern_ready = False
        return ok

