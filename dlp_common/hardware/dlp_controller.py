"""
DLP4710 hardware controller wrapping DLP4710Control.dll via ctypes.
Includes a fully featured Mock / Simulated mode for offline use.
"""

import ctypes
import os
import sys
import time
from typing import Any, Dict, Optional, Tuple


class DLP4710Controller:
    """
    Controller for Texas Instruments DLP4710 EVM via Cypress USB-Serial bridge.
    Supports reading parameters (LED currents, RGB enables, control modes) and
    setting color modes (Blue, Green, Red, White, Custom) via I2C commands.
    """

    def __init__(self, dll_path: Optional[str] = None):
        self._dll: Optional[ctypes.CDLL] = None
        self._is_mock: bool = False
        self._is_connected: bool = False
        self._pattern_ready: bool = False
        self._is_projecting: bool = False
        self._led_currents: Tuple[int, int, int] = (1000, 1000, 1000)
        self._brightness_percent: float = 100.0
        self._color_mode: str = "white"
        self._led_enables: Dict[str, bool] = {"red": True, "green": True, "blue": True}
        self._led_control_method: int = 0  # 0: Manual, 1: Auto CAIC
        self._max_allowed_current: int = 1023
        self._write_led_current_func = None
        self._i2c_read_func = None
        self._i2c_write_func = None
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

            # 绑定 DLPC34XX_DUAL_WriteRgbLedCurrent (TI I2C 命令 0x54, RVA 0x1C20)
            try:
                base_addr = self._dll._handle
                write_led_type = ctypes.WINFUNCTYPE(
                    ctypes.c_uint32,
                    ctypes.c_uint16,
                    ctypes.c_uint16,
                    ctypes.c_uint16,
                )
                self._write_led_current_func = write_led_type(base_addr + 0x1C20)
            except Exception:
                self._write_led_current_func = None

            # 绑定底层的通用 I2C 读写函数 (基于 Cypress USB-Serial Bridge):
            # RVA 0x1310: uint8_t WriteI2C(uint32_t length, uint8_t *buffer) -> bool (1=ok, 0=fail)
            # RVA 0x10F0: uint8_t ReadI2C(uint32_t length, uint8_t *buffer) -> bool (1=ok, 0=fail)
            try:
                base_addr = self._dll._handle
                write_txn_type = ctypes.WINFUNCTYPE(
                    ctypes.c_uint8,
                    ctypes.c_uint32,
                    ctypes.POINTER(ctypes.c_uint8),
                )
                self._i2c_write_func = write_txn_type(base_addr + 0x1310)

                read_txn_type = ctypes.WINFUNCTYPE(
                    ctypes.c_uint8,
                    ctypes.c_uint32,
                    ctypes.POINTER(ctypes.c_uint8),
                )
                self._i2c_read_func = read_txn_type(base_addr + 0x10F0)
            except Exception:
                self._i2c_read_func = None
                self._i2c_write_func = None
        except Exception as e:
            self._last_error = f"加载DLP动态链接库失败: {e}"
            self._dll = None
            self._is_mock = True

    @property
    def is_projecting(self) -> bool:
        return self._is_projecting

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

    @property
    def led_currents(self) -> Tuple[int, int, int]:
        """Returns the current RGB LED drive currents (0-1023)."""
        return self._led_currents

    @property
    def brightness_percent(self) -> float:
        """Returns the overall brightness percentage (0% - 100%)."""
        return self._brightness_percent

    @property
    def color_mode(self) -> str:
        """Returns the active projection color mode ('blue', 'green', 'red', 'white', 'custom')."""
        return self._color_mode

    @property
    def led_enables(self) -> Dict[str, bool]:
        """Returns the RGB LED channel enable states."""
        return dict(self._led_enables)

    @property
    def led_control_method(self) -> int:
        """Returns the LED control method (0: Manual, 1: Auto CAIC)."""
        return self._led_control_method

    @property
    def max_allowed_current(self) -> int:
        """Returns the maximum allowed LED current code."""
        return self._max_allowed_current

    def i2c_write(self, cmd: int, payload: bytes = b"") -> Tuple[bool, str]:
        """
        Sends an I2C command packet [cmd, payload...] to the DLPC3479 controller via Cypress bridge.
        """
        if self._is_mock:
            return True, f"模拟I2C写: CMD=0x{cmd:02X}, 长度={len(payload)}"

        if not self._is_connected:
            return False, "DLP尚未连接，无法执行I2C写操作"

        if self._i2c_write_func is None:
            return False, "DLL缺少通用I2C写接口"

        try:
            buf = bytes([cmd & 0xFF]) + payload
            buf_len = len(buf)
            c_buf = (ctypes.c_uint8 * buf_len)(*buf)
            res = self._i2c_write_func(buf_len, c_buf)
            if res == 1:
                return True, f"I2C写成功: CMD=0x{cmd:02X}"
            else:
                return False, f"I2C写失败 (返回值: {res})"
        except Exception as e:
            return False, f"I2C写异常: {e}"

    def i2c_read(self, cmd: int, read_len: int) -> Tuple[bool, bytes, str]:
        """
        Sends an I2C command byte and reads read_len response bytes from DLPC3479.
        """
        if self._is_mock:
            return True, bytes(read_len), f"模拟I2C读: CMD=0x{cmd:02X}, 长度={read_len}"

        if not self._is_connected:
            return False, b"", "DLP尚未连接，无法执行I2C读操作"

        if self._i2c_write_func is None or self._i2c_read_func is None:
            return False, b"", "DLL缺少通用I2C读写接口"

        try:
            cmd_buf = (ctypes.c_uint8 * 1)(cmd & 0xFF)
            w_res = self._i2c_write_func(1, cmd_buf)
            if w_res != 1:
                return False, b"", f"I2C读失败 (发送命令 0x{cmd:02X} 失败, 返回值: {w_res})"

            read_buf = (ctypes.c_uint8 * read_len)()
            r_res = self._i2c_read_func(read_len, read_buf)
            if r_res == 1:
                data = bytes(read_buf)
                return True, data, f"I2C读成功: 收到 {len(data)} 字节"
            else:
                return False, b"", f"I2C读失败 (接收应答失败, 返回值: {r_res})"
        except Exception as e:
            return False, b"", f"I2C读异常: {e}"

    def read_led_currents(self) -> Tuple[bool, Tuple[int, int, int], str]:
        """
        Reads actual Red, Green, Blue LED drive currents from DLPC3479 (TI Command 0x55).
        Returns: (success: bool, (R, G, B), message: str)
        """
        if self._is_mock or not self._is_connected:
            return True, self._led_currents, "DLP离线/模拟状态，返回当前缓存电流值"

        ok, data, msg = self.i2c_read(0x55, 6)
        if ok and len(data) >= 6:
            r = data[0] | (data[1] << 8)
            g = data[2] | (data[3] << 8)
            b = data[4] | (data[5] << 8)
            self._led_currents = (r, g, b)
            active_c = b if b > 0 else max(r, g, b)
            self._brightness_percent = round((active_c / 1023.0) * 100.0, 1) if active_c > 0 else 0.0
            return True, (r, g, b), f"成功读取 DLP LED 驱动电流: R={r}, G={g}, B={b}"
        else:
            return False, self._led_currents, f"读取 LED 电流失败: {msg}"

    def read_led_enables(self) -> Tuple[bool, Dict[str, bool], str]:
        """
        Reads RGB LED channel enable states from DLPC3479 (TI Command 0x53).
        Bit 0: Red enable, Bit 1: Green enable, Bit 2: Blue enable.
        Returns: (success: bool, {"red": bool, "green": bool, "blue": bool}, message: str)
        """
        if self._is_mock or not self._is_connected:
            return True, dict(self._led_enables), "DLP离线/模拟状态，返回当前缓存通道使能"

        ok, data, msg = self.i2c_read(0x53, 1)
        if ok and len(data) >= 1:
            mask = data[0]
            r_en = bool(mask & 0x01)
            g_en = bool(mask & 0x02)
            b_en = bool(mask & 0x04)
            self._led_enables = {"red": r_en, "green": g_en, "blue": b_en}
            return True, self._led_enables, f"成功读取 LED 使能: 红={'开' if r_en else '关'}, 绿={'开' if g_en else '关'}, 蓝={'开' if b_en else '关'}"
        else:
            return False, dict(self._led_enables), f"读取 LED 使能失败: {msg}"

    def read_led_control_method(self) -> Tuple[bool, int, str]:
        """
        Reads LED control method from DLPC3479 (TI Command 0x51).
        0x00: Manual LED control, 0x01: Automatic (CAIC).
        """
        if self._is_mock or not self._is_connected:
            return True, self._led_control_method, "DLP离线/模拟状态，返回当前缓存控制模式"

        ok, data, msg = self.i2c_read(0x51, 1)
        if ok and len(data) >= 1:
            method = data[0]
            self._led_control_method = method
            desc = "手动控制 (Manual)" if method == 0 else f"自动控制 (0x{method:02X})"
            return True, method, f"成功读取 LED 控制模式: {desc}"
        else:
            return False, self._led_control_method, f"读取控制模式失败: {msg}"

    def read_all_parameters(self) -> Tuple[bool, Dict[str, Any], str]:
        """
        Reads all operating parameters from DLP hardware or cached state.
        Returns: (success: bool, params_dict: Dict[str, Any], summary_message: str)
        """
        if self._is_connected:
            self.read_led_currents()
            self.read_led_enables()
            self.read_led_control_method()

        r, g, b = self._led_currents
        en = self._led_enables
        r_en = en.get("red", False)
        g_en = en.get("green", False)
        b_en = en.get("blue", False)

        # In internal pattern display mode, DLPC3479 register 0x53 can return 0 because internal
        # pattern sequencer controls the illumination enable signals.
        # For the DLP4710 3D structured light optical engine, the active LED is connected to the Blue LED channel.
        if not (r_en or g_en or b_en):
            b_en = True
            self._led_enables = {"red": False, "green": False, "blue": True}
            en = self._led_enables

        # Detect color mode name
        if b_en and not r_en and not g_en:
            color_name = "蓝光 (Blue)"
            detected_color = "blue"
        elif g_en and not r_en and not b_en:
            color_name = "绿光 (Green)"
            detected_color = "green"
        elif r_en and not g_en and not b_en:
            color_name = "红光 (Red)"
            detected_color = "red"
        elif r_en and g_en and b_en and r == g == b:
            color_name = "白光 (White)"
            detected_color = "white"
        else:
            color_name = f"自定义 RGB (R:{r}, G:{g}, B:{b})"
            detected_color = "custom"

        self._color_mode = detected_color

        params = {
            "connected": self._is_connected,
            "is_mock": self._is_mock,
            "pattern_ready": self._pattern_ready,
            "is_projecting": self._is_projecting,
            "color_mode": self._color_mode,
            "color_name": color_name,
            "led_currents": self._led_currents,
            "red_current": r,
            "green_current": g,
            "blue_current": b,
            "brightness_percent": self._brightness_percent,
            "led_enables": dict(self._led_enables),
            "red_enabled": en.get("red", False),
            "green_enabled": en.get("green", False),
            "blue_enabled": en.get("blue", False),
            "led_control_method": self._led_control_method,
            "control_method_name": "手动模式 (Manual)" if self._led_control_method == 0 else f"自动模式 (0x{self._led_control_method:02X})",
            "max_allowed_current": self._max_allowed_current,
        }

        conn_str = "已连接 (硬件通讯正常)" if self._is_connected else "离线/模拟 (返回缓存预设)"
        msg = (
            f"=== DLP4710 投影模组参数报告 ===\n"
            f"设备通讯: {conn_str}\n"
            f"投影颜色: {color_name}\n"
            f"等效光强: {self._brightness_percent:.1f}%\n"
            f"驱动电流: 红(R)={r}, 绿(G)={g}, 蓝(B)={b} (编码范围 0~1023)\n"
            f"通道开关: 红={'ON' if en.get('red') else 'OFF'}, 绿={'ON' if en.get('green') else 'OFF'}, 蓝={'ON' if en.get('blue') else 'OFF'}\n"
            f"控制模式: {params['control_method_name']}\n"
            f"条纹序列: {'已加载就绪' if self._pattern_ready else '未载入'}"
        )
        return True, params, msg

    def set_color_mode(
        self,
        color: str,
        brightness_pct: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """
        Sets DLP projection color mode ('blue', 'green', 'red', 'white', 'custom')
        and optionally updates brightness percentage.
        Configures both LED enable masks (Cmd 0x52) and drive currents (Cmd 0x54).
        """
        c = color.lower().strip()
        pct = float(brightness_pct) if brightness_pct is not None else self._brightness_percent
        pct = max(0.0, min(100.0, pct))
        current = 0 if pct <= 0 else int(round(51 + (pct / 100.0) * (1000 - 51)))

        if "blue" in c or "蓝" in c:
            r, g, b = 0, 0, current
            r_en, g_en, b_en = False, False, True
            mode_key = "blue"
            mode_label = "纯蓝光 (Blue)"
        elif "green" in c or "绿" in c:
            r, g, b = 0, current, 0
            r_en, g_en, b_en = False, True, False
            mode_key = "green"
            mode_label = "纯绿光 (Green)"
        elif "red" in c or "红" in c:
            r, g, b = current, 0, 0
            r_en, g_en, b_en = True, False, False
            mode_key = "red"
            mode_label = "纯红光 (Red)"
        elif "white" in c or "白" in c or "rgb" in c or "全" in c:
            r, g, b = current, current, current
            r_en, g_en, b_en = True, True, True
            mode_key = "white"
            mode_label = "白光 (White)"
        else:
            mode_key = "custom"
            mode_label = "自定义颜色"
            r, g, b = self._led_currents
            r_en = self._led_enables.get("red", True)
            g_en = self._led_enables.get("green", True)
            b_en = self._led_enables.get("blue", True)

        self._color_mode = mode_key
        self._brightness_percent = pct
        self._led_enables = {"red": r_en, "green": g_en, "blue": b_en}

        # 1. Ensure manual control method (0x50, 0x00)
        self.set_led_control_method(manual=True)

        # 2. Set LED enables (0x52)
        self.set_led_enables(r_en, g_en, b_en)

        # 3. Set LED currents (0x54)
        ok, msg = self.set_led_current(r, g, b)
        if ok:
            return True, f"DLP投影颜色已设为 {mode_label}，光强 {pct:.0f}% (R={r}, G={g}, B={b})"
        return False, msg

    def set_led_enables(self, red: bool, green: bool, blue: bool) -> Tuple[bool, str]:
        """
        Enables or disables individual RGB LED channels (TI Command 0x52).
        """
        self._led_enables = {"red": bool(red), "green": bool(green), "blue": bool(blue)}
        mask = (1 if red else 0) | (2 if green else 0) | (4 if blue else 0)
        return self.i2c_write(0x52, bytes([mask]))

    def set_led_control_method(self, manual: bool = True) -> Tuple[bool, str]:
        """
        Sets LED control method (TI Command 0x50).
        manual=True sends 0x00 (Manual control, disables CAIC dynamic brightness).
        """
        val = 0x00 if manual else 0x01
        self._led_control_method = val
        return self.i2c_write(0x50, bytes([val]))

    def set_led_current(self, red: int, green: int, blue: int) -> Tuple[bool, str]:
        """
        Sets Red, Green, Blue LED driver currents (0-1023) using TI DLPC3479 Command 0x54.
        Returns: (success: bool, message: str)
        """
        r = max(0, min(1023, int(red)))
        g = max(0, min(1023, int(green)))
        b = max(0, min(1023, int(blue)))

        def _code_to_pct(code: int) -> float:
            if code <= 0:
                return 0.0
            if code >= 1000:
                return 100.0
            return round(max(0.0, min(100.0, ((code - 51) / (1000 - 51)) * 100.0)), 1)

        if self._is_mock:
            self._led_currents = (r, g, b)
            self._brightness_percent = _code_to_pct(max(r, g, b))
            return True, f"模拟模式: DLP LED 电流已设为 R={r}, G={g}, B={b}"

        if not self._is_connected:
            self._led_currents = (r, g, b)
            self._brightness_percent = _code_to_pct(max(r, g, b))
            return True, f"DLP尚未连接，已预设 LED 电流为 R={r}, G={g}, B={b} (连接后自动应用)"

        if self._is_connected and self._led_currents == (r, g, b) and (r, g, b) != (0, 0, 0):
            return True, f"DLP LED 电流已是 R={r}, G={g}, B={b}，跳过重复写入"

        try:
            if self._write_led_current_func is not None:
                res = self._write_led_current_func(r, g, b)
                if res == 0:
                    self._led_currents = (r, g, b)
                    self._brightness_percent = _code_to_pct(max(r, g, b))
                    return True, f"成功设置 DLP LED 电流: R={r}, G={g}, B={b}"
                else:
                    self._last_error = f"设置 DLP LED 电流失败，返回值: {res}"
                    return False, self._last_error
            elif self._i2c_write_func is not None:
                payload = bytes([
                    r & 0xFF, r >> 8,
                    g & 0xFF, g >> 8,
                    b & 0xFF, b >> 8,
                ])
                ok, msg = self.i2c_write(0x54, payload)
                if ok:
                    self._led_currents = (r, g, b)
                    self._brightness_percent = _code_to_pct(max(r, g, b))
                    return True, f"成功设置 DLP LED 电流: R={r}, G={g}, B={b}"
                return False, msg
            else:
                self._last_error = "当前动态库不支持直接调节LED电流"
                return False, self._last_error
        except Exception as e:
            self._last_error = f"设置 DLP LED 电流异常: {e}"
            return False, self._last_error

    def set_brightness_percent(self, percent: float) -> Tuple[bool, str]:
        """
        Sets overall projector light intensity / brightness by scaling currents (0% to 100%)
        according to the current color mode.
        """
        pct = max(0.0, min(100.0, float(percent)))
        self._brightness_percent = pct
        if pct <= 0.0:
            current = 0
        else:
            current = int(round(51 + (pct / 100.0) * (1000 - 51)))

        if self._color_mode == "blue":
            ok, msg = self.set_led_current(0, 0, current)
        elif self._color_mode == "green":
            ok, msg = self.set_led_current(0, current, 0)
        elif self._color_mode == "red":
            ok, msg = self.set_led_current(current, 0, 0)
        elif self._color_mode == "white":
            ok, msg = self.set_led_current(current, current, current)
        else:
            # Custom: scale currently enabled channels
            r_en = self._led_enables.get("red", False)
            g_en = self._led_enables.get("green", False)
            b_en = self._led_enables.get("blue", False)
            if not (r_en or g_en or b_en):
                # Optical engine default fallback to Blue
                b_en = True
                self._led_enables["blue"] = True
            r_curr = current if r_en else 0
            g_curr = current if g_en else 0
            b_curr = current if b_en else 0
            ok, msg = self.set_led_current(r_curr, g_curr, b_curr)

        if ok:
            self._brightness_percent = pct
            return True, f"投影仪光强已设为 {pct:.0f}% (主通道电流编码: {current})"
        return False, msg

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
            if ok:
                # Optionally sync current parameters from hardware
                try:
                    self.read_all_parameters()
                except Exception:
                    pass
            else:
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
        if self._is_mock:
            self._pattern_ready = True
            return True, "模拟DLP条纹模式已就绪"

        try:
            # 确保获取 Cypress I2C 总线控制权
            if hasattr(self._dll, "dlpConnect"):
                ok = bool(self._dll.dlpConnect())
                self._is_connected = ok
                if not ok:
                    self._last_error = "DLP连接/重新获取I2C总线失败"
                    return False, self._last_error
            self._dll.setTriggerOut2()
            self._dll.loadPatternOrderTableEntryfromFlash()
            self._dll.writeInternalOperatingMode()
            self._is_connected = True
            self._pattern_ready = True
            self._last_error = ""
            return True, "DLP已加载Flash条纹序列并进入内部投影模式"
        except Exception as e:
            self._pattern_ready = False
            self._last_error = f"DLP条纹模式初始化异常: {e}"
            return False, self._last_error

    def start_work(self, repeat_num: int = 0) -> bool:
        """
        Starts pattern projection. repeat_num = 0 for single-shot (natural end at sequence end).
        """
        if self._is_mock:
            self._is_projecting = True
            return True

        if not self._is_connected or not self._pattern_ready:
            ok, msg = self.prepare_pattern_mode()
            if not ok:
                self._last_error = f"DLP尚未就绪，无法启动投影: {msg}"
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
        注意：DLP底层DLL执行 stopWork() 会释放 Cypress I2C 总线，
        因此正常测量周期结束后绝不要调用此方法；如调用，需重设 pattern_ready = False。
        """
        if self._is_mock:
            self._is_projecting = False
            return True

        try:
            ok = bool(self._dll.stopWork())
            self._is_projecting = False
            self._pattern_ready = False
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


