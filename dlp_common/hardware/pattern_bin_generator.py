"""
DLP4710 PATN Pattern BIN Generator with Color and Frequency Customization.
Generates 12-frame (3-frequency x 4-step) horizontal fringe BIN files compatible with DLPC3479.
Supports setting illumination color: Blue (0x04), Green (0x02), Red (0x01), White (0x07).
"""

import os
from pathlib import Path
import struct
from typing import Optional, Sequence, Tuple
import numpy as np

HEIGHT = 1080
PAD = 8
BYTES_PER_PATTERN = HEIGHT + PAD  # 1088 bytes (8 bit planes x 136 bytes)
PATTERNS_PER_SET = 6
SET_COUNT = 2

# TI DLPC3479 LED Selection Codes for Pattern Order Table
LED_COLOR_CODES = {
    "red": 0x01,
    "红": 0x01,
    "green": 0x02,
    "绿": 0x02,
    "blue": 0x04,
    "蓝": 0x04,
    "white": 0x07,  # Red | Green | Blue (0x01 | 0x02 | 0x04)
    "白": 0x07,
    "rgb": 0x07,
}


def encode_profile(profile: np.ndarray) -> bytes:
    """Encodes 1088 grayscale samples into 8 LSB-first bit planes."""
    gray = np.asarray(profile, dtype=np.uint8)
    if gray.shape != (HEIGHT + PAD,):
        raise ValueError(f"Expected {HEIGHT + PAD} grayscale samples, got {gray.shape}")
    planes = (gray[None, :] >> np.arange(8, dtype=np.uint8)[:, None]) & 1
    return np.packbits(planes, axis=1, bitorder="little").tobytes()


def decode_profile(raw: bytes) -> np.ndarray:
    """Decodes one 1088-byte encoded pattern into 1088 grayscale samples."""
    if len(raw) != BYTES_PER_PATTERN:
        raise ValueError("Wrong encoded pattern length")
    encoded = np.frombuffer(raw, dtype=np.uint8).reshape(8, 136)
    planes = np.unpackbits(encoded, axis=1, bitorder="little")
    return (planes.T @ (1 << np.arange(8))).astype(np.uint8)


def build_horizontal_profiles(
    frequencies: Sequence[float] = (70.0, 64.0, 59.0),
    mean: float = 128.0,
    amplitude: float = 127.0,
) -> list:
    """
    Generates horizontal sinusoidal fringe stripes along vertical 1080 DMD rows.
    Returns list of 12 profiles (3 frequencies x 4 phase shifts: 0, 90, 180, 270 deg).
    """
    x = np.arange(1, HEIGHT + 1, dtype=np.float64)
    profiles = []
    for freq in frequencies:
        for phase_deg in (0, 90, 180, 270):
            phase_rad = np.deg2rad(phase_deg)
            intensity = mean + amplitude * np.cos(2 * np.pi * freq * x / float(HEIGHT) + phase_rad)
            values = np.clip(np.rint(intensity), 0, 255).astype(np.uint8)
            profile = np.zeros(HEIGHT + PAD, dtype=np.uint8)
            profile[PAD:] = values
            profiles.append(profile)
    return profiles


def make_patn_header(led_code: int = 0x04) -> bytes:
    """
    Constructs the 84-byte standalone PATN header with the specified illumination LED code.
    led_code: 0x01 (Red), 0x02 (Green), 0x04 (Blue), 0x07 (White/All).
    Timing values: exposure=4000 us, dark=10000 us, period=40000 us.
    """
    header = bytearray()
    header += struct.pack("<4sIIII", b"PATN", 72, 26140, 20, 52)
    header += struct.pack("<I", 2)  # Two pattern order entries
    for set_id in (0, 1):
        # set_id, count(6), led_code, reserved, inversion, 3 timing fields
        header += struct.pack(
            "<BBBBQIII", set_id, 6, led_code & 0xFF, 0, 0, 4000, 10000, 40000
        )
    header += struct.pack("<III", 2, 84, 13148)
    if len(header) != 84:
        raise RuntimeError("Unexpected PATN header length")
    return bytes(header)


def generate_pattern_bin(
    output_path: str,
    led_color: str = "blue",
    freqs: Sequence[float] = (70.0, 64.0, 59.0),
    mean: float = 128.0,
    amplitude: float = 127.0,
) -> Tuple[bool, str]:
    """
    Generates a complete standalone 12-frame horizontal fringe PATN BIN file.
    Supports LED color: 'blue', 'green', 'red', 'white'.
    """
    try:
        color_key = led_color.lower().strip()
        led_code = LED_COLOR_CODES.get(color_key, 0x04)

        profiles = build_horizontal_profiles(freqs, mean=mean, amplitude=amplitude)
        if len(profiles) != 12:
            return False, "需生成刚好12张正弦条纹图样"

        result = bytearray(make_patn_header(led_code=led_code))
        for set_number in (0, 1):
            raw = b"".join(
                encode_profile(p) for p in profiles[set_number * 6 : (set_number + 1) * 6]
            )
            result += struct.pack("<4BI", 6, 0, 8, 0, len(raw) * 2)
            result += raw * 2

        out_bytes = bytes(result)
        if len(out_bytes) != 26212:
            return False, f"BIN文件长度异常: {len(out_bytes)} (期望 26212)"

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(out_bytes)

        return True, f"成功生成DLP条纹BIN: {output_path} (颜色: {led_color}, 频率: {freqs})"
    except Exception as exc:
        return False, f"生成BIN文件失败: {exc}"

