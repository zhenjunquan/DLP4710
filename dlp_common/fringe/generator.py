"""
Fringe pattern generator for DLP4710 (Horizontal and Vertical sinusoidal patterns).
"""

import os
from typing import List, Tuple
import cv2
import numpy as np


class FringeGenerator:
    """
    Generates sinusoidal fringe patterns for 3-frequency 4-step structured light.
    Direction:
      0: Horizontal stripes (phase varies along DLP Y axis, vertical height).
      1: Vertical stripes (phase varies along DLP X axis, horizontal width).
    """

    def __init__(
        self,
        width: int = 1920,
        height: int = 1080,
        periods: List[float] = None,
        step_num: int = 4,
        intensity: float = 128.0,
        amplitude: float = 127.0,
    ):
        self.width = width
        self.height = height
        self.periods = periods if periods is not None else [70.0, 64.0, 59.0]
        self.step_num = step_num
        self.intensity = intensity
        self.amplitude = amplitude

    def generate_directional_fringes(self, direction: int) -> List[np.ndarray]:
        """
        Generates 12 fringe images (3 frequencies * 4 phase shifts).
        direction: 0 = Horizontal (Y-gradient), 1 = Vertical (X-gradient)
        """
        if direction not in (0, 1):
            raise ValueError(f"Invalid direction: {direction}. Must be 0 (horizontal) or 1 (vertical).")

        axis_length = float(self.width if direction == 1 else self.height)
        fringe_images: List[np.ndarray] = []

        # Coordinate grid: starts from 1, exactly matching C++: k = direction == 1 ? (x + 1) : (y + 1)
        if direction == 1:
            k = np.arange(1, self.width + 1, dtype=np.float64)[None, :]
        else:
            k = np.arange(1, self.height + 1, dtype=np.float64)[:, None]

        for period_cycles in self.periods:
            freq = period_cycles / axis_length
            for step in range(self.step_num):
                phase = float(step) * 2.0 * np.pi / float(self.step_num)
                val_1d = self.intensity + self.amplitude * np.cos(2.0 * np.pi * k * freq + phase)

                if direction == 1:
                    # Broadcast across all rows
                    img = np.tile(val_1d, (self.height, 1))
                else:
                    # Broadcast across all columns
                    img = np.tile(val_1d, (1, self.width))

                img_uint8 = np.clip(np.round(img), 0, 255).astype(np.uint8)
                fringe_images.append(img_uint8)

        return fringe_images

    def save_fringes(self, direction: int, output_dir: str) -> List[str]:
        """
        Generates and saves the 12 fringe images to output_dir.
        Returns list of saved file paths.
        """
        os.makedirs(output_dir, exist_ok=True)
        images = self.generate_directional_fringes(direction)
        prefix = "V_Fringe_" if direction == 1 else "H_Fringe_"
        saved_paths: List[str] = []

        for idx, img in enumerate(images, start=1):
            file_name = f"{prefix}{idx}.bmp"
            full_path = os.path.join(output_dir, file_name)
            cv2.imwrite(full_path, img)
            saved_paths.append(full_path)

        return saved_paths

