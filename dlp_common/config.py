"""
Global configurations and default constants for DLP4710 Mono Structured Light System.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class SystemConfig:
    # Projector specs (DLP4710)
    projector_width: int = 1920
    projector_height: int = 1080

    # Camera specs (Daheng 24MP Telecentric)
    camera_width: int = 5328
    camera_height: int = 4608

    # Default fringe pattern parameters
    fringe_periods: List[float] = field(default_factory=lambda: [70.0, 64.0, 59.0])
    phase_step_count: int = 4  # 4-step phase shifting
    modulation_intensity: float = 128.0
    modulation_amplitude: float = 127.0

    # Default calibration board parameters
    board_cols: int = 11  # Number of inner corners horizontally
    board_rows: int = 8   # Number of inner corners vertically
    # The physical board used by the original calibration UI is 3.0 mm.
    square_size_mm: float = 3.0

    # Algorithm hyperparameters
    phase_roi_margin: int = 32
    minimum_view_count: int = 8
    enable_scheimpflug_tilt: bool = True
    enable_projector_tangential: bool = False
    enable_camera_distortion: bool = True
    enable_camera_k3: bool = False
    run_joint_optimization: bool = True
    refine_projector_in_joint: bool = True
    camera_only_iterations: int = 35
    joint_iterations: int = 25
    huber_delta_pixels: float = 1.5

    # Measurement default parameters
    reconstruction_sample_step: int = 4  # Downsampling for interactive performance
    default_exposure_us: float = 150000.0
    default_gain_db: float = 0.0
    calibration_pattern_exposure_us: float = 3000.0
    board_exposure_us: float = 150000.0


DEFAULT_CONFIG = SystemConfig()

