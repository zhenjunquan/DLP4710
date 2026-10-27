"""
Data models for mono structured light calibration.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class CalibrationPose:
    """Represents a single calibration board pose with associated images and points."""
    name: str = ""
    folder_path: str = ""
    camera_image_size: Tuple[int, int] = (0, 0)  # (width, height)
    board_image: Optional[np.ndarray] = None
    horizontal_frames: List[np.ndarray] = field(default_factory=list)
    vertical_frames: List[np.ndarray] = field(default_factory=list)

    camera_corners: Optional[np.ndarray] = None      # Shape (N, 2), float32/float64
    projector_corners: Optional[np.ndarray] = None   # Shape (N, 2), float32/float64

    horizontal_phase: Optional[np.ndarray] = None
    vertical_phase: Optional[np.ndarray] = None
    phase_roi: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)

    corner_valid: bool = False
    phase_valid: bool = False
    camera_view_error: float = -1.0
    projector_view_error: float = -1.0


@dataclass
class CalibrationView:
    """Internal point correspondences for a single validated calibration pose."""
    name: str
    object_points: np.ndarray      # Shape (N, 3), float32/float64
    camera_points: np.ndarray      # Shape (N, 2), float32/float64
    projector_points: np.ndarray   # Shape (N, 2), float32/float64


@dataclass
class CalibrationResult:
    """Holds calibration parameters and evaluation metrics."""
    # DLP Projector (3x3 matrix, 14 distortion params with Scheimpflug tilt)
    projector_matrix: Optional[np.ndarray] = None      # (3, 3)
    projector_dist_14: Optional[np.ndarray] = None     # (14, 1)

    # Telecentric Camera (2x4 affine projection, 5 distortion params)
    telecentric_projection_2x4: Optional[np.ndarray] = None  # (2, 4)
    telecentric_dist_5: Optional[np.ndarray] = None          # (5, 1)

    # Board poses relative to DLP coordinate system
    board_rvecs: List[np.ndarray] = field(default_factory=list)  # List of (3, 1) or (3,)
    board_tvecs: List[np.ndarray] = field(default_factory=list)  # List of (3, 1) or (3,)
    view_names: List[str] = field(default_factory=list)

    # Inspection-only extrinsic decomposition (translation along telecentric axis fixed to 0 gauge)
    projector_to_camera_rotation: Optional[np.ndarray] = None     # (3, 3)
    projector_to_camera_translation: Optional[np.ndarray] = None  # (3, 1)
    telecentric_scale_x: float = 0.0  # px/mm
    telecentric_scale_y: float = 0.0  # px/mm

    # Reprojection errors (pixels)
    camera_rms: float = -1.0
    projector_rms: float = -1.0
    joint_rms: float = -1.0
    camera_view_rms: List[float] = field(default_factory=list)
    projector_view_rms: List[float] = field(default_factory=list)

    warnings: List[str] = field(default_factory=list)

