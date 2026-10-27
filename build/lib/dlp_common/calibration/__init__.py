"""
Calibration module for telecentric camera + Scheimpflug DLP system.
"""

from .calib_io import CalibrationIO
from .corner_detector import CornerDetector
from .joint_optimizer import JointCalibrationEngine
from .models import CalibrationPose, CalibrationResult, CalibrationView
from .scheimpflug_dlp import build_dlp_row_plane_normals, compute_tilt_matrix, project_to_dlp
from .telecentric_camera import (
    decompose_affine_camera,
    fit_linear_affine,
    project_telecentric_point,
    undistort_telecentric_pixel,
)

__all__ = [
    "CalibrationIO",
    "CornerDetector",
    "JointCalibrationEngine",
    "CalibrationPose",
    "CalibrationResult",
    "CalibrationView",
    "build_dlp_row_plane_normals",
    "compute_tilt_matrix",
    "project_to_dlp",
    "decompose_affine_camera",
    "fit_linear_affine",
    "project_telecentric_point",
    "undistort_telecentric_pixel",
]

