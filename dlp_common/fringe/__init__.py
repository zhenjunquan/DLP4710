"""
Fringe generation and phase unwrapping module.
"""

from .generator import FringeGenerator
from .phase_unwrapping import PhaseProcessor
from .multi_exposure import (
    compute_sequence_modulation,
    fuse_multi_exposure_fringe_patterns,
)
from .cuda_phase import (
    is_cuda_available,
    cuda_solve_wrapped_phases,
    cuda_unwrap_wrapped_phases,
    cuda_compute_mask,
)
from .cuda_multi_exposure import cuda_fuse_multi_exposure_fringe_patterns
from .utils import bilinear_sample, bilinear_sample_points

__all__ = [
    "FringeGenerator",
    "PhaseProcessor",
    "compute_sequence_modulation",
    "fuse_multi_exposure_fringe_patterns",
    "is_cuda_available",
    "cuda_solve_wrapped_phases",
    "cuda_unwrap_wrapped_phases",
    "cuda_compute_mask",
    "cuda_fuse_multi_exposure_fringe_patterns",
    "bilinear_sample",
    "bilinear_sample_points",
]

