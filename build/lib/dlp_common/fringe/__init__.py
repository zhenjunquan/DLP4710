"""
Fringe generation and phase unwrapping module.
"""

from .generator import FringeGenerator
from .phase_unwrapping import PhaseProcessor
from .utils import bilinear_sample, bilinear_sample_points

__all__ = [
    "FringeGenerator",
    "PhaseProcessor",
    "bilinear_sample",
    "bilinear_sample_points",
]

