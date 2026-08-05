"""Pipeline A front end (M1): preprocessing + filterbank + structured primitives."""

from .filterbank import PhysicalFilterbankTokenizer
from .preprocess import (
    accel_gyro_triads,
    estimate_gravity,
    find_triads,
    gravity_align,
)
from .primitives import Primitive, compute_primitives

__all__ = [
    "PhysicalFilterbankTokenizer",
    "Primitive",
    "accel_gyro_triads",
    "compute_primitives",
    "estimate_gravity",
    "find_triads",
    "gravity_align",
]
