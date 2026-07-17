"""Pipeline A front end (M1): preprocessing + filterbank + structured primitives."""

from .filterbank import PhysicalFilterbankTokenizer
from .preprocess import (
    accel_gyro_triads,
    create_patches,
    estimate_gravity,
    find_triads,
    gravity_align,
    preprocess_imu_data,
    zero_pad_patches,
)
from .primitives import Primitive, compute_primitives
from .scattering import build_frontend

__all__ = [
    "PhysicalFilterbankTokenizer",
    "Primitive",
    "accel_gyro_triads",
    "build_frontend",
    "compute_primitives",
    "create_patches",
    "estimate_gravity",
    "find_triads",
    "gravity_align",
    "preprocess_imu_data",
    "zero_pad_patches",
]
