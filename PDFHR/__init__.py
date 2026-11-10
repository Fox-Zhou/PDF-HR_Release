"""Shared PDF-HR model definitions and checkpoint loading utilities."""

from .field_specs import FEATURE_LAYOUT, FieldSpec, get_field_spec
from .model import (
    build_distance_field,
    build_pose_distance_field,
    get_default_model_cfg,
    load_acceleration_distance_field,
    load_distance_field,
    load_pose_distance_field,
    load_velocity_distance_field,
)

__all__ = [
    "FEATURE_LAYOUT",
    "FieldSpec",
    "build_distance_field",
    "build_pose_distance_field",
    "get_default_model_cfg",
    "get_field_spec",
    "load_acceleration_distance_field",
    "load_distance_field",
    "load_pose_distance_field",
    "load_velocity_distance_field",
]
