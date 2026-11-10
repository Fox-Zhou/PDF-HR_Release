from __future__ import annotations

from dataclasses import dataclass


FEATURE_LAYOUT = "root_gravity_body_v2"


@dataclass(frozen=True)
class StatGroup:
    name: str
    value_slice: slice
    width: int


@dataclass(frozen=True)
class FieldSpec:
    name: str
    sample_columns: int
    input_dim: int
    root_target_index: int
    joint_target_index: int
    stat_groups: tuple[StatGroup, ...]

    def target_index(self, root: bool) -> int:
        return self.root_target_index if root else self.joint_target_index


POSE_SPEC = FieldSpec(
    name="pose",
    sample_columns=34,
    input_dim=32,
    root_target_index=32,
    joint_target_index=33,
    stat_groups=(StatGroup("pose_joint", slice(3, 32), 29),),
)

VELOCITY_SPEC = FieldSpec(
    name="vel",
    sample_columns=66,
    input_dim=64,
    root_target_index=64,
    joint_target_index=65,
    stat_groups=(
        StatGroup("pose_joint", slice(3, 32), 29),
        StatGroup("root_velocity", slice(32, 35), 3),
        StatGroup("joint_velocity", slice(35, 64), 29),
    ),
)

ACCELERATION_SPEC = FieldSpec(
    name="acc",
    sample_columns=98,
    input_dim=96,
    root_target_index=96,
    joint_target_index=97,
    stat_groups=(
        StatGroup("pose_joint", slice(3, 32), 29),
        StatGroup("root_velocity", slice(32, 35), 3),
        StatGroup("joint_velocity", slice(35, 64), 29),
        StatGroup("root_acceleration", slice(64, 67), 3),
        StatGroup("joint_acceleration", slice(67, 96), 29),
    ),
)


FIELD_SPECS = {
    "pose": POSE_SPEC,
    "vel": VELOCITY_SPEC,
    "acc": ACCELERATION_SPEC,
}


def get_field_spec(name: str) -> FieldSpec:
    normalized = str(name).strip().lower()
    try:
        return FIELD_SPECS[normalized]
    except KeyError as error:
        raise ValueError(
            f"Unsupported field {name!r}; choose one of {sorted(FIELD_SPECS)}"
        ) from error
