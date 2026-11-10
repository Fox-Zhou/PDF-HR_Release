from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import pickle
from typing import Any

import torch
import torch.nn as nn

from .encoders import (
    ConcatFusion,
    ConditionalRootVectorEncoder,
    ConfigurableMLP,
    HierarchicalJointEncoder,
    RootTiltEncoder,
    int_list,
    merge_strict,
)
from .field_specs import FEATURE_LAYOUT


ROOT_ENCODER_CFG: dict[str, Any] = {
    "hidden_dims": [32, 16],
    "output_dim": 6,
    "activation": "silu",
    "norm": "layernorm",
    "dropout": 0.0,
    "beta": 100.0,
}

JOINT_ENCODER_CFG: dict[str, Any] = {
    "parent_mapping": "g1",
    "per_joint_hidden_dims": [16],
    "local_output_dim": 6,
    "activation": "silu",
    "norm": "layernorm",
    "dropout": 0.0,
    "beta": 100.0,
}

HEAD_CFG: dict[str, Any] = {
    "hidden_dims": [256, 128, 64],
    "activation": "softplus",
    "output_activation": "softplus",
    "norm": None,
    "dropout": 0.0,
    "beta": 100.0,
}


def _stage_cfg() -> dict[str, Any]:
    return {
        "root_encoder": deepcopy(ROOT_ENCODER_CFG),
        "joint_encoder": deepcopy(JOINT_ENCODER_CFG),
    }


DEFAULT_POSE_MODEL_CFG: dict[str, Any] = {
    **_stage_cfg(),
    "fusion": {"type": "concat"},
    "head": deepcopy(HEAD_CFG),
}

DEFAULT_VELOCITY_MODEL_CFG: dict[str, Any] = {
    "pose_encoder": _stage_cfg(),
    "velocity_encoder": _stage_cfg(),
    "fusion": {"type": "concat"},
    "head": {**deepcopy(HEAD_CFG), "hidden_dims": [512, 256, 128, 64]},
}

DEFAULT_ACCELERATION_MODEL_CFG: dict[str, Any] = {
    "pose_encoder": _stage_cfg(),
    "velocity_encoder": _stage_cfg(),
    "acceleration_encoder": _stage_cfg(),
    "fusion": {"type": "concat"},
    "head": {**deepcopy(HEAD_CFG), "hidden_dims": [512, 256, 128, 64]},
}

DEFAULT_MODEL_CFGS = {
    "pose": DEFAULT_POSE_MODEL_CFG,
    "vel": DEFAULT_VELOCITY_MODEL_CFG,
    "acc": DEFAULT_ACCELERATION_MODEL_CFG,
}


def _model_stage_names(field: str) -> tuple[str | None, ...]:
    normalized = str(field).lower()
    if normalized == "pose":
        return (None,)
    if normalized == "vel":
        return ("pose_encoder", "velocity_encoder")
    if normalized == "acc":
        return ("pose_encoder", "velocity_encoder", "acceleration_encoder")
    raise ValueError(f"Unsupported field {field!r}")


def validate_root_encoder_config(
    field: str, root: bool, model_cfg: dict[str, Any] | None
) -> None:
    """Reject explicit root-encoder settings that contradict the target mode."""
    if model_cfg is None:
        return
    if not isinstance(model_cfg, dict):
        raise TypeError("model must be an object")
    for stage_name in _model_stage_names(field):
        stage_cfg = model_cfg if stage_name is None else model_cfg.get(stage_name)
        if stage_cfg is None or not isinstance(stage_cfg, dict):
            continue
        if "root_encoder" not in stage_cfg:
            continue
        has_root_encoder = stage_cfg["root_encoder"] is not None
        if has_root_encoder != bool(root):
            path = "model.root_encoder" if stage_name is None else f"model.{stage_name}.root_encoder"
            expected = "an encoder object" if root else "null or omission"
            raise ValueError(
                f"Config root={bool(root)} conflicts with {path}; expected {expected}"
            )


def get_default_model_cfg(field: str, root: bool = True) -> dict[str, Any]:
    normalized = str(field).lower()
    try:
        result = deepcopy(DEFAULT_MODEL_CFGS[normalized])
    except KeyError as error:
        raise ValueError(f"Unsupported field {field!r}") from error
    if not root:
        for stage_name in _model_stage_names(normalized):
            stage_cfg = result if stage_name is None else result[stage_name]
            stage_cfg["root_encoder"] = None
    return result


def normalize_tilt_vector(tilt: torch.Tensor) -> torch.Tensor:
    if tilt.ndim != 2 or tilt.shape[1] != 3:
        raise ValueError(f"Expected root tilt shape (B, 3), got {tuple(tilt.shape)}")
    norm = torch.linalg.vector_norm(tilt, dim=-1, keepdim=True)
    if torch.any(norm < 1e-8):
        raise ValueError("Input contains a near-zero root tilt vector")
    return tilt / norm


def _flat_stage(root: torch.Tensor | None, joints: torch.Tensor) -> torch.Tensor:
    flat_joints = joints.flatten(start_dim=1)
    return flat_joints if root is None else torch.cat((root, flat_joints), dim=-1)


class FeatureNormalizedField(nn.Module):
    def __init__(
        self,
        feature_stats: dict[str, dict[str, Any]],
        required_stats: dict[str, int],
    ) -> None:
        super().__init__()
        self.stat_names = tuple(required_stats)
        unknown = sorted(set(feature_stats) - set(required_stats))
        missing = sorted(set(required_stats) - set(feature_stats))
        if unknown or missing:
            raise ValueError(
                f"Feature-stat groups mismatch; missing={missing}, unknown={unknown}"
            )
        for name, width in required_stats.items():
            group = feature_stats[name]
            if not isinstance(group, dict) or set(group) != {"mean", "std"}:
                raise ValueError(f"Feature-stat group {name} must contain mean and std")
            mean = torch.as_tensor(group["mean"], dtype=torch.float32)
            std = torch.as_tensor(group["std"], dtype=torch.float32)
            if mean.shape != (width,) or std.shape != (width,):
                raise ValueError(
                    f"Feature-stat group {name} must have width {width}, "
                    f"got mean={tuple(mean.shape)}, std={tuple(std.shape)}"
                )
            if (
                not torch.isfinite(mean).all()
                or not torch.isfinite(std).all()
                or torch.any(std <= 0)
            ):
                raise ValueError(f"Feature-stat group {name} is invalid")
            self.register_buffer(f"{name}_mean", mean.clone())
            self.register_buffer(f"{name}_std", std.clone())

    def normalize_group(self, name: str, value: torch.Tensor) -> torch.Tensor:
        return (value - getattr(self, f"{name}_mean")) / getattr(
            self, f"{name}_std"
        )

    def export_feature_stats(self) -> dict[str, dict[str, list[float]]]:
        return {
            name: {
                "mean": getattr(self, f"{name}_mean").detach().cpu().tolist(),
                "std": getattr(self, f"{name}_std").detach().cpu().tolist(),
            }
            for name in self.stat_names
        }


class PoseStageEncoder(nn.Module):
    def __init__(self, cfg: dict[str, Any], path: str) -> None:
        super().__init__()
        root_cfg = cfg["root_encoder"]
        self.root = (
            None if root_cfg is None else RootTiltEncoder(root_cfg, f"{path}.root_encoder")
        )
        self.joints = HierarchicalJointEncoder(
            cfg["joint_encoder"],
            root_parent_dim=self.root_output_dim,
            path=f"{path}.joint_encoder",
        )

    def forward(
        self, root_tilt: torch.Tensor, joint_positions: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        root_feature = (
            None if self.root is None else self.root(normalize_tilt_vector(root_tilt))
        )
        joint_features = self.joints(joint_positions, root_feature)
        return root_feature, joint_features

    @property
    def root_output_dim(self) -> int:
        return 0 if self.root is None else self.root.output_dim

    @property
    def flat_dim(self) -> int:
        return self.root_output_dim + self.joints.output_dim


class VelocityStageEncoder(nn.Module):
    def __init__(
        self,
        cfg: dict[str, Any],
        pose_root_dim: int,
        path: str,
    ) -> None:
        super().__init__()
        root_cfg = cfg["root_encoder"]
        self.root = (
            None
            if root_cfg is None
            else ConditionalRootVectorEncoder(
                root_cfg, [pose_root_dim], f"{path}.root_encoder"
            )
        )
        self.joints = HierarchicalJointEncoder(
            cfg["joint_encoder"],
            root_parent_dim=self.root_output_dim,
            condition_dims=[1],
            path=f"{path}.joint_encoder",
        )

    def forward(
        self,
        root_velocity: torch.Tensor,
        joint_velocity: torch.Tensor,
        pose_root: torch.Tensor | None,
        joint_positions: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        if self.root is None:
            root_feature = None
        else:
            if pose_root is None:
                raise ValueError("Velocity root encoder requires a pose root feature")
            root_feature = self.root(root_velocity, pose_root)
        joint_features = self.joints(
            joint_velocity, root_feature, joint_positions.unsqueeze(-1)
        )
        return root_feature, joint_features

    @property
    def root_output_dim(self) -> int:
        return 0 if self.root is None else self.root.output_dim

    @property
    def flat_dim(self) -> int:
        return self.root_output_dim + self.joints.output_dim


class AccelerationStageEncoder(nn.Module):
    def __init__(
        self,
        cfg: dict[str, Any],
        pose_root_dim: int,
        velocity_root_dim: int,
        path: str,
    ) -> None:
        super().__init__()
        root_cfg = cfg["root_encoder"]
        self.root = (
            None
            if root_cfg is None
            else ConditionalRootVectorEncoder(
                root_cfg,
                [pose_root_dim, velocity_root_dim],
                f"{path}.root_encoder",
            )
        )
        self.joints = HierarchicalJointEncoder(
            cfg["joint_encoder"],
            root_parent_dim=self.root_output_dim,
            condition_dims=[1, 1],
            path=f"{path}.joint_encoder",
        )

    def forward(
        self,
        root_acceleration: torch.Tensor,
        joint_acceleration: torch.Tensor,
        pose_root: torch.Tensor | None,
        joint_positions: torch.Tensor,
        velocity_root: torch.Tensor | None,
        joint_velocities: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        if self.root is None:
            root_feature = None
        else:
            if pose_root is None or velocity_root is None:
                raise ValueError(
                    "Acceleration root encoder requires pose and velocity root features"
                )
            root_feature = self.root(root_acceleration, pose_root, velocity_root)
        joint_features = self.joints(
            joint_acceleration,
            root_feature,
            joint_velocities.unsqueeze(-1),
            joint_positions.unsqueeze(-1),
        )
        return root_feature, joint_features

    @property
    def root_output_dim(self) -> int:
        return 0 if self.root is None else self.root.output_dim

    @property
    def flat_dim(self) -> int:
        return self.root_output_dim + self.joints.output_dim


def _build_head(input_dim: int, cfg: dict[str, Any]) -> ConfigurableMLP:
    output_activation = str(cfg["output_activation"]).lower()
    if output_activation not in {"softplus", "relu"}:
        raise ValueError("head.output_activation must be 'softplus' or 'relu'")
    return ConfigurableMLP(
        input_dim,
        int_list(cfg["hidden_dims"], "head.hidden_dims"),
        1,
        activation_name=cfg["activation"],
        norm=cfg["norm"],
        dropout=float(cfg["dropout"]),
        beta=float(cfg["beta"]),
        activate_output=False,
        output_activation=output_activation,
    )


class PoseDistanceField(FeatureNormalizedField):
    field = "pose"
    input_dim = 32

    def __init__(
        self,
        model_cfg: dict[str, Any] | None,
        feature_stats: dict[str, dict[str, Any]],
        root: bool = True,
    ) -> None:
        super().__init__(feature_stats, {"pose_joint": 29})
        self.root = bool(root)
        validate_root_encoder_config(self.field, self.root, model_cfg)
        self.model_cfg = merge_strict(
            get_default_model_cfg(self.field, self.root), model_cfg, "model"
        )
        if self.model_cfg["fusion"]["type"] != "concat":
            raise ValueError("fusion.type must be 'concat'")
        self.pose_encoder = PoseStageEncoder(self.model_cfg, "pose_encoder")
        self.fusion = ConcatFusion()
        self.head = _build_head(self.pose_encoder.flat_dim, self.model_cfg["head"])

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        _validate_input(features, self.input_dim, self.field)
        pose_root, pose_joints = self.pose_encoder(
            features[:, 0:3],
            self.normalize_group("pose_joint", features[:, 3:32]),
        )
        fused = self.fusion(_flat_stage(pose_root, pose_joints))
        return self.head(fused).squeeze(-1)


class VelocityDistanceField(FeatureNormalizedField):
    field = "vel"
    input_dim = 64

    def __init__(
        self,
        model_cfg: dict[str, Any] | None,
        feature_stats: dict[str, dict[str, Any]],
        root: bool = True,
    ) -> None:
        super().__init__(
            feature_stats,
            {"pose_joint": 29, "root_velocity": 3, "joint_velocity": 29},
        )
        self.root = bool(root)
        validate_root_encoder_config(self.field, self.root, model_cfg)
        self.model_cfg = merge_strict(
            get_default_model_cfg(self.field, self.root), model_cfg, "model"
        )
        if self.model_cfg["fusion"]["type"] != "concat":
            raise ValueError("fusion.type must be 'concat'")
        self.pose_encoder = PoseStageEncoder(
            self.model_cfg["pose_encoder"], "pose_encoder"
        )
        self.velocity_encoder = VelocityStageEncoder(
            self.model_cfg["velocity_encoder"],
            self.pose_encoder.root_output_dim,
            "velocity_encoder",
        )
        self.fusion = ConcatFusion()
        self.head = _build_head(
            self.pose_encoder.flat_dim + self.velocity_encoder.flat_dim,
            self.model_cfg["head"],
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        _validate_input(features, self.input_dim, self.field)
        joint_positions = self.normalize_group("pose_joint", features[:, 3:32])
        pose_root, pose_joints = self.pose_encoder(
            features[:, 0:3],
            joint_positions,
        )
        root_velocity = features[:, 32:35]
        if self.velocity_encoder.root is not None:
            root_velocity = self.normalize_group("root_velocity", root_velocity)
        velocity_root, velocity_joints = self.velocity_encoder(
            root_velocity,
            self.normalize_group("joint_velocity", features[:, 35:64]),
            pose_root,
            joint_positions,
        )
        fused = self.fusion(
            _flat_stage(pose_root, pose_joints),
            _flat_stage(velocity_root, velocity_joints),
        )
        return self.head(fused).squeeze(-1)


class AccelerationDistanceField(FeatureNormalizedField):
    field = "acc"
    input_dim = 96

    def __init__(
        self,
        model_cfg: dict[str, Any] | None,
        feature_stats: dict[str, dict[str, Any]],
        root: bool = True,
    ) -> None:
        super().__init__(
            feature_stats,
            {
                "pose_joint": 29,
                "root_velocity": 3,
                "joint_velocity": 29,
                "root_acceleration": 3,
                "joint_acceleration": 29,
            },
        )
        self.root = bool(root)
        validate_root_encoder_config(self.field, self.root, model_cfg)
        self.model_cfg = merge_strict(
            get_default_model_cfg(self.field, self.root), model_cfg, "model"
        )
        if self.model_cfg["fusion"]["type"] != "concat":
            raise ValueError("fusion.type must be 'concat'")
        self.pose_encoder = PoseStageEncoder(
            self.model_cfg["pose_encoder"], "pose_encoder"
        )
        self.velocity_encoder = VelocityStageEncoder(
            self.model_cfg["velocity_encoder"],
            self.pose_encoder.root_output_dim,
            "velocity_encoder",
        )
        self.acceleration_encoder = AccelerationStageEncoder(
            self.model_cfg["acceleration_encoder"],
            self.pose_encoder.root_output_dim,
            self.velocity_encoder.root_output_dim,
            "acceleration_encoder",
        )
        self.fusion = ConcatFusion()
        self.head = _build_head(
            self.pose_encoder.flat_dim
            + self.velocity_encoder.flat_dim
            + self.acceleration_encoder.flat_dim,
            self.model_cfg["head"],
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        _validate_input(features, self.input_dim, self.field)
        joint_positions = self.normalize_group("pose_joint", features[:, 3:32])
        joint_velocities = self.normalize_group(
            "joint_velocity", features[:, 35:64]
        )
        pose_root, pose_joints = self.pose_encoder(
            features[:, 0:3],
            joint_positions,
        )
        root_velocity = features[:, 32:35]
        if self.velocity_encoder.root is not None:
            root_velocity = self.normalize_group("root_velocity", root_velocity)
        velocity_root, velocity_joints = self.velocity_encoder(
            root_velocity,
            joint_velocities,
            pose_root,
            joint_positions,
        )
        root_acceleration = features[:, 64:67]
        if self.acceleration_encoder.root is not None:
            root_acceleration = self.normalize_group(
                "root_acceleration", root_acceleration
            )
        acceleration_root, acceleration_joints = self.acceleration_encoder(
            root_acceleration,
            self.normalize_group("joint_acceleration", features[:, 67:96]),
            pose_root,
            joint_positions,
            velocity_root,
            joint_velocities,
        )
        fused = self.fusion(
            _flat_stage(pose_root, pose_joints),
            _flat_stage(velocity_root, velocity_joints),
            _flat_stage(acceleration_root, acceleration_joints),
        )
        return self.head(fused).squeeze(-1)


def _validate_input(features: torch.Tensor, width: int, field: str) -> None:
    if features.ndim != 2 or features.shape[1] != width:
        raise ValueError(
            f"{field} distance field expects (B, {width}), got {tuple(features.shape)}"
        )
    if not torch.isfinite(features).all():
        raise ValueError(f"{field} input contains NaN or infinity")


def build_distance_field(
    field: str,
    model_cfg: dict[str, Any] | None,
    feature_stats: dict[str, dict[str, Any]],
    device: torch.device | str | None = None,
    *,
    root: bool = True,
) -> FeatureNormalizedField:
    normalized = str(field).lower()
    classes = {
        "pose": PoseDistanceField,
        "vel": VelocityDistanceField,
        "acc": AccelerationDistanceField,
    }
    model_class = classes.get(normalized)
    if model_class is None:
        raise ValueError(f"Unsupported field {field!r}")
    model = model_class(model_cfg, feature_stats, root=root)
    if device is not None:
        model.to(device)
    return model


def build_pose_distance_field(
    model_cfg: dict[str, Any] | None = None,
    *,
    feature_stats: dict[str, dict[str, Any]] | None = None,
    joint_mean: torch.Tensor | list[float] | None = None,
    joint_std: torch.Tensor | list[float] | None = None,
    device: torch.device | str | None = None,
    root: bool = True,
) -> PoseDistanceField:
    if feature_stats is None:
        mean = torch.zeros(29) if joint_mean is None else torch.as_tensor(joint_mean)
        std = torch.ones(29) if joint_std is None else torch.as_tensor(joint_std)
        feature_stats = {"pose_joint": {"mean": mean, "std": std}}
    return build_distance_field(
        "pose", model_cfg, feature_stats, device, root=root
    )  # type: ignore[return-value]


def torch_load_safe(path: str | Path, map_location: torch.device | str = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except (TypeError, pickle.UnpicklingError):
        return torch.load(path, map_location=map_location)


def load_distance_field(
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
    expected_field: str | None = None,
) -> tuple[FeatureNormalizedField, dict[str, Any]]:
    # Training checkpoints include optimizer/scaler tensors. Deserialize on CPU
    # first so they do not temporarily consume simulation GPU memory.
    checkpoint = torch_load_safe(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a PDF-HR checkpoint dictionary")
    if not isinstance(checkpoint.get("root"), bool):
        raise ValueError("PDF-HR checkpoint must contain a boolean root field")
    if checkpoint.get("feature_layout") != FEATURE_LAYOUT:
        raise ValueError(
            f"Checkpoint feature layout must be {FEATURE_LAYOUT!r}; "
            f"got {checkpoint.get('feature_layout')!r}"
        )
    field = str(checkpoint.get("field", "")).lower()
    if expected_field is not None and field != str(expected_field).lower():
        raise ValueError(f"Expected {expected_field} checkpoint, got field={field!r}")
    model = build_distance_field(
        field,
        deepcopy(checkpoint["model_cfg"]),
        checkpoint["feature_stats"],
        None,
        root=checkpoint["root"],
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    return model, checkpoint


def load_pose_distance_field(
    checkpoint_path: str | Path, device: torch.device | str = "cpu"
) -> tuple[FeatureNormalizedField, dict[str, Any]]:
    return load_distance_field(checkpoint_path, device, "pose")


def load_velocity_distance_field(
    checkpoint_path: str | Path, device: torch.device | str = "cpu"
) -> tuple[FeatureNormalizedField, dict[str, Any]]:
    return load_distance_field(checkpoint_path, device, "vel")


def load_acceleration_distance_field(
    checkpoint_path: str | Path, device: torch.device | str = "cpu"
) -> tuple[FeatureNormalizedField, dict[str, Any]]:
    return load_distance_field(checkpoint_path, device, "acc")
