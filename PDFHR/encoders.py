from __future__ import annotations

from copy import deepcopy
from typing import Any, Sequence

import torch
import torch.nn as nn


G1_PARENT_MAPPING = [
    12, 0, 1, 2, 3, 4,
    12, 6, 7, 8, 9, 10,
    -1, 12, 13,
    14, 15, 16, 17, 18, 19, 20,
    14, 22, 23, 24, 25, 26, 27,
]


def merge_strict(
    default: dict[str, Any], user: dict[str, Any] | None, path: str
) -> dict[str, Any]:
    result = deepcopy(default)
    if user is None:
        return result
    if not isinstance(user, dict):
        raise TypeError(f"{path} must be an object")
    unknown = sorted(set(user) - set(default))
    if unknown:
        raise KeyError(f"Unknown {path} fields: {', '.join(unknown)}")
    for key, value in user.items():
        if isinstance(default[key], dict):
            result[key] = merge_strict(default[key], value, f"{path}.{key}")
        else:
            result[key] = value
    return result


def int_list(value: Any, name: str) -> list[int]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list of positive integers")
    result = [int(item) for item in value]
    if any(item <= 0 for item in result):
        raise ValueError(f"{name} must contain only positive integers")
    return result


def activation(name: str, beta: float = 100.0) -> nn.Module:
    normalized = str(name).lower()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "silu":
        return nn.SiLU()
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "softplus":
        return nn.Softplus(beta=float(beta))
    if normalized == "lrelu":
        return nn.LeakyReLU()
    raise ValueError(f"Unsupported activation: {name!r}")


def norm_layer(name: str | None, width: int) -> nn.Module:
    if name is None or str(name).lower() == "none":
        return nn.Identity()
    if str(name).lower() == "layernorm":
        return nn.LayerNorm(width)
    raise ValueError(f"Unsupported normalization: {name!r}")


class ConfigurableMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        *,
        activation_name: str,
        norm: str | None,
        dropout: float,
        beta: float,
        activate_output: bool,
        output_activation: str | None = None,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("MLP input_dim and output_dim must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        dims = [int(input_dim), *hidden_dims, int(output_dim)]
        layers: list[nn.Module] = []
        for index in range(len(dims) - 1):
            is_output = index == len(dims) - 2
            layers.append(nn.Linear(dims[index], dims[index + 1]))
            if not is_output or activate_output:
                layers.append(norm_layer(norm, dims[index + 1]))
                layers.append(activation(activation_name, beta))
                if dropout > 0:
                    layers.append(nn.Dropout(float(dropout)))
            elif output_activation is not None:
                layers.append(activation(output_activation, beta))
        self.net = nn.Sequential(*layers)
        self.output_dim = int(output_dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


def _encoder_mlp(
    input_dim: int, cfg: dict[str, Any], path: str
) -> ConfigurableMLP:
    return ConfigurableMLP(
        input_dim,
        int_list(cfg["hidden_dims"], f"{path}.hidden_dims"),
        int(cfg["output_dim"]),
        activation_name=cfg["activation"],
        norm=cfg["norm"],
        dropout=float(cfg["dropout"]),
        beta=float(cfg["beta"]),
        activate_output=True,
    )


class RootTiltEncoder(nn.Module):
    def __init__(self, cfg: dict[str, Any], path: str = "root_encoder") -> None:
        super().__init__()
        self.net = _encoder_mlp(3, cfg, path)
        self.output_dim = self.net.output_dim

    def forward(self, tilt_xyz: torch.Tensor) -> torch.Tensor:
        if tilt_xyz.ndim != 2 or tilt_xyz.shape[1] != 3:
            raise ValueError(
                f"RootTiltEncoder expects (B, 3), got {tuple(tilt_xyz.shape)}"
            )
        return self.net(tilt_xyz)


class ConditionalRootVectorEncoder(nn.Module):
    def __init__(
        self,
        cfg: dict[str, Any],
        condition_dims: Sequence[int],
        path: str,
    ) -> None:
        super().__init__()
        self.condition_dims = tuple(int(width) for width in condition_dims)
        if any(width <= 0 for width in self.condition_dims):
            raise ValueError("Root condition dimensions must be positive")
        self.net = _encoder_mlp(3 + sum(self.condition_dims), cfg, path)
        self.output_dim = self.net.output_dim

    def forward(
        self, vector_xyz: torch.Tensor, *conditions: torch.Tensor
    ) -> torch.Tensor:
        if vector_xyz.ndim != 2 or vector_xyz.shape[1] != 3:
            raise ValueError(
                f"Root vector encoder expects (B, 3), got {tuple(vector_xyz.shape)}"
            )
        if len(conditions) != len(self.condition_dims):
            raise ValueError(
                f"Expected {len(self.condition_dims)} root conditions, got {len(conditions)}"
            )
        values = [vector_xyz]
        for condition, width in zip(conditions, self.condition_dims):
            if condition.ndim != 2 or condition.shape != (len(vector_xyz), width):
                raise ValueError(
                    f"Root condition must have shape (B, {width}), "
                    f"got {tuple(condition.shape)}"
                )
            values.append(condition)
        return self.net(torch.cat(values, dim=-1))


class JointNodeEncoder(nn.Module):
    def __init__(
        self, input_dim: int, local_dim: int, cfg: dict[str, Any], path: str
    ) -> None:
        super().__init__()
        self.net = ConfigurableMLP(
            input_dim,
            int_list(cfg["per_joint_hidden_dims"], f"{path}.per_joint_hidden_dims"),
            local_dim,
            activation_name=cfg["activation"],
            norm=cfg["norm"],
            dropout=float(cfg["dropout"]),
            beta=float(cfg["beta"]),
            activate_output=True,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class HierarchicalJointEncoder(nn.Module):
    """Encode one scalar per joint with same-joint conditions and parent features."""

    def __init__(
        self,
        cfg: dict[str, Any],
        *,
        root_parent_dim: int,
        condition_dims: Sequence[int] = (),
        path: str = "joint_encoder",
    ) -> None:
        super().__init__()
        mapping = cfg["parent_mapping"]
        if mapping == "g1":
            self.parent_mapping = list(G1_PARENT_MAPPING)
        else:
            if not isinstance(mapping, (list, tuple)):
                raise TypeError(f"{path}.parent_mapping must be 'g1' or a list")
            self.parent_mapping = [int(value) for value in mapping]
        self.num_joints = len(self.parent_mapping)
        self.local_dim = int(cfg["local_output_dim"])
        self.root_parent_dim = int(root_parent_dim)
        self.condition_dims = tuple(int(width) for width in condition_dims)
        if self.local_dim <= 0 or self.root_parent_dim < 0:
            raise ValueError(
                "Joint local dimension must be positive and root-parent dimension "
                "must be non-negative"
            )
        if any(width <= 0 for width in self.condition_dims):
            raise ValueError("Joint condition dimensions must be positive")
        self.topological_order = self._topological_order()
        condition_width = sum(self.condition_dims)
        self.nodes = nn.ModuleList(
            [
                JointNodeEncoder(
                    1
                    + condition_width
                    + (self.root_parent_dim if parent == -1 else self.local_dim),
                    self.local_dim,
                    cfg,
                    path,
                )
                for parent in self.parent_mapping
            ]
        )
        self.output_dim = self.num_joints * self.local_dim

    def _topological_order(self) -> list[int]:
        count = len(self.parent_mapping)
        children = {index: [] for index in range(count)}
        roots: list[int] = []
        for index, parent in enumerate(self.parent_mapping):
            if parent == -1:
                roots.append(index)
            elif parent < 0 or parent >= count:
                raise ValueError(f"Invalid parent {parent} for joint {index}")
            else:
                children[parent].append(index)
        order: list[int] = []
        queue = list(roots)
        while queue:
            current = queue.pop(0)
            order.append(current)
            queue.extend(children[current])
        if len(order) != count:
            raise ValueError("joint parent mapping must form an acyclic forest")
        return order

    def forward(
        self,
        joint_values: torch.Tensor,
        root_parent_feature: torch.Tensor | None,
        *joint_conditions: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = len(joint_values)
        if joint_values.ndim != 2 or joint_values.shape[1] != self.num_joints:
            raise ValueError(
                f"Joint encoder expects (B, {self.num_joints}), "
                f"got {tuple(joint_values.shape)}"
            )
        if self.root_parent_dim == 0:
            if root_parent_feature is not None:
                raise ValueError("Root parent feature must be None without a root encoder")
        elif root_parent_feature is None or root_parent_feature.ndim != 2 or (
            root_parent_feature.shape != (batch_size, self.root_parent_dim)
        ):
            shape = None if root_parent_feature is None else tuple(root_parent_feature.shape)
            raise ValueError(
                f"Root parent feature must have shape (B, {self.root_parent_dim}), "
                f"got {shape}"
            )
        if len(joint_conditions) != len(self.condition_dims):
            raise ValueError(
                f"Expected {len(self.condition_dims)} joint conditions, "
                f"got {len(joint_conditions)}"
            )
        for condition, width in zip(joint_conditions, self.condition_dims):
            if condition.ndim != 3 or condition.shape != (
                batch_size,
                self.num_joints,
                width,
            ):
                raise ValueError(
                    f"Joint condition must have shape "
                    f"(B, {self.num_joints}, {width}), got {tuple(condition.shape)}"
                )

        features: list[torch.Tensor | None] = [None] * self.num_joints
        for index in self.topological_order:
            parent = self.parent_mapping[index]
            values = [joint_values[:, index : index + 1]]
            values.extend(condition[:, index, :] for condition in joint_conditions)
            if parent == -1:
                if root_parent_feature is not None:
                    values.append(root_parent_feature)
            else:
                parent_feature = features[parent]
                if parent_feature is None:
                    raise RuntimeError(f"Parent feature for joint {index} is unavailable")
                values.append(parent_feature)
            features[index] = self.nodes[index](torch.cat(values, dim=-1))
        return torch.stack(
            [feature for feature in features if feature is not None], dim=1
        )


class ConcatFusion(nn.Module):
    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        if not features:
            raise ValueError("ConcatFusion requires at least one feature tensor")
        return torch.cat(features, dim=-1)
