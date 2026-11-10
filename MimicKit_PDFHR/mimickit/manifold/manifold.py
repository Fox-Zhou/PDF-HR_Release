import torch

from .model import load_pose_distance_field


class PoseField:
    NAME = "posefield"
    INPUT_SIZE = 32
    ROOT_ROT_SIZE = 4
    DOF_SIZE = 29

    DEFAULT_WEIGHT = 0.2
    DEFAULT_SCALE = 1.0
    DEFAULT_MODE = "static"
    DEFAULT_TOLERANCE = 0.05
    DEFAULT_BAD_DISTANCE = 0.4
    DEFAULT_MAX_GOOD_DISTANCE = 0.395
    DEFAULT_REFERENCE_BATCH_SIZE = 1024

    def __init__(self, config, device):
        config = {} if config is None else config
        self._enable_reward = bool(config.get("enable_reward", False))
        self._device = device
        self._model = None
        self._checkpoint = None
        self._static_good_distance = None

        if not self._enable_reward:
            return

        self._model_path = self._require(config, "model_path")
        self._weight = float(config.get("weight", self.DEFAULT_WEIGHT))
        self._scale = float(config.get("scale", self.DEFAULT_SCALE))
        self._mode = str(config.get("mode", self.DEFAULT_MODE)).lower()
        self._tolerance = float(
            config.get("tolerance", self.DEFAULT_TOLERANCE)
        )
        self._bad_distance = float(
            config.get("bad_distance", self.DEFAULT_BAD_DISTANCE)
        )
        self._max_good_distance = float(
            config.get("max_good_distance", self.DEFAULT_MAX_GOOD_DISTANCE)
        )
        self._reference_batch_size = int(
            config.get(
                "reference_batch_size", self.DEFAULT_REFERENCE_BATCH_SIZE
            )
        )

        if self._mode not in ("static", "dynamic"):
            raise ValueError(
                "posefield.mode must be either 'static' or 'dynamic', got "
                f"{self._mode!r}"
            )
        if self._weight < 0.0:
            raise ValueError("posefield.weight must be non-negative")
        if self._scale < 0.0:
            raise ValueError("posefield.scale must be non-negative")
        if self._reference_batch_size <= 0:
            raise ValueError("posefield.reference_batch_size must be positive")
        if not self._max_good_distance < self._bad_distance:
            raise ValueError(
                "posefield.max_good_distance must be smaller than "
                "posefield.bad_distance"
            )

        self._model, self._checkpoint = load_pose_distance_field(
            self._model_path, device=device
        )
        self._model.eval()
        for param in self._model.parameters():
            param.requires_grad_(False)

    def is_enabled(self):
        return self._enable_reward

    def is_static(self):
        return self._mode == "static"

    def initialize_reference(self, root_rot, dof_pos):
        if not self._enable_reward or not self.is_static():
            return

        features = self._build_features(root_rot, dof_pos, "reference motion")
        max_distance = 0.0
        with torch.no_grad():
            for start in range(0, features.shape[0], self._reference_batch_size):
                end = start + self._reference_batch_size
                distance = self._model(features[start:end])
                max_distance = max(max_distance, distance.max().item())

        self._static_good_distance = min(
            self._max_good_distance,
            max_distance + self._tolerance,
        )

    def compute_reward(
        self,
        root_rot,
        dof_pos,
        ref_root_rot=None,
        ref_dof_pos=None,
    ):
        if not self._enable_reward:
            raise RuntimeError("Pose field reward is disabled")

        features = self._build_features(root_rot, dof_pos, "character")
        with torch.no_grad():
            distance = self._model(features).view(-1)

            if self.is_static():
                if self._static_good_distance is None:
                    raise RuntimeError(
                        "Pose field static reference distance has not been initialized"
                    )
                good_distance = self._static_good_distance
            else:
                if ref_root_rot is None or ref_dof_pos is None:
                    raise ValueError(
                        "posefield.mode='dynamic' requires a complete reference pose"
                    )
                ref_features = self._build_features(
                    ref_root_rot, ref_dof_pos, "dynamic reference"
                )
                ref_distance = self._model(ref_features).view(-1)
                good_distance = torch.clamp(
                    ref_distance + self._tolerance,
                    min=0.0,
                    max=self._max_good_distance,
                )

            normalized_distance = (
                (distance - good_distance)
                / (self._bad_distance - good_distance)
            )
            normalized_distance = torch.clamp(normalized_distance, 0.0, 1.0)
            reward = self._weight * torch.exp(
                -self._scale * normalized_distance
            )
        return reward

    def _build_features(self, root_rot, dof_pos, source):
        self._validate_tensor(root_rot, self.ROOT_ROT_SIZE, source, "root rotation")
        self._validate_tensor(dof_pos, self.DOF_SIZE, source, "DOF pose")
        if root_rot.shape[:-1] != dof_pos.shape[:-1]:
            raise ValueError(
                f"Pose field {source} root rotation and DOF pose batch shapes differ: "
                f"{tuple(root_rot.shape)} vs {tuple(dof_pos.shape)}"
            )

        root_gravity = self._root_gravity(root_rot)
        features = torch.cat((root_gravity, dof_pos), dim=-1)
        if features.ndim != 2 or features.shape[-1] != self.INPUT_SIZE:
            raise ValueError(
                f"Pose field expects a flat (B, {self.INPUT_SIZE}) input, got "
                f"{tuple(features.shape)} from {source}"
            )
        return features

    @staticmethod
    def _root_gravity(quat):
        """Return physical gravity in the root body frame: R(q)^T [0,0,-1]."""
        norm = torch.linalg.vector_norm(quat, dim=-1, keepdim=True)
        if torch.any(norm < 1e-8):
            raise ValueError("Pose field input contains a near-zero root quaternion")
        quat = quat / norm
        x, y, z, w = quat.unbind(dim=-1)
        gravity = torch.stack(
            (
                -2.0 * (x * z - w * y),
                -2.0 * (y * z + w * x),
                -1.0 + 2.0 * (x * x + y * y),
            ),
            dim=-1,
        )
        return gravity / torch.linalg.vector_norm(
            gravity, dim=-1, keepdim=True
        ).clamp_min(1e-8)

    @staticmethod
    def _validate_tensor(value, width, source, name):
        if value.ndim != 2 or value.shape[-1] != width:
            raise ValueError(
                f"Pose field expects {source} {name} with shape (B, {width}), "
                f"got {tuple(value.shape)}"
            )
        if not torch.isfinite(value).all():
            raise ValueError(f"Pose field {source} {name} contains NaN or infinity")

    @staticmethod
    def _require(config, key):
        if key not in config:
            raise KeyError(
                f"Missing required PoseField config field: posefield.{key}"
            )
        return config[key]


class Manifold:
    """Unified entry point for pose, velocity, and acceleration fields.

    Only posefield is registered for now.  Velfield and accfield will be added
    here without changing the environment-facing API.
    """

    def __init__(self, config, device):
        config = {} if config is None else config
        unknown = sorted(set(config) - {PoseField.NAME})
        if unknown:
            raise KeyError(
                "Unknown manifold fields: {}".format(", ".join(unknown))
            )

        self._fields = {
            PoseField.NAME: PoseField(config.get(PoseField.NAME, {}), device)
        }
        self._fields = {
            name: field for name, field in self._fields.items()
            if field.is_enabled()
        }

    def is_enabled(self):
        return len(self._fields) > 0

    def get_reward_component_names(self):
        return tuple(self._fields)

    def has_static_fields(self):
        return any(field.is_static() for field in self._fields.values())

    def initialize_reference(self, *, root_rot, dof_pos):
        for field in self._fields.values():
            field.initialize_reference(root_rot, dof_pos)

    def compute_rewards(
        self,
        *,
        root_rot,
        dof_pos,
        ref_root_rot=None,
        ref_dof_pos=None,
    ):
        return {
            name: field.compute_reward(
                root_rot=root_rot,
                dof_pos=dof_pos,
                ref_root_rot=ref_root_rot,
                ref_dof_pos=ref_dof_pos,
            )
            for name, field in self._fields.items()
        }
