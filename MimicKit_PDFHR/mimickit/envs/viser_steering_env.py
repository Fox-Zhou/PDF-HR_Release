import numpy as np

from envs.viser_base_env import ViserBaseEnv


_ARROW_VERTICES = np.asarray(
    [
        [-0.25, -0.035, 0.0],
        [0.08, -0.035, 0.0],
        [0.08, -0.10, 0.0],
        [0.25, 0.0, 0.0],
        [0.08, 0.10, 0.0],
        [0.08, 0.035, 0.0],
        [-0.25, 0.035, 0.0],
    ],
    dtype=np.float32,
)
_ARROW_FACES = np.asarray(
    [
        [0, 1, 5],
        [0, 5, 6],
        [1, 2, 3],
        [1, 3, 5],
        [3, 4, 5],
    ],
    dtype=np.int32,
)


class ViserSteeringEnv(ViserBaseEnv):
    """Viser scene additions for :class:`TaskSteeringEnv`."""

    def _init_task_viser(self, env_config):
        self._viser_move_arrow = None
        self._viser_face_arrow = None
        self._viser_show_move_dir = self.server.gui.add_checkbox(
            label="Move direction",
            initial_value=True,
        )
        self._viser_show_face_dir = self.server.gui.add_checkbox(
            label="Facing direction",
            initial_value=True,
        )
        return

    def _build_task_viser(self):
        self._viser_move_arrow = self._add_direction_arrow(
            name="/steering/move_direction",
            color=(220, 35, 35),
        )
        self._viser_face_arrow = self._add_direction_arrow(
            name="/steering/facing_direction",
            color=(35, 200, 70),
        )
        return

    def _add_direction_arrow(self, name, color):
        return self.server.scene.add_mesh_simple(
            name=name,
            vertices=_ARROW_VERTICES,
            faces=_ARROW_FACES,
            color=color,
            material="standard",
            side="double",
            cast_shadow=False,
            receive_shadow=False,
            position=(0.0, 0.0, -10.0),
        )

    def _update_task_viser(self):
        env_id = self._viser_env_id
        root_pos = self._engine.get_root_pos(self._get_char_id())[env_id]
        root_pos = root_pos.detach().cpu().numpy().astype(np.float32)
        move_dir = self._tar_dir[env_id].detach().cpu().numpy().astype(np.float32)
        face_dir = self._face_dir[env_id].detach().cpu().numpy().astype(np.float32)
        target_speed = float(self._tar_speed[env_id].detach().cpu().item())

        speed_range = float(self._tar_speed_max - self._tar_speed_min)
        if speed_range > 1e-8:
            speed_ratio = (target_speed - self._tar_speed_min) / speed_range
        else:
            speed_ratio = 0.0
        move_dist = 1.0 + 0.5 * float(np.clip(speed_ratio, 0.0, 1.0))

        move_pos = np.array(
            [
                root_pos[0] + move_dist * move_dir[0],
                root_pos[1] + move_dist * move_dir[1],
                0.04,
            ],
            dtype=np.float32,
        )
        face_pos = np.array(
            [
                root_pos[0] + face_dir[0],
                root_pos[1] + face_dir[1],
                0.05,
            ],
            dtype=np.float32,
        )

        self._viser_move_arrow.position = move_pos
        self._viser_move_arrow.wxyz = self._direction_to_wxyz(move_dir)
        self._viser_move_arrow.visible = bool(self._viser_show_move_dir.value)

        self._viser_face_arrow.position = face_pos
        self._viser_face_arrow.wxyz = self._direction_to_wxyz(face_dir)
        self._viser_face_arrow.visible = bool(self._viser_show_face_dir.value)
        return

    @staticmethod
    def _direction_to_wxyz(direction):
        theta = float(np.arctan2(direction[1], direction[0]))
        half_theta = 0.5 * theta
        return np.asarray(
            [np.cos(half_theta), 0.0, 0.0, np.sin(half_theta)],
            dtype=np.float32,
        )

    def _get_rollout_task_state(self, env_id):
        return {
            "target_dir": self._tar_dir[env_id],
            "face_dir": self._face_dir[env_id],
            "target_speed": self._tar_speed[env_id],
        }
