import os
import xml.etree.ElementTree as ET

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
    [[0, 1, 5], [0, 5, 6], [1, 2, 3], [1, 3, 5], [3, 4, 5]],
    dtype=np.int32,
)


class ViserDribbleEnv(ViserBaseEnv):
    """Viser scene additions for :class:`TaskDribbleEnv`."""

    def _init_task_viser(self, env_config):
        self._viser_ball_handle = None
        self._viser_target_velocity_handle = None
        self._viser_ball_velocity_handle = None
        self._viser_ball_velocity_scale = float(
            env_config.get("viser_ball_velocity_scale", 0.4)
        )
        self._viser_ball_radius = self._read_ball_asset_radius()
        self._viser_show_ball = self.server.gui.add_checkbox(
            label="Dribble ball", initial_value=True
        )
        self._viser_show_target_velocity = self.server.gui.add_checkbox(
            label="Target ball velocity", initial_value=True
        )
        self._viser_show_ball_velocity = self.server.gui.add_checkbox(
            label="Actual ball velocity", initial_value=True
        )
        return

    def _build_task_viser(self):
        self._viser_ball_handle = self.server.scene.add_icosphere(
            name="/dribble/ball",
            radius=self._viser_ball_radius,
            color=self._to_rgb255(self._ball_color),
            position=(0.0, 0.0, -10.0),
            material="standard",
            cast_shadow=True,
            receive_shadow=True,
        )
        self._viser_target_velocity_handle = self.server.scene.add_mesh_simple(
            name="/dribble/target_velocity",
            vertices=_ARROW_VERTICES,
            faces=_ARROW_FACES,
            color=self._to_rgb255(self._velocity_marker_color),
            material="standard",
            side="double",
            cast_shadow=False,
            receive_shadow=False,
            position=(0.0, 0.0, -10.0),
        )
        return

    def _read_ball_asset_radius(self):
        """Read the physical sphere radius so Viser matches collision geometry."""
        asset_file = str(self._ball_asset_file)
        if asset_file.lower().endswith(".xml") and os.path.isfile(asset_file):
            root = ET.parse(asset_file).getroot()
            sphere = root.find(".//geom[@type='sphere']")
            if sphere is not None:
                size = sphere.get("size")
                if size:
                    return float(size.replace(",", " ").split()[0])
        return float(self._ball_radius)

    def _update_task_viser(self):
        env_id = self._viser_env_id
        ball_id = self._get_ball_id()
        ball_pos = self._engine.get_root_pos(ball_id)[env_id]
        ball_vel = self._engine.get_root_vel(ball_id)[env_id]
        ball_pos = ball_pos.detach().cpu().numpy().astype(np.float32)
        ball_vel = ball_vel.detach().cpu().numpy().astype(np.float32)
        target_dir = self._tar_dir[env_id].detach().cpu().numpy().astype(np.float32)
        target_speed = float(self._tar_speed[env_id].detach().cpu().item())

        self._viser_ball_handle.position = ball_pos
        self._viser_ball_handle.visible = bool(self._viser_show_ball.value)

        if self._tar_speed_max > self._tar_speed_min:
            speed_alpha = (target_speed - self._tar_speed_min) / (
                self._tar_speed_max - self._tar_speed_min
            )
        else:
            speed_alpha = 0.0
        marker_dist = self._velocity_marker_dist * (
            0.75 + 0.5 * float(np.clip(speed_alpha, 0.0, 1.0))
        )
        marker_pos = np.asarray(
            [
                ball_pos[0] + marker_dist * target_dir[0],
                ball_pos[1] + marker_dist * target_dir[1],
                self._velocity_marker_height,
            ],
            dtype=np.float32,
        )
        self._viser_target_velocity_handle.position = marker_pos
        self._viser_target_velocity_handle.wxyz = self._direction_to_wxyz(target_dir)
        self._viser_target_velocity_handle.visible = bool(
            self._viser_show_target_velocity.value
        )

        velocity_end = ball_pos + self._viser_ball_velocity_scale * ball_vel
        self._replace_ball_velocity_line(
            ball_pos,
            velocity_end,
            visible=bool(self._viser_show_ball_velocity.value),
        )
        return

    def _replace_ball_velocity_line(self, start, end, visible):
        if self._viser_ball_velocity_handle is not None:
            self._viser_ball_velocity_handle.remove()
        self._viser_ball_velocity_handle = self.server.scene.add_line_segments(
            name="/dribble/actual_velocity",
            points=np.asarray([[start, end]], dtype=np.float32),
            colors=(255, 210, 40),
            line_width=2.0,
            visible=visible,
        )
        return

    @staticmethod
    def _direction_to_wxyz(direction):
        theta = float(np.arctan2(direction[1], direction[0]))
        half_theta = 0.5 * theta
        return np.asarray(
            [np.cos(half_theta), 0.0, 0.0, np.sin(half_theta)],
            dtype=np.float32,
        )

    @staticmethod
    def _to_rgb255(color):
        color = np.asarray(color, dtype=np.float32)
        if np.max(color) <= 1.0:
            color = 255.0 * color
        return tuple(np.clip(np.rint(color[:3]), 0, 255).astype(np.uint8).tolist())

    def _get_rollout_task_state(self, env_id):
        ball_id = self._get_ball_id()
        ball_pos = self._engine.get_root_pos(ball_id)[env_id]
        ball_vel = self._engine.get_root_vel(ball_id)[env_id]
        return {
            "ball_pos": ball_pos,
            "ball_vel": ball_vel,
            "target_dir": self._tar_dir[env_id],
            "target_speed": self._tar_speed[env_id],
        }
