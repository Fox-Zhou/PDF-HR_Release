import numpy as np

from envs.viser_base_env import ViserBaseEnv


class ViserDodgeballEnv(ViserBaseEnv):
    """Viser scene additions for :class:`TaskDodgeballEnv`."""

    def _init_task_viser(self, env_config):
        self._viser_projectile_handles = []
        self._viser_velocity_handles = []
        self._viser_velocity_scale = float(
            env_config.get("viser_projectile_velocity_scale", 0.1)
        )
        self._viser_show_projectiles = self.server.gui.add_checkbox(
            label="Dodgeballs",
            initial_value=True,
        )
        self._viser_show_projectile_velocity = self.server.gui.add_checkbox(
            label="Dodgeball velocity",
            initial_value=True,
        )
        return

    def _build_task_viser(self):
        for proj_idx in range(self._get_num_projectiles()):
            ball_handle = self.server.scene.add_icosphere(
                name="/dodgeball/projectile_{:02d}/ball".format(proj_idx),
                radius=float(self._proj_radius),
                color=(248, 107, 29),
                position=(0.0, 0.0, -10.0),
                material="standard",
                cast_shadow=True,
                receive_shadow=True,
                visible=False,
            )
            self._viser_projectile_handles.append(ball_handle)
            self._viser_velocity_handles.append(None)
        return

    def _update_task_viser(self):
        env_id = self._viser_env_id
        proj_pos, proj_vel = self._get_proj_states()
        proj_pos = proj_pos[env_id].detach().cpu().numpy().astype(np.float32)
        proj_vel = proj_vel[env_id].detach().cpu().numpy().astype(np.float32)

        show_balls = bool(self._viser_show_projectiles.value)
        show_velocity = bool(self._viser_show_projectile_velocity.value)
        for proj_idx, ball_handle in enumerate(self._viser_projectile_handles):
            position = proj_pos[proj_idx]
            velocity = proj_vel[proj_idx]
            active = not self._is_inactive_projectile(proj_idx, position, velocity)
            visible = show_balls and active

            ball_handle.position = position
            ball_handle.visible = visible
            velocity_end = position + self._viser_velocity_scale * velocity
            self._replace_velocity_line(
                proj_idx,
                position,
                velocity_end,
                visible=visible and show_velocity,
            )
        return

    def _replace_velocity_line(self, proj_idx, start, end, visible):
        old_handle = self._viser_velocity_handles[proj_idx]
        if old_handle is not None:
            old_handle.remove()

        points = np.asarray([[start, end]], dtype=np.float32)
        self._viser_velocity_handles[proj_idx] = self.server.scene.add_line_segments(
            name="/dodgeball/projectile_{:02d}/velocity".format(proj_idx),
            points=points,
            colors=(255, 210, 40),
            line_width=2.0,
            visible=visible,
        )
        return

    def _is_inactive_projectile(self, proj_idx, position, velocity):
        inactive_pos = np.asarray(
            [10.0 + proj_idx, 10.0, self._proj_radius], dtype=np.float32
        )
        return (
            np.linalg.norm(position - inactive_pos) < 1e-4
            and np.linalg.norm(velocity) < 1e-4
        )

    def _get_rollout_task_state(self, env_id):
        proj_pos, proj_vel = self._get_proj_states()
        return {
            "projectile_pos": proj_pos[env_id],
            "projectile_vel": proj_vel[env_id],
        }
