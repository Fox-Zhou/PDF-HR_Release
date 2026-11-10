import numpy as np
import torch

from envs.viser_base_env import ViserBaseEnv


class ViserTrajEnv(ViserBaseEnv):
    """Viser scene additions for :class:`TaskTrajEnv`."""

    def _init_task_viser(self, env_config):
        self._viser_traj_sample_handles = []
        self._viser_traj_path_handle = None
        self._viser_cached_traj_verts = None
        self._viser_traj_height = float(env_config.get("viser_traj_height", 0.8))
        self._viser_show_traj_samples = self.server.gui.add_checkbox(
            label="Trajectory samples",
            initial_value=True,
        )
        self._viser_show_full_traj = self.server.gui.add_checkbox(
            label="Full trajectory",
            initial_value=True,
        )
        return

    def _build_task_viser(self):
        marker_color = self._to_rgb255(self._traj_marker_color)
        for sample_idx in range(self._num_traj_samples):
            radius = 0.10 if sample_idx == 0 else 0.055
            handle = self.server.scene.add_icosphere(
                name="/trajectory/sample_{:02d}".format(sample_idx),
                radius=radius,
                color=marker_color,
                position=(0.0, 0.0, -10.0),
                material="standard",
                cast_shadow=False,
                receive_shadow=False,
            )
            self._viser_traj_sample_handles.append(handle)
        return

    def _update_task_viser(self):
        env_id = self._viser_env_id
        env_ids = torch.tensor([env_id], device=self._device, dtype=torch.long)
        samples = self._fetch_traj_samples(env_ids)[0]
        samples = samples.detach().cpu().numpy().astype(np.float32)
        samples[:, 2] = self._viser_traj_height

        show_samples = bool(self._viser_show_traj_samples.value)
        for handle, position in zip(self._viser_traj_sample_handles, samples):
            handle.position = position
            handle.visible = show_samples

        traj_verts = self._traj_gen.get_traj_verts(env_id)
        traj_verts = traj_verts.detach().cpu().numpy().astype(np.float32).copy()
        traj_verts[:, 2] = self._viser_traj_height
        self._update_traj_path(traj_verts)
        return

    def _update_traj_path(self, traj_verts):
        changed = (
            self._viser_cached_traj_verts is None
            or self._viser_cached_traj_verts.shape != traj_verts.shape
            or not np.array_equal(self._viser_cached_traj_verts, traj_verts)
        )
        if changed:
            if self._viser_traj_path_handle is not None:
                self._viser_traj_path_handle.remove()

            points = np.stack([traj_verts[:-1], traj_verts[1:]], axis=1)
            self._viser_traj_path_handle = self.server.scene.add_line_segments(
                name="/trajectory/path",
                points=points,
                colors=(35, 100, 255),
                line_width=2.0,
                visible=bool(self._viser_show_full_traj.value),
            )
            self._viser_cached_traj_verts = traj_verts.copy()
        elif self._viser_traj_path_handle is not None:
            self._viser_traj_path_handle.visible = bool(
                self._viser_show_full_traj.value
            )
        return

    def _get_rollout_task_state(self, env_id):
        env_ids = torch.tensor([env_id], device=self._device, dtype=torch.long)
        return {
            "traj_samples": self._fetch_traj_samples(env_ids)[0],
        }

    @staticmethod
    def _to_rgb255(color):
        color = np.asarray(color, dtype=np.float32)
        if np.max(color) <= 1.0:
            color = 255.0 * color
        return tuple(np.clip(np.rint(color[:3]), 0, 255).astype(np.uint8).tolist())
