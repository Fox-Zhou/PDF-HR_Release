import os

from envs.viser_base_env import (
    ViserBaseEnv,
    _MJCFHumanoidVisual,
    _xyzw_to_wxyz,
)


class ViserTrackingEnv(ViserBaseEnv):
    """Viser additions shared by DeepMimic, AMP, ADD, and SMP."""

    def _enable_ref_char(self):
        # Reference updates are needed by Viser, but enabling the simulator's
        # native visualization would also construct a camera and viewer.
        return bool(self._visualize_ref_char)

    def _build_engine(self, engine_config, num_envs, device, visualize,
                      record_video=False):
        # Viser owns rendering, so the simulator must remain headless.
        return super()._build_engine(
            engine_config=engine_config,
            num_envs=num_envs,
            device=device,
            visualize=False,
            record_video=record_video,
        )

    def _init_task_viser(self, env_config):
        self._viser_ref_render_type = None
        self._viser_ref_handles = None
        self._viser_ref_root_handle = None
        self._viser_ref_urdf = None
        self._viser_show_ref = self.server.gui.add_checkbox(
            label="Show reference",
            initial_value=bool(env_config.get("viser_show_ref", True)),
        )

        @self._viser_show_ref.on_update
        def _on_show_ref_change(_event):
            self._set_reference_visible(bool(self._viser_show_ref.value))

        return

    def _build_task_viser(self):
        body_names = self._engine.get_obj_body_names(self._get_char_id())
        char_file_lower = self._viser_char_file.lower()

        if "g1" in char_file_lower:
            from viser.extras import ViserUrdf
            import yourdfpy

            urdf_file = "data/assets/viser_g1/g1_29dof_rev_1_0_colored.urdf"
            if not os.path.isfile(urdf_file):
                urdf_file = "data/assets/viser_g1/g1_29dof_rev_1_0.urdf"
            mesh_dir = "data/assets/viser_g1/meshes"
            assert os.path.isfile(urdf_file), (
                "Missing G1 reference Viser URDF: {}".format(urdf_file)
            )

            self._viser_ref_root_handle = self.server.scene.add_frame(
                name="/reference_root",
                position=(0.0, 0.0, 0.0),
                wxyz=(1.0, 0.0, 0.0, 0.0),
                show_axes=False,
            )
            urdf = yourdfpy.URDF.load(urdf_file, mesh_dir=mesh_dir)
            self._viser_ref_urdf = ViserUrdf(
                self.server,
                urdf,
                root_node_name="/reference_root",
            )
            self._viser_ref_render_type = "g1_urdf"
        else:
            assert self._viser_char_file.endswith(".xml"), (
                "Humanoid Viser requires an MJCF XML character file"
            )
            humanoid_visual = _MJCFHumanoidVisual(self._viser_char_file)
            self._viser_ref_handles = humanoid_visual.build(
                self.server,
                body_names,
                prefix="/reference",
            )
            self._viser_ref_urdf = humanoid_visual
            self._viser_ref_render_type = "humanoid_mjcf"

        self._set_reference_visible(bool(self._viser_show_ref.value))
        return

    def _update_task_viser(self):
        env_id = self._viser_env_id
        ref_offset = self._ref_char_offset.detach().cpu().numpy()

        if self._viser_ref_render_type == "g1_urdf":
            root_pos = self._ref_root_pos[env_id].detach().cpu().numpy()
            root_rot = self._ref_root_rot[env_id].detach().cpu().numpy()
            dof_pos = self._ref_dof_pos[env_id].detach().cpu().numpy()
            self._viser_ref_root_handle.position = root_pos + ref_offset
            self._viser_ref_root_handle.wxyz = _xyzw_to_wxyz(root_rot)
            self._viser_ref_urdf.update_cfg(dof_pos)
        else:
            body_pos = self._ref_body_pos[env_id].detach().cpu().numpy()
            body_rot = self._ref_body_rot[env_id].detach().cpu().numpy()
            self._viser_ref_urdf.update(
                self._viser_ref_handles,
                body_pos + ref_offset,
                body_rot,
            )

        self._set_reference_visible(bool(self._viser_show_ref.value))
        return

    def _set_reference_visible(self, visible):
        if self._viser_ref_render_type == "g1_urdf":
            if self._viser_ref_root_handle is not None:
                self._viser_ref_root_handle.visible = bool(visible)
        elif self._viser_ref_handles is not None:
            for _body_idx, _spec, handle in self._viser_ref_handles:
                handle.visible = bool(visible)
        return

    def _prepare_viser_capture(self):
        # Tracking screenshots and frame recordings contain policy only.
        was_visible = bool(self._viser_show_ref.value)
        self._set_reference_visible(False)
        return was_visible

    def _restore_viser_capture(self, capture_state):
        self._set_reference_visible(bool(capture_state))
        return

    def _get_rollout_task_state(self, env_id):
        return {
            "ref_root_pos": self._ref_root_pos[env_id],
            "ref_root_rot": self._ref_root_rot[env_id],
            "ref_dof_pos": self._ref_dof_pos[env_id],
            "ref_body_pos": self._ref_body_pos[env_id],
            "ref_body_rot": self._ref_body_rot[env_id],
            "motion_id": self._motion_ids[env_id],
            "motion_time": self._get_motion_times()[env_id],
        }
