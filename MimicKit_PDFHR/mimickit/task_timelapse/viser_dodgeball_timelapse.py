import argparse

import numpy as np

try:
    from .viser_timelapse_base import ViserTimelapseBase
except ImportError:
    from viser_timelapse_base import ViserTimelapseBase

_PROJECTILE_RADIUS = 0.1
_VELOCITY_SCALE = 0.1


class ViserDodgeballTimelapse(ViserTimelapseBase):
    """Offline multi-frame Viser visualization for TaskDodgeballEnv."""

    def _build_task_scene(self):
        if self.metadata.get("env_name") != "task_dodgeball":
            raise ValueError("Dodgeball timelapse requires env_name=task_dodgeball")
        required = {
            "projectile_pos",
            "projectile_vel",
        }
        fields = set(self.rollout["task"].dtype.names or ())
        if fields != required:
            raise ValueError(
                "Dodgeball rollout task fields must be exactly: {}".format(
                    ", ".join(sorted(required))
                )
            )

        projectile_pos = self.rollout["task"]["projectile_pos"]
        if projectile_pos.ndim != 3 or projectile_pos.shape[2] != 3:
            raise ValueError("task.projectile_pos must have shape [frames, N, 3]")
        self._num_projectiles = int(projectile_pos.shape[1])
        self._dodgeball_slots = {}
        with self.server.gui.add_folder("Dodgeball Task"):
            self._show_projectiles = self.server.gui.add_checkbox(
                label="Show dodgeballs", initial_value=True
            )
            self._show_projectile_velocity = self.server.gui.add_checkbox(
                label="Show velocity", initial_value=True
            )

        @self._show_projectiles.on_update
        def _on_show_projectiles(_event):
            self._update_all_task_slots()

        @self._show_projectile_velocity.on_update
        def _on_show_velocity(_event):
            self._update_all_task_slots()

    def _update_all_task_slots(self):
        for slot_id in list(self._task_frame_slots.keys()):
            self.update_task_frame_slot(slot_id)

    def _create_task_slot(self, slot_id):
        prefix = "/timelapse/dodgeball_{:04d}".format(slot_id)
        projectiles = []
        for proj_idx in range(self._num_projectiles):
            ball = self.server.scene.add_icosphere(
                name=prefix + "/projectile_{:02d}/ball".format(proj_idx),
                radius=_PROJECTILE_RADIUS,
                color=(248, 107, 29),
                position=(0.0, 0.0, -10.0),
                opacity=1.0,
                material="standard",
                cast_shadow=True,
                receive_shadow=True,
                visible=False,
            )
            projectiles.append({"ball": ball, "velocity": None})
        self._dodgeball_slots[slot_id] = projectiles

    def _update_task_slot(self, slot_id, frame_index, opacity):
        projectiles = self._dodgeball_slots.get(slot_id)
        if projectiles is None:
            return
        task = self.rollout["task"]
        positions = np.asarray(task["projectile_pos"][frame_index], dtype=np.float32)
        velocities = np.asarray(task["projectile_vel"][frame_index], dtype=np.float32)
        for proj_idx, projectile in enumerate(projectiles):
            position = positions[proj_idx]
            velocity = velocities[proj_idx]
            inactive_pos = np.asarray(
                [10.0 + proj_idx, 10.0, _PROJECTILE_RADIUS], dtype=np.float32
            )
            active = not (
                np.linalg.norm(position - inactive_pos) < 1e-4
                and np.linalg.norm(velocity) < 1e-4
            )
            visible = bool(self._show_projectiles.value) and active
            projectile["ball"].position = position
            projectile["ball"].opacity = opacity
            projectile["ball"].visible = visible
            self._replace_velocity_line(
                slot_id,
                proj_idx,
                position,
                position + _VELOCITY_SCALE * velocity,
                visible and bool(self._show_projectile_velocity.value),
                opacity,
            )

    def _replace_velocity_line(
        self, slot_id, proj_idx, start, end, visible, opacity
    ):
        projectile = self._dodgeball_slots[slot_id][proj_idx]
        if projectile["velocity"] is not None:
            projectile["velocity"].remove()
        projectile["velocity"] = self._add_task_line_mesh(
            name="/timelapse/dodgeball_{:04d}/projectile_{:02d}/velocity".format(
                slot_id, proj_idx
            ),
            points=np.asarray([[start, end]], dtype=np.float32),
            color=(255, 210, 40),
            opacity=opacity,
            visible=visible,
        )

    def _remove_task_slot(self, slot_id):
        projectiles = self._dodgeball_slots.pop(slot_id, None)
        if projectiles is None:
            return
        for projectile in projectiles:
            projectile["ball"].remove()
            if projectile["velocity"] is not None:
                projectile["velocity"].remove()

    def _set_task_slot_visible(self, slot_id, visible):
        projectiles = self._dodgeball_slots.get(slot_id)
        if projectiles is not None:
            for projectile in projectiles:
                projectile["ball"].visible = bool(visible)
                if projectile["velocity"] is not None:
                    projectile["velocity"].visible = bool(visible)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rollout_file",
        required=True,
        help="Path to a dodgeball rollout NPZ file",
    )
    args = parser.parse_args()
    viewer = ViserDodgeballTimelapse(args.rollout_file)
    viewer.run_forever()


if __name__ == "__main__":
    main()
