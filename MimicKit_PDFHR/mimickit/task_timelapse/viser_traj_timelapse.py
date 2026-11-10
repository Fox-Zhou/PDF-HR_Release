import argparse

import numpy as np

try:
    from .viser_timelapse_base import ViserTimelapseBase
except ImportError:
    from viser_timelapse_base import ViserTimelapseBase


_TRAJ_HEIGHT = 0.8
_TRAJ_COLOR = (35, 100, 255)
_MARKER_COLOR = (204, 0, 0)
_SAMPLE_RADIUS = 0.055


class ViserTrajTimelapse(ViserTimelapseBase):
    """Offline multi-frame Viser visualization for TaskTrajEnv."""

    def _build_task_scene(self):
        if self.metadata.get("env_name") != "task_traj":
            raise ValueError("Trajectory timelapse requires env_name=task_traj")
        fields = set(self.rollout["task"].dtype.names or ())
        if fields != {"traj_samples"}:
            raise ValueError(
                "Trajectory rollout task fields must be exactly: traj_samples"
            )
        samples = self.rollout["task"]["traj_samples"]
        if samples.ndim != 3 or samples.shape[2] != 3:
            raise ValueError("task.traj_samples must have shape [frames, K, 3]")

        self._num_samples = int(samples.shape[1])
        self._traj_slots = {}
        with self.server.gui.add_folder("Trajectory Task"):
            self._show_samples = self.server.gui.add_checkbox(
                label="Show trajectory samples", initial_value=True
            )
            self._show_path = self.server.gui.add_checkbox(
                label="Show sampled path", initial_value=True
            )

        @self._show_samples.on_update
        def _on_show_samples(_event):
            self._update_all_task_slots()

        @self._show_path.on_update
        def _on_show_path(_event):
            self._update_all_task_slots()

    def _update_all_task_slots(self):
        for slot_id in list(self._task_frame_slots.keys()):
            self.update_task_frame_slot(slot_id)

    def _create_task_slot(self, slot_id):
        prefix = "/timelapse/traj_{:04d}".format(slot_id)
        markers = []
        for sample_idx in range(self._num_samples):
            markers.append(
                self.server.scene.add_icosphere(
                    name=prefix + "/sample_{:02d}".format(sample_idx),
                    radius=_SAMPLE_RADIUS,
                    color=_MARKER_COLOR,
                    position=(0.0, 0.0, -10.0),
                    opacity=1.0,
                    material="standard",
                    cast_shadow=False,
                    receive_shadow=False,
                    visible=False,
                )
            )
        self._traj_slots[slot_id] = {"markers": markers, "path": None}

    def _update_task_slot(self, slot_id, frame_index, opacity):
        slot = self._traj_slots.get(slot_id)
        if slot is None:
            return
        samples = np.asarray(
            self.rollout["task"]["traj_samples"][frame_index], dtype=np.float32
        ).copy()
        samples[:, 2] = _TRAJ_HEIGHT

        for marker, position in zip(slot["markers"], samples):
            marker.position = position
            marker.opacity = opacity
            marker.visible = bool(self._show_samples.value)
        self._replace_path(
            slot_id, samples, bool(self._show_path.value), opacity
        )

    def _replace_path(self, slot_id, samples, visible, opacity):
        slot = self._traj_slots[slot_id]
        if slot["path"] is not None:
            slot["path"].remove()
            slot["path"] = None
        if len(samples) < 2:
            return
        slot["path"] = self._add_task_line_mesh(
            name="/timelapse/traj_{:04d}/sampled_path".format(slot_id),
            points=np.stack([samples[:-1], samples[1:]], axis=1),
            color=_TRAJ_COLOR,
            opacity=opacity,
            visible=visible,
        )

    def _remove_task_slot(self, slot_id):
        slot = self._traj_slots.pop(slot_id, None)
        if slot is None:
            return
        for marker in slot["markers"]:
            marker.remove()
        if slot["path"] is not None:
            slot["path"].remove()

    def _set_task_slot_visible(self, slot_id, visible):
        slot = self._traj_slots.get(slot_id)
        if slot is not None:
            for marker in slot["markers"]:
                marker.visible = bool(visible)
            if slot["path"] is not None:
                slot["path"].visible = bool(visible)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rollout_file",
        required=True,
        help="Path to a trajectory rollout NPZ file",
    )
    args = parser.parse_args()
    viewer = ViserTrajTimelapse(args.rollout_file)
    viewer.run_forever()


if __name__ == "__main__":
    main()
