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
        self._num_samples = {}
        for source in ("baseline", "ours"):
            if self.metadatas[source].get("env_name") != "task_traj":
                raise ValueError(
                    "{} rollout requires env_name=task_traj".format(source)
                )
            fields = set(self.rollouts[source]["task"].dtype.names or ())
            if fields != {"traj_samples"}:
                raise ValueError(
                    "{} trajectory task fields must be exactly: traj_samples".format(
                        source
                    )
                )
            samples = self.rollouts[source]["task"]["traj_samples"]
            if samples.ndim != 3 or samples.shape[2] != 3:
                raise ValueError(
                    "{}.task.traj_samples must have shape [frames, K, 3]".format(
                        source
                    )
                )
            self._num_samples[source] = int(samples.shape[1])
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
        for source in ("baseline", "ours"):
            for slot_id in list(self._task_frame_slots[source].keys()):
                self.update_task_frame_slot(source, slot_id)

    def _create_task_slot(self, source, slot_id):
        prefix = "/compare/{}/traj_{:04d}".format(source, slot_id)
        markers = []
        for sample_idx in range(self._num_samples[source]):
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
        self._traj_slots[(source, slot_id)] = {"markers": markers, "path": None}

    def _update_task_slot(self, source, slot_id, frame_index, opacity):
        slot = self._traj_slots.get((source, slot_id))
        if slot is None:
            return
        samples = np.asarray(
            self.rollouts[source]["task"]["traj_samples"][frame_index],
            dtype=np.float32,
        ).copy()
        samples += self._get_source_offset(source)
        samples[:, 2] = _TRAJ_HEIGHT

        for marker, position in zip(slot["markers"], samples):
            marker.position = position
            marker.opacity = opacity
            marker.visible = bool(self._show_samples.value)
        self._replace_path(
            source, slot_id, samples, bool(self._show_path.value), opacity
        )

    def _replace_path(self, source, slot_id, samples, visible, opacity):
        slot = self._traj_slots[(source, slot_id)]
        if slot["path"] is not None:
            slot["path"].remove()
            slot["path"] = None
        if len(samples) < 2:
            return
        slot["path"] = self._add_task_line_mesh(
            name="/compare/{}/traj_{:04d}/sampled_path".format(
                source, slot_id
            ),
            points=np.stack([samples[:-1], samples[1:]], axis=1),
            color=_TRAJ_COLOR,
            opacity=opacity,
            visible=visible,
        )

    def _remove_task_slot(self, source, slot_id):
        slot = self._traj_slots.pop((source, slot_id), None)
        if slot is None:
            return
        for marker in slot["markers"]:
            marker.remove()
        if slot["path"] is not None:
            slot["path"].remove()

    def _set_task_slot_visible(self, source, slot_id, visible):
        slot = self._traj_slots.get((source, slot_id))
        if slot is not None:
            for marker in slot["markers"]:
                marker.visible = bool(visible)
            if slot["path"] is not None:
                slot["path"].visible = bool(visible)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline_file",
        required=True,
        help="Path to the baseline trajectory rollout NPZ file",
    )
    parser.add_argument("--ours_file", required=True)
    args = parser.parse_args()
    viewer = ViserTrajTimelapse(args.baseline_file, args.ours_file)
    viewer.run_forever()


if __name__ == "__main__":
    main()
