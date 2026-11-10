import argparse

import numpy as np

try:
    from .viser_timelapse_base import ViserTimelapseBase
except ImportError:
    from viser_timelapse_base import ViserTimelapseBase


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
_TARGET_SPEED_MIN = 0.5
_TARGET_SPEED_MAX = 5.0


class ViserSteeringTimelapse(ViserTimelapseBase):
    """Offline multi-frame Viser visualization for TaskSteeringEnv."""

    def _build_task_scene(self):
        if self.metadata.get("env_name") != "task_steering":
            raise ValueError("Steering timelapse requires env_name=task_steering")
        required = {
            "target_dir",
            "face_dir",
            "target_speed",
        }
        task_fields = set(self.rollout["task"].dtype.names or ())
        if task_fields != required:
            raise ValueError(
                "Steering rollout task fields must be exactly: {}".format(
                    ", ".join(sorted(required))
                )
            )

        self._steering_slots = {}
        with self.server.gui.add_folder("Steering Task"):
            self._show_move_direction = self.server.gui.add_checkbox(
                label="Show move direction", initial_value=True
            )
            self._show_face_direction = self.server.gui.add_checkbox(
                label="Show facing direction", initial_value=True
            )

        @self._show_move_direction.on_update
        def _on_show_move(_event):
            self._update_all_task_slots()

        @self._show_face_direction.on_update
        def _on_show_face(_event):
            self._update_all_task_slots()

    def _update_all_task_slots(self):
        for slot_id in list(self._task_frame_slots.keys()):
            self.update_task_frame_slot(slot_id)

    def _create_task_slot(self, slot_id):
        prefix = "/timelapse/steering_{:04d}".format(slot_id)
        move_arrow = self._add_arrow(prefix + "/move_direction", (220, 35, 35))
        face_arrow = self._add_arrow(prefix + "/face_direction", (35, 200, 70))
        self._steering_slots[slot_id] = {
            "move_arrow": move_arrow,
            "face_arrow": face_arrow,
        }

    def _add_arrow(self, name, color):
        return self.server.scene.add_mesh_simple(
            name=name,
            vertices=_ARROW_VERTICES,
            faces=_ARROW_FACES,
            color=color,
            opacity=1.0,
            material="standard",
            side="double",
            cast_shadow=False,
            receive_shadow=False,
            position=(0.0, 0.0, -10.0),
        )

    def _update_task_slot(self, slot_id, frame_index, opacity):
        slot = self._steering_slots.get(slot_id)
        if slot is None:
            return
        task = self.rollout["task"]
        root_pos = np.asarray(self.rollout["root_pos"][frame_index], dtype=np.float32)
        target_dir = np.asarray(task["target_dir"][frame_index], dtype=np.float32)
        face_dir = np.asarray(task["face_dir"][frame_index], dtype=np.float32)
        target_speed = float(task["target_speed"][frame_index])
        speed_delta = _TARGET_SPEED_MAX - _TARGET_SPEED_MIN
        if speed_delta > 1e-8:
            speed_ratio = (target_speed - _TARGET_SPEED_MIN) / speed_delta
        else:
            speed_ratio = 0.0
        move_dist = 1.0 + 0.5 * float(np.clip(speed_ratio, 0.0, 1.0))
        move_pos = np.asarray(
            [
                root_pos[0] + move_dist * target_dir[0],
                root_pos[1] + move_dist * target_dir[1],
                0.04,
            ],
            dtype=np.float32,
        )
        face_pos = np.asarray(
            [
                root_pos[0] + face_dir[0],
                root_pos[1] + face_dir[1],
                0.05,
            ],
            dtype=np.float32,
        )

        slot["move_arrow"].position = move_pos
        slot["move_arrow"].wxyz = self._direction_to_wxyz(target_dir)
        slot["move_arrow"].opacity = opacity
        slot["move_arrow"].visible = bool(self._show_move_direction.value)

        slot["face_arrow"].position = face_pos
        slot["face_arrow"].wxyz = self._direction_to_wxyz(face_dir)
        slot["face_arrow"].opacity = opacity
        slot["face_arrow"].visible = bool(self._show_face_direction.value)

    def _remove_task_slot(self, slot_id):
        slot = self._steering_slots.pop(slot_id, None)
        if slot is None:
            return
        slot["move_arrow"].remove()
        slot["face_arrow"].remove()

    def _set_task_slot_visible(self, slot_id, visible):
        slot = self._steering_slots.get(slot_id)
        if slot is not None:
            slot["move_arrow"].visible = bool(visible)
            slot["face_arrow"].visible = bool(visible)

    @staticmethod
    def _direction_to_wxyz(direction):
        theta = float(np.arctan2(direction[1], direction[0]))
        half_theta = 0.5 * theta
        return np.asarray(
            [np.cos(half_theta), 0.0, 0.0, np.sin(half_theta)],
            dtype=np.float32,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rollout_file",
        required=True,
        help="Path to a steering rollout NPZ file",
    )
    args = parser.parse_args()
    viewer = ViserSteeringTimelapse(args.rollout_file)
    viewer.run_forever()


if __name__ == "__main__":
    main()
