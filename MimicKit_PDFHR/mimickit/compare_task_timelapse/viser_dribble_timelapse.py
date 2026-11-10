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
_BALL_RADIUS = 0.1
_BALL_COLOR = (248, 107, 29)
_TARGET_MARKER_COLOR = (204, 0, 0)
_TARGET_SPEED_MIN = 1.2
_TARGET_SPEED_MAX = 2.0
_TARGET_MARKER_DIST = 1.5
_TARGET_MARKER_HEIGHT = 0.02
_BALL_VELOCITY_SCALE = 0.4


class ViserDribbleTimelapse(ViserTimelapseBase):
    """Offline multi-frame Viser visualization for TaskDribbleEnv."""

    def _build_task_scene(self):
        required = {
            "ball_pos",
            "ball_vel",
            "target_dir",
            "target_speed",
        }
        for source in ("baseline", "ours"):
            if self.metadatas[source].get("env_name") != "task_dribble":
                raise ValueError(
                    "{} rollout requires env_name=task_dribble".format(source)
                )
            fields = set(self.rollouts[source]["task"].dtype.names or ())
            if fields != required:
                raise ValueError(
                    "{} dribble task fields must be exactly: {}".format(
                        source, ", ".join(sorted(required))
                    )
                )

        self._dribble_slots = {}
        with self.server.gui.add_folder("Dribble Task"):
            self._show_ball = self.server.gui.add_checkbox(
                label="Show ball", initial_value=True
            )
            self._show_target_velocity = self.server.gui.add_checkbox(
                label="Show target velocity", initial_value=True
            )
            self._show_ball_velocity = self.server.gui.add_checkbox(
                label="Show actual velocity", initial_value=True
            )

        for handle in (
            self._show_ball,
            self._show_target_velocity,
            self._show_ball_velocity,
        ):
            @handle.on_update
            def _on_visibility(_event):
                self._update_all_task_slots()

    def _update_all_task_slots(self):
        for source in ("baseline", "ours"):
            for slot_id in list(self._task_frame_slots[source].keys()):
                self.update_task_frame_slot(source, slot_id)

    def _create_task_slot(self, source, slot_id):
        prefix = "/compare/{}/dribble_{:04d}".format(source, slot_id)
        ball = self.server.scene.add_icosphere(
            name=prefix + "/ball",
            radius=_BALL_RADIUS,
            color=_BALL_COLOR,
            position=(0.0, 0.0, -10.0),
            opacity=1.0,
            material="standard",
            cast_shadow=True,
            receive_shadow=True,
        )
        target = self.server.scene.add_mesh_simple(
            name=prefix + "/target_velocity",
            vertices=_ARROW_VERTICES,
            faces=_ARROW_FACES,
            color=_TARGET_MARKER_COLOR,
            opacity=1.0,
            material="standard",
            side="double",
            cast_shadow=False,
            receive_shadow=False,
            position=(0.0, 0.0, -10.0),
        )
        self._dribble_slots[(source, slot_id)] = {
            "ball": ball,
            "target": target,
            "velocity": None,
        }

    def _update_task_slot(self, source, slot_id, frame_index, opacity):
        slot = self._dribble_slots.get((source, slot_id))
        if slot is None:
            return
        task = self.rollouts[source]["task"]
        ball_pos = (
            np.asarray(task["ball_pos"][frame_index], dtype=np.float32)
            + self._get_source_offset(source)
        )
        ball_vel = np.asarray(task["ball_vel"][frame_index], dtype=np.float32)
        target_dir = np.asarray(task["target_dir"][frame_index], dtype=np.float32)
        target_speed = float(task["target_speed"][frame_index])
        speed_delta = _TARGET_SPEED_MAX - _TARGET_SPEED_MIN
        if speed_delta > 1e-8:
            speed_alpha = (target_speed - _TARGET_SPEED_MIN) / speed_delta
        else:
            speed_alpha = 0.0
        marker_dist = _TARGET_MARKER_DIST * (
            0.75 + 0.5 * float(np.clip(speed_alpha, 0.0, 1.0))
        )
        marker_pos = np.asarray(
            [
                ball_pos[0] + marker_dist * target_dir[0],
                ball_pos[1] + marker_dist * target_dir[1],
                _TARGET_MARKER_HEIGHT,
            ],
            dtype=np.float32,
        )

        slot["ball"].position = ball_pos
        slot["ball"].opacity = opacity
        slot["ball"].visible = bool(self._show_ball.value)
        slot["target"].position = marker_pos
        slot["target"].wxyz = self._direction_to_wxyz(target_dir)
        slot["target"].opacity = opacity
        slot["target"].visible = bool(self._show_target_velocity.value)
        self._replace_velocity_line(
            slot_id,
            source,
            ball_pos,
            ball_pos + _BALL_VELOCITY_SCALE * ball_vel,
            bool(self._show_ball_velocity.value),
            opacity,
        )

    def _replace_velocity_line(
        self, slot_id, source, start, end, visible, opacity
    ):
        slot = self._dribble_slots[(source, slot_id)]
        if slot["velocity"] is not None:
            slot["velocity"].remove()
        slot["velocity"] = self._add_task_line_mesh(
            name="/compare/{}/dribble_{:04d}/actual_velocity".format(
                source, slot_id
            ),
            points=np.asarray([[start, end]], dtype=np.float32),
            color=(255, 210, 40),
            opacity=opacity,
            visible=visible,
        )

    def _remove_task_slot(self, source, slot_id):
        slot = self._dribble_slots.pop((source, slot_id), None)
        if slot is None:
            return
        slot["ball"].remove()
        slot["target"].remove()
        if slot["velocity"] is not None:
            slot["velocity"].remove()

    def _set_task_slot_visible(self, source, slot_id, visible):
        slot = self._dribble_slots.get((source, slot_id))
        if slot is not None:
            slot["ball"].visible = bool(visible)
            slot["target"].visible = bool(visible)
            if slot["velocity"] is not None:
                slot["velocity"].visible = bool(visible)

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
        "--baseline_file",
        required=True,
        help="Path to the baseline dribble rollout NPZ file",
    )
    parser.add_argument("--ours_file", required=True)
    args = parser.parse_args()
    viewer = ViserDribbleTimelapse(args.baseline_file, args.ours_file)
    viewer.run_forever()


if __name__ == "__main__":
    main()
