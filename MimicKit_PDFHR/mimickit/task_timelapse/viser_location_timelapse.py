import argparse

import numpy as np
import trimesh

try:
    from .viser_timelapse_base import ViserTimelapseBase
except ImportError:
    from viser_timelapse_base import ViserTimelapseBase


def _make_wavy_flag_mesh(
    width=0.55,
    height=0.32,
    nx=28,
    nz=14,
    wave_amp=0.035,
    wave_freq=2.2,
):
    vertices = []
    faces = []
    for iz in range(nz + 1):
        v = iz / nz
        z = -v * height
        for ix in range(nx + 1):
            u = ix / nx
            x = u * width
            y = wave_amp * u * np.sin(
                2.0 * np.pi * wave_freq * u + 0.8 * v
            )
            vertices.append([x, y, z])
    for iz in range(nz):
        for ix in range(nx):
            a = iz * (nx + 1) + ix
            b = a + 1
            c = a + nx + 1
            d = c + 1
            faces.append([a, c, b])
            faces.append([b, c, d])
    return (
        np.asarray(vertices, dtype=np.float32),
        np.asarray(faces, dtype=np.int32),
    )


def _add_location_target_flag(server, root_name):
    height = 1.25
    pole_radius = 0.018
    flag_width = 0.55
    flag_height = 0.32
    root = server.scene.add_frame(root_name, show_axes=False)
    handles = []

    pole = trimesh.creation.cylinder(
        radius=pole_radius, height=height, sections=32
    )
    handles.append(
        server.scene.add_mesh_simple(
            name=root_name + "/pole",
            vertices=np.asarray(pole.vertices, dtype=np.float32),
            faces=np.asarray(pole.faces, dtype=np.int32),
            color=(0.70, 0.70, 0.70),
            position=(0.0, 0.0, 0.5 * height),
            opacity=1.0,
        )
    )

    top = trimesh.creation.uv_sphere(
        radius=pole_radius * 2.2, count=[16, 16]
    )
    handles.append(
        server.scene.add_mesh_simple(
            name=root_name + "/top_ball",
            vertices=np.asarray(top.vertices, dtype=np.float32),
            faces=np.asarray(top.faces, dtype=np.int32),
            color=(0.82, 0.82, 0.82),
            position=(0.0, 0.0, height + pole_radius * 1.8),
            opacity=1.0,
        )
    )

    base = trimesh.creation.cylinder(
        radius=pole_radius * 4.5,
        height=pole_radius * 1.6,
        sections=32,
    )
    handles.append(
        server.scene.add_mesh_simple(
            name=root_name + "/base",
            vertices=np.asarray(base.vertices, dtype=np.float32),
            faces=np.asarray(base.faces, dtype=np.int32),
            color=(0.50, 0.50, 0.50),
            position=(0.0, 0.0, pole_radius * 0.8),
            opacity=1.0,
        )
    )

    cloth_vertices, cloth_faces = _make_wavy_flag_mesh(
        width=flag_width, height=flag_height
    )
    handles.append(
        server.scene.add_mesh_simple(
            name=root_name + "/cloth",
            vertices=cloth_vertices,
            faces=cloth_faces,
            color=(0.85, 0.05, 0.04),
            position=(pole_radius * 0.9, 0.0, height * 0.88),
            opacity=1.0,
        )
    )

    connector = trimesh.creation.uv_sphere(
        radius=pole_radius * 1.25, count=[12, 12]
    )
    for index, z_rel in enumerate([0.0, -flag_height * 0.88]):
        handles.append(
            server.scene.add_mesh_simple(
                name=root_name + "/connector_{}".format(index),
                vertices=np.asarray(connector.vertices, dtype=np.float32),
                faces=np.asarray(connector.faces, dtype=np.int32),
                color=(0.90, 0.90, 0.90),
                position=(
                    pole_radius * 0.4,
                    0.0,
                    height * 0.88 + z_rel,
                ),
                opacity=1.0,
            )
        )
    return root, handles


class ViserLocationTimelapse(ViserTimelapseBase):
    """Offline multi-frame Viser visualization for TaskLocationEnv."""

    def _build_task_scene(self):
        if self.metadata.get("env_name") != "task_location":
            raise ValueError("Location timelapse requires env_name=task_location")
        task_fields = set(self.rollout["task"].dtype.names or ())
        if task_fields != {"target_pos"}:
            raise ValueError("Location rollout task fields must be exactly: target_pos")

        self._location_slots = {}
        with self.server.gui.add_folder("Location Task"):
            self._show_location_targets = self.server.gui.add_checkbox(
                label="Show targets",
                initial_value=True,
            )
            self._target_rotation = self.server.gui.add_number(
                label="Target rotation (degrees)",
                initial_value=0.0,
                step=5.0,
            )

        @self._show_location_targets.on_update
        def _on_show_targets(_event):
            for slot_id in list(self._task_frame_slots.keys()):
                self.update_task_frame_slot(slot_id)

        @self._target_rotation.on_update
        def _on_target_rotation(_event):
            for slot_id in list(self._task_frame_slots.keys()):
                self.update_task_frame_slot(slot_id)

    def _create_task_slot(self, slot_id):
        prefix = "/timelapse/location_{:04d}".format(slot_id)
        root, handles = _add_location_target_flag(
            self.server, prefix + "/target_flag"
        )
        root.position = (0.0, 0.0, -10.0)
        self._location_slots[slot_id] = {
            "root": root,
            "handles": handles,
        }

    def _update_task_slot(self, slot_id, frame_index, opacity):
        slot = self._location_slots.get(slot_id)
        if slot is None:
            return
        target_pos = np.asarray(
            self.rollout["task"]["target_pos"][frame_index], dtype=np.float32
        )
        visible = bool(self._show_location_targets.value)

        slot["root"].position = target_pos
        angle = np.deg2rad(float(self._target_rotation.value))
        half_angle = 0.5 * angle
        slot["root"].wxyz = np.asarray(
            [np.cos(half_angle), 0.0, 0.0, np.sin(half_angle)],
            dtype=np.float32,
        )
        for handle in slot["handles"]:
            handle.visible = visible
            handle.opacity = opacity

    def _remove_task_slot(self, slot_id):
        slot = self._location_slots.pop(slot_id, None)
        if slot is None:
            return
        for handle in slot["handles"]:
            handle.remove()
        slot["root"].remove()

    def _set_task_slot_visible(self, slot_id, visible):
        slot = self._location_slots.get(slot_id)
        if slot is not None:
            for handle in slot["handles"]:
                handle.visible = bool(visible)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rollout_file",
        required=True,
        help="Path to a location rollout NPZ file",
    )
    args = parser.parse_args()
    viewer = ViserLocationTimelapse(args.rollout_file)
    viewer.run_forever()


if __name__ == "__main__":
    main()
