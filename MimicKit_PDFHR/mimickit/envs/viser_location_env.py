import numpy as np
import trimesh

from envs.viser_base_env import ViserBaseEnv


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


class ViserLocationEnv(ViserBaseEnv):
    """Viser scene additions for :class:`TaskLocationEnv`."""

    def _init_task_viser(self, env_config):
        self._viser_target_flag_root = None
        self._viser_target_flag_handles = []
        self._viser_target_line = None

        self._viser_show_target = self.server.gui.add_checkbox(
            label="Location target",
            initial_value=True,
        )
        self._viser_show_target_line = self.server.gui.add_checkbox(
            label="Target line",
            initial_value=True,
        )
        return

    def _build_task_viser(self):
        (
            self._viser_target_flag_root,
            self._viser_target_flag_handles,
        ) = _add_location_target_flag(
            self.server, "/location/target_flag"
        )
        self._viser_target_flag_root.position = (0.0, 0.0, -10.0)
        self._replace_target_line(
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            visible=False,
        )
        return

    def _update_task_viser(self):
        env_id = self._viser_env_id
        target_pos = self._tar_pos[env_id].detach().cpu().numpy().astype(np.float32)
        root_pos = self._engine.get_root_pos(self._get_char_id())[env_id]
        root_pos = root_pos.detach().cpu().numpy().astype(np.float32)

        show_target = bool(self._viser_show_target.value)
        self._viser_target_flag_root.position = target_pos
        for handle in self._viser_target_flag_handles:
            handle.visible = show_target

        show_line = show_target and bool(self._viser_show_target_line.value)
        self._replace_target_line(root_pos, target_pos, visible=show_line)
        return

    def _replace_target_line(self, start, end, visible):
        if self._viser_target_line is not None:
            self._viser_target_line.remove()

        points = np.asarray([[start, end]], dtype=np.float32)
        self._viser_target_line = self.server.scene.add_line_segments(
            name="/location/target/line",
            points=points,
            colors=(255, 70, 70),
            line_width=2.0,
            visible=visible,
        )
        return

    def _get_rollout_task_state(self, env_id):
        return {
            "target_pos": self._tar_pos[env_id],
        }
