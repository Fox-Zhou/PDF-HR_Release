import json
import os
import time
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod

import numpy as np
import viser


_HAND_MARKER_SPECS = (
    ("left", (60, 180, 255)),
    ("right", (255, 170, 50)),
)


def _normalize(vec, eps=1e-8):
    vec = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm < eps:
        return np.zeros_like(vec)
    return vec / norm


def _quat_mul_xyzw(lhs, rhs):
    x1, y1, z1, w1 = np.asarray(lhs, dtype=np.float32)
    x2, y2, z2, w2 = np.asarray(rhs, dtype=np.float32)
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ], dtype=np.float32)


def _quat_rotate_xyzw(quat, vec):
    quat = np.asarray(quat, dtype=np.float32)
    quat_conj = np.array([-quat[0], -quat[1], -quat[2], quat[3]], dtype=np.float32)
    vec_quat = np.array([vec[0], vec[1], vec[2], 0.0], dtype=np.float32)
    return _quat_mul_xyzw(_quat_mul_xyzw(quat, vec_quat), quat_conj)[:3]


def _quat_from_two_vectors_xyzw(source, target):
    source = _normalize(source)
    target = _normalize(target)
    if np.linalg.norm(source) < 1e-8 or np.linalg.norm(target) < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if dot > 0.999999:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    if dot < -0.999999:
        axis = np.cross(source, np.array([1.0, 0.0, 0.0], dtype=np.float32))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(source, np.array([0.0, 1.0, 0.0], dtype=np.float32))
        axis = _normalize(axis)
        return np.array([axis[0], axis[1], axis[2], 0.0], dtype=np.float32)

    axis = np.cross(source, target)
    quat = np.array([axis[0], axis[1], axis[2], 1.0 + dot], dtype=np.float32)
    return _normalize(quat)


def _xyzw_to_wxyz(quat):
    quat = np.asarray(quat, dtype=np.float32)
    return np.array([quat[3], quat[0], quat[1], quat[2]], dtype=np.float32)


def _parse_floats(value, default=None):
    if value is None:
        return default
    return [float(x) for x in str(value).replace(",", " ").split()]


def _make_sphere_mesh(radius, sectors=20, stacks=10):
    vertices = []
    for stack in range(stacks + 1):
        phi = np.pi * stack / stacks
        z = radius * np.cos(phi)
        ring_radius = radius * np.sin(phi)
        for sector in range(sectors):
            theta = 2.0 * np.pi * sector / sectors
            vertices.append([
                ring_radius * np.cos(theta),
                ring_radius * np.sin(theta),
                z,
            ])

    faces = []
    for stack in range(stacks):
        for sector in range(sectors):
            a = stack * sectors + sector
            b = stack * sectors + (sector + 1) % sectors
            c = (stack + 1) * sectors + sector
            d = (stack + 1) * sectors + (sector + 1) % sectors
            if stack != 0:
                faces.append([a, c, b])
            if stack != stacks - 1:
                faces.append([b, c, d])

    return (np.asarray(vertices, dtype=np.float32),
            np.asarray(faces, dtype=np.int32))


def _make_capsule_mesh(radius, length, sectors=20, hemi_rings=7):
    if length < 1e-6:
        return _make_sphere_mesh(radius, sectors, 2 * hemi_rings)

    half_length = 0.5 * length
    profile = []
    for ring in range(hemi_rings + 1):
        angle = -0.5 * np.pi + 0.5 * np.pi * ring / hemi_rings
        profile.append((half_length * -1.0 + radius * np.sin(angle),
                        radius * np.cos(angle)))
    profile.append((half_length, radius))
    for ring in range(1, hemi_rings + 1):
        angle = 0.5 * np.pi * ring / hemi_rings
        profile.append((half_length + radius * np.sin(angle),
                        radius * np.cos(angle)))

    vertices = []
    for z, ring_radius in profile:
        for sector in range(sectors):
            theta = 2.0 * np.pi * sector / sectors
            vertices.append([
                ring_radius * np.cos(theta),
                ring_radius * np.sin(theta),
                z,
            ])

    faces = []
    for ring in range(len(profile) - 1):
        for sector in range(sectors):
            a = ring * sectors + sector
            b = ring * sectors + (sector + 1) % sectors
            c = (ring + 1) * sectors + sector
            d = (ring + 1) * sectors + (sector + 1) % sectors
            if profile[ring][1] > 1e-8:
                faces.append([a, c, b])
            if profile[ring + 1][1] > 1e-8:
                faces.append([b, c, d])

    return (np.asarray(vertices, dtype=np.float32),
            np.asarray(faces, dtype=np.int32))


def _make_cone_mesh_between(base_center, tip, radius, sectors=18):
    base_center = np.asarray(base_center, dtype=np.float32)
    tip = np.asarray(tip, dtype=np.float32)
    axis = tip - base_center
    length = np.linalg.norm(axis)
    if length < 1e-6:
        return _make_sphere_mesh(radius, sectors, 6)

    direction = axis / length
    tangent = np.cross(direction, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    if np.linalg.norm(tangent) < 1e-6:
        tangent = np.cross(direction, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    tangent = _normalize(tangent)
    bitangent = _normalize(np.cross(direction, tangent))

    vertices = [tip]
    for sector in range(sectors):
        theta = 2.0 * np.pi * sector / sectors
        radial = np.cos(theta) * tangent + np.sin(theta) * bitangent
        vertices.append(base_center + radius * radial)
    vertices.append(base_center)

    base_center_idx = len(vertices) - 1
    faces = []
    for sector in range(sectors):
        curr_idx = 1 + sector
        next_idx = 1 + (sector + 1) % sectors
        faces.append([0, curr_idx, next_idx])
        faces.append([base_center_idx, next_idx, curr_idx])

    return (np.asarray(vertices, dtype=np.float32),
            np.asarray(faces, dtype=np.int32))


def _add_mesh(server, name, vertices, faces, color, fallback_dimensions):
    if hasattr(server.scene, "add_mesh_simple"):
        try:
            return server.scene.add_mesh_simple(
                name=name,
                vertices=vertices,
                faces=faces,
                color=color,
                material="standard",
                cast_shadow=True,
                receive_shadow=True,
            )
        except TypeError:
            return server.scene.add_mesh_simple(
                name=name,
                vertices=vertices,
                faces=faces,
                color=color,
            )

    return server.scene.add_box(
        name=name,
        dimensions=tuple(fallback_dimensions),
        position=(0.0, 0.0, -10.0),
        color=color,
        material="standard",
        side="double",
    )


class _MJCFHumanoidVisual:
    def __init__(self, mjcf_file):
        self._geom_specs = []
        self._parse(mjcf_file)

    def _parse(self, mjcf_file):
        root = ET.parse(mjcf_file).getroot()
        worldbody = root.find("worldbody")
        if worldbody is None:
            raise ValueError("MJCF has no worldbody: {}".format(mjcf_file))

        def walk_body(body_node):
            body_name = body_node.get("name")
            if body_name is None:
                body_name = "body_{}".format(len(self._geom_specs))

            for geom_idx, geom_node in enumerate(body_node.findall("geom")):
                spec = self._parse_geom(body_name, geom_idx, geom_node)
                if spec is not None:
                    self._geom_specs.append(spec)

            for child_node in body_node.findall("body"):
                walk_body(child_node)

        for body_node in worldbody.findall("body"):
            walk_body(body_node)
        return

    def _parse_geom(self, body_name, geom_idx, geom_node):
        size = _parse_floats(geom_node.get("size"))
        if size is None or len(size) == 0:
            return None

        geom_name = geom_node.get("name", "geom_{}".format(geom_idx))
        geom_type = geom_node.get("type")
        fromto = _parse_floats(geom_node.get("fromto"))
        local_pos = np.asarray(
            _parse_floats(geom_node.get("pos"), [0.0, 0.0, 0.0]),
            dtype=np.float32,
        )
        local_rot = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        quat_wxyz = _parse_floats(geom_node.get("quat"))
        if quat_wxyz is not None and len(quat_wxyz) == 4:
            local_rot = _normalize(np.array([
                quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]
            ], dtype=np.float32))

        radius = None
        length = None
        if fromto is not None and len(fromto) == 6:
            point0 = np.asarray(fromto[:3], dtype=np.float32)
            point1 = np.asarray(fromto[3:], dtype=np.float32)
            segment = point1 - point0
            radius = float(size[0])
            length = float(np.linalg.norm(segment))
            local_pos = 0.5 * (point0 + point1)
            local_rot = _quat_from_two_vectors_xyzw([0.0, 0.0, 1.0], segment)
            dimensions = np.array([2.0 * radius, 2.0 * radius,
                                   length + 2.0 * radius], dtype=np.float32)
            draw_type = "capsule"
        else:
            if geom_type is None:
                geom_type = "sphere" if len(size) == 1 else "box"
            geom_type = geom_type.lower()

            if geom_type == "sphere":
                radius = float(size[0])
                dimensions = np.full(3, 2.0 * radius, dtype=np.float32)
                draw_type = "sphere"
            elif geom_type == "box" and len(size) >= 3:
                dimensions = 2.0 * np.asarray(size[:3], dtype=np.float32)
                draw_type = "box"
            elif geom_type == "capsule":
                radius = float(size[0])
                length = 2.0 * float(size[1]) if len(size) > 1 else 0.0
                dimensions = np.array([2.0 * radius, 2.0 * radius,
                                       length + 2.0 * radius], dtype=np.float32)
                draw_type = "capsule"
            else:
                return None

        rgba = _parse_floats(geom_node.get("rgba"))
        color = None
        if rgba is not None and len(rgba) >= 3:
            color = tuple(int(255 * np.clip(value, 0.0, 1.0)) for value in rgba[:3])

        return {
            "body_name": body_name,
            "geom_name": geom_name,
            "draw_type": draw_type,
            "local_pos": local_pos,
            "local_rot": local_rot,
            "dimensions": dimensions,
            "radius": radius,
            "length": length,
            "color": color,
        }

    def build(self, server, body_names, prefix="/character"):
        body_name_to_idx = {name: idx for idx, name in enumerate(body_names)}
        handles = []

        for geom_idx, spec in enumerate(self._geom_specs):
            body_idx = body_name_to_idx.get(spec["body_name"])
            if body_idx is None:
                continue

            color = spec["color"] or (75, 145, 230)
            safe_body = spec["body_name"].replace("/", "_")
            safe_geom = spec["geom_name"].replace("/", "_")
            name = "{}/{:03d}_{}_{}".format(prefix, geom_idx, safe_body, safe_geom)

            if spec["draw_type"] == "sphere":
                vertices, faces = _make_sphere_mesh(spec["radius"])
                handle = _add_mesh(server, name, vertices, faces, color,
                                   spec["dimensions"])
            elif spec["draw_type"] == "capsule":
                vertices, faces = _make_capsule_mesh(spec["radius"], spec["length"])
                handle = _add_mesh(server, name, vertices, faces, color,
                                   spec["dimensions"])
            else:
                handle = server.scene.add_box(
                    name=name,
                    dimensions=tuple(spec["dimensions"]),
                    position=(0.0, 0.0, -10.0),
                    color=color,
                    material="standard",
                    side="double",
                    cast_shadow=True,
                    receive_shadow=True,
                )

            handles.append((body_idx, spec, handle))

        return handles

    def update(self, handles, body_pos, body_rot):
        for body_idx, spec, handle in handles:
            position = body_pos[body_idx]
            rotation = body_rot[body_idx]
            world_position = position + _quat_rotate_xyzw(rotation, spec["local_pos"])
            world_rotation = _normalize(_quat_mul_xyzw(rotation, spec["local_rot"]))
            handle.position = world_position
            handle.wxyz = _xyzw_to_wxyz(world_rotation)
        return


class ViserBaseEnv(ABC):
    """Shared Viser lifecycle for task-specific visualization mixins.

    The class is intentionally used as a mixin. A concrete Viser task class
    inherits from this class and the corresponding simulation environment is
    supplied later by ``ViserEnvBuilder`` through multiple inheritance.

    Subclasses own task nodes through ``_build_task_viser()`` and
    ``_update_task_viser()``. Robot rendering, environment selection, camera
    tracking, screenshots, and frame-sequence recording live here.
    """

    def __init__(self, *args, **kwargs):
        self._viser_ready = False
        self._viser_env_id = 0
        self._viser_char_mode = None
        self._viser_char_handles = None
        self._viser_char_root_handle = None
        self._viser_char_urdf = None
        env_config = kwargs.get("env_config", {})
        self._viser_char_file = str(env_config.get("char_file", ""))
        self._viser_env_name = str(env_config.get("env_name", "env"))
        self._viser_recording = False
        self._viser_step_id = 0
        self._viser_playing = False
        rollout_config = env_config.get("viser_rollout", {})
        self._viser_rollout_enabled = bool(rollout_config.get("enabled", False))
        self._viser_rollout_file = os.path.abspath(
            os.path.expanduser(
                str(
                    rollout_config.get(
                        "output_file",
                        os.path.join(
                            "record", "data", self._viser_env_name + "_rollout.npz"
                        ),
                    )
                )
            )
        )
        self._viser_rollout_metadata = {
            "env_name": self._viser_env_name,
            "char_file": self._viser_char_file,
            "seed": int(rollout_config.get("seed", 0)),
        }
        self._viser_rollout_metadata.update(rollout_config.get("metadata", {}))
        self._viser_rollout_frames = []
        self._viser_rollout_root_pos = []
        self._viser_rollout_root_rot = []
        self._viser_rollout_dof_pos = []
        self._viser_rollout_body_pos = []
        self._viser_rollout_body_rot = []
        self._viser_rollout_actions = []
        self._viser_rollout_task_data = {}

        super().__init__(*args, **kwargs)

        assert self.get_num_envs() > self._viser_env_id
        self._init_viser(env_config)
        self._init_task_viser(env_config)
        self._build_viser_scene()
        self._viser_ready = True
        self._update_viser_scene()
        return

    def _init_viser(self, env_config):
        self.server = viser.ViserServer()
        self.server.scene.set_up_direction("+z")

        ground_config = env_config.get("viser_ground", {})
        grid_width = float(ground_config.get("width", 100.0))
        grid_position = ground_config.get("position", [0.0, 0.0, 0.01])
        grid_cell_size = float(ground_config.get("cell_size", 1.0))
        grid_thickness = float(ground_config.get("cell_thickness", 1.0))
        grid_color = tuple(ground_config.get("color", [180, 180, 180]))
        self._viser_grid = self.server.scene.add_grid(
            name="/grid",
            width=grid_width,
            height=grid_width,
            position=tuple(grid_position),
            cell_size=grid_cell_size,
            cell_thickness=grid_thickness,
            section_size=5.0 * grid_cell_size,
            cell_color=grid_color,
            section_color=grid_color,
            plane="xy",
        )

        camera_config = env_config.get("viser_cam", {})
        cam_offset = camera_config.get("cam_delta", [3.0, -3.0, 2.0])
        look_offset = camera_config.get("cam_toward_delta", [0.0, 0.0, 0.9])
        camera_mode = str(camera_config.get("mode", "track"))
        if camera_mode not in ("track", "user"):
            camera_mode = "track"

        self._viser_play_control = self.server.gui.add_checkbox(
            label="Playing",
            initial_value=False,
        )
        self._viser_camera_mode = self.server.gui.add_dropdown(
            label="Camera",
            options=["track", "user"],
            initial_value=camera_mode,
        )
        self._viser_cam_x = self.server.gui.add_number(
            label="Camera X", initial_value=float(cam_offset[0]), step=0.1
        )
        self._viser_cam_y = self.server.gui.add_number(
            label="Camera Y", initial_value=float(cam_offset[1]), step=0.1
        )
        self._viser_cam_z = self.server.gui.add_number(
            label="Camera Z", initial_value=float(cam_offset[2]), step=0.1
        )
        self._viser_look_x = self.server.gui.add_number(
            label="Look-at X", initial_value=float(look_offset[0]), step=0.1
        )
        self._viser_look_y = self.server.gui.add_number(
            label="Look-at Y", initial_value=float(look_offset[1]), step=0.1
        )
        self._viser_look_z = self.server.gui.add_number(
            label="Look-at Z", initial_value=float(look_offset[2]), step=0.1
        )
        self._viser_cam_fov = self.server.gui.add_number(
            label="FOV (degrees)",
            initial_value=float(camera_config.get("fov", 45.0)),
            step=1.0,
        )
        self._viser_set_camera = self.server.gui.add_button(label="Set camera")
        self._viser_early_termination_control = self.server.gui.add_checkbox(
            label="Early termination",
            initial_value=True,
        )

        record_config = env_config.get("viser_record", {})
        resolution = record_config.get("resolution", [960, 1080])
        self._viser_record_width = int(resolution[0])
        self._viser_record_height = int(resolution[1])
        self._viser_record_every = max(1, int(record_config.get("every_n_steps", 1)))
        default_dir = os.path.join("record_viser", self._viser_env_name)
        self._viser_record_dir = os.path.abspath(
            os.path.expanduser(str(record_config.get("output_dir", default_dir)))
        )
        self._viser_record_control = self.server.gui.add_checkbox(
            label="Record frames",
            initial_value=False,
        )
        self._viser_screenshot_button = self.server.gui.add_button(label="Save screenshot")

        self._viser_grid_size = self.server.gui.add_number(
            label="Grid cell size", initial_value=grid_cell_size, step=0.1
        )
        self._viser_grid_thickness = self.server.gui.add_number(
            label="Grid thickness", initial_value=grid_thickness, step=0.1
        )
        self._viser_ground_width = self.server.gui.add_number(
            label="Ground width", initial_value=grid_width, step=1.0
        )
        self._viser_ground_x = self.server.gui.add_number(
            label="Ground X", initial_value=float(grid_position[0]), step=0.5
        )
        self._viser_ground_y = self.server.gui.add_number(
            label="Ground Y", initial_value=float(grid_position[1]), step=0.5
        )
        self._viser_grid_color = self.server.gui.add_rgb(
            label="Grid color", initial_value=grid_color
        )

        @self._viser_play_control.on_update
        def _on_playing_change(_event):
            self._viser_playing = bool(self._viser_play_control.value)

        @self._viser_early_termination_control.on_update
        def _on_early_termination_change(_event):
            self._enable_early_termination = bool(
                self._viser_early_termination_control.value
            )

        @self._viser_set_camera.on_click
        def _on_set_camera(_event):
            self._apply_viser_camera()

        @self._viser_record_control.on_update
        def _on_recording_change(_event):
            self._viser_recording = bool(self._viser_record_control.value)

        @self._viser_screenshot_button.on_click
        def _on_screenshot(_event):
            self._capture_viser_frame(prefix="screenshot")

        def _update_grid(_event):
            cell_size = max(0.01, float(self._viser_grid_size.value))
            thickness = max(0.01, float(self._viser_grid_thickness.value))
            width = max(0.1, float(self._viser_ground_width.value))
            self._viser_grid.cell_size = cell_size
            self._viser_grid.section_size = 5.0 * cell_size
            self._viser_grid.cell_thickness = thickness
            self._viser_grid.section_thickness = thickness
            self._viser_grid.width = width
            self._viser_grid.height = width
            self._viser_grid.position = (
                float(self._viser_ground_x.value),
                float(self._viser_ground_y.value),
                float(grid_position[2]),
            )

        @self._viser_grid_size.on_update
        def _on_grid_size(event):
            _update_grid(event)

        @self._viser_grid_thickness.on_update
        def _on_grid_thickness(event):
            _update_grid(event)

        @self._viser_ground_width.on_update
        def _on_ground_width(event):
            _update_grid(event)

        @self._viser_ground_x.on_update
        def _on_ground_x(event):
            _update_grid(event)

        @self._viser_ground_y.on_update
        def _on_ground_y(event):
            _update_grid(event)

        @self._viser_grid_color.on_update
        def _on_grid_color(_event):
            color = tuple(self._viser_grid_color.value)
            self._viser_grid.cell_color = color
            self._viser_grid.section_color = color

        self._viser_playing = bool(self._viser_play_control.value)
        self._viser_recording = bool(self._viser_record_control.value)
        self._enable_early_termination = True
        return

    def _init_task_viser(self, env_config):
        """Initialize task-specific state and GUI controls."""
        return

    def _build_viser_scene(self):
        self._build_character_viser()
        self._build_task_viser()
        return

    @abstractmethod
    def _build_task_viser(self):
        """Create task objects after the simulation and server are ready."""
        raise NotImplementedError

    def _build_character_viser(self):
        char_id = self._get_char_id()
        body_names = self._engine.get_obj_body_names(char_id)
        char_file_lower = self._viser_char_file.lower()

        if "g1" in char_file_lower:
            from viser.extras import ViserUrdf
            import yourdfpy

            urdf_file = "data/assets/viser_g1/g1_29dof_rev_1_0.urdf"
            mesh_dir = "data/assets/viser_g1/meshes"
            assert os.path.isfile(urdf_file), "Missing G1 Viser URDF: {}".format(urdf_file)

            self._viser_char_root_handle = self.server.scene.add_frame(
                name="/character_root",
                position=(0.0, 0.0, 0.0),
                wxyz=(1.0, 0.0, 0.0, 0.0),
                show_axes=False,
            )
            urdf = yourdfpy.URDF.load(urdf_file, mesh_dir=mesh_dir)
            self._viser_char_urdf = ViserUrdf(
                self.server,
                urdf,
                root_node_name="/character_root",
            )
            self._viser_char_mode = "g1_urdf"
        else:
            assert self._viser_char_file.endswith(".xml"), (
                "Humanoid Viser requires an MJCF XML character file"
            )
            humanoid_visual = _MJCFHumanoidVisual(self._viser_char_file)
            self._viser_char_handles = humanoid_visual.build(
                self.server,
                body_names,
                prefix="/character",
            )
            self._viser_char_urdf = humanoid_visual
            self._viser_char_mode = "humanoid_mjcf"
        return

    def reset(self, env_ids=None):
        result = super().reset(env_ids)
        if self._viser_ready:
            self._reset_task_viser(env_ids)
            self._update_viser_scene()
        return result

    def step(self, action):
        result = super().step(action)
        if self._viser_ready:
            self._update_viser_scene()
            self._viser_step_id += 1
            if self._viser_rollout_enabled:
                self._record_viser_rollout(action)
            if (self._viser_recording
                    and self._viser_step_id % self._viser_record_every == 0):
                self._capture_viser_frame(prefix="frame")
            while not self._viser_playing:
                time.sleep(0.05)
        return result

    def _record_viser_rollout(self, action):
        env_id = self._viser_env_id
        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)[env_id]
        root_rot = self._engine.get_root_rot(char_id)[env_id]
        dof_pos = self._engine.get_dof_pos(char_id)[env_id]
        body_pos = self._engine.get_body_pos(char_id)[env_id]
        body_rot = self._engine.get_body_rot(char_id)[env_id]

        self._viser_rollout_frames.append(self._viser_step_id)
        self._viser_rollout_root_pos.append(self._to_numpy(root_pos))
        self._viser_rollout_root_rot.append(self._to_numpy(root_rot))
        self._viser_rollout_dof_pos.append(self._to_numpy(dof_pos))
        self._viser_rollout_body_pos.append(self._to_numpy(body_pos))
        self._viser_rollout_body_rot.append(self._to_numpy(body_rot))
        self._viser_rollout_actions.append(self._select_action(action, env_id))

        task_state = self._get_rollout_task_state(env_id)
        if not isinstance(task_state, dict):
            raise TypeError("_get_rollout_task_state() must return a dict")
        if len(task_state) == 0:
            raise ValueError("Task rollout state must contain at least one field")
        expected_keys = set(self._viser_rollout_task_data.keys())
        current_keys = set(task_state.keys())
        if len(self._viser_rollout_frames) > 1 and current_keys != expected_keys:
            raise ValueError(
                "Task rollout keys changed from {} to {}".format(
                    sorted(expected_keys), sorted(current_keys)
                )
            )
        for key, value in task_state.items():
            self._viser_rollout_task_data.setdefault(str(key), []).append(
                self._to_numpy(value)
            )
        return

    def _get_rollout_task_state(self, env_id):
        """Return task-specific arrays for one env; task Viser envs override it."""
        raise NotImplementedError(
            "This task Viser env must implement _get_rollout_task_state()"
        )

    @staticmethod
    def _to_numpy(value):
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value, dtype=np.float32).copy()

    @classmethod
    def _select_action(cls, action, env_id):
        action = cls._to_numpy(action)
        if action.ndim > 1:
            action = action[env_id]
        return action

    def save_viser_rollout(self):
        if not self._viser_rollout_enabled or len(self._viser_rollout_frames) == 0:
            return None

        arrays = {
            "frame_num": np.asarray(self._viser_rollout_frames, dtype=np.int64),
            "root_pos": np.stack(self._viser_rollout_root_pos).astype(np.float32),
            "root_rot": np.stack(self._viser_rollout_root_rot).astype(np.float32),
            "dof_pos": np.stack(self._viser_rollout_dof_pos).astype(np.float32),
            "body_pos": np.stack(self._viser_rollout_body_pos).astype(np.float32),
            "body_rot": np.stack(self._viser_rollout_body_rot).astype(np.float32),
            "actions": np.stack(self._viser_rollout_actions).astype(np.float32),
        }
        task_keys = sorted(self._viser_rollout_task_data.keys())
        task_dtype = []
        task_schema = {}
        for key in task_keys:
            field_values = np.stack(self._viser_rollout_task_data[key]).astype(
                np.float32
            )
            field_shape = field_values.shape[1:]
            if len(field_shape) == 0:
                task_dtype.append((key, np.float32))
            else:
                task_dtype.append((key, np.float32, field_shape))
            task_schema[key] = {
                "dtype": "float32",
                "shape": list(field_shape),
            }

        task_array = np.empty(len(self._viser_rollout_frames), dtype=task_dtype)
        for key in task_keys:
            task_array[key] = np.stack(self._viser_rollout_task_data[key]).astype(
                np.float32
            )
        arrays["task"] = task_array
        metadata = dict(self._viser_rollout_metadata)
        metadata["body_names"] = list(
            self._engine.get_obj_body_names(self._get_char_id())
        )
        metadata["task_schema"] = task_schema
        arrays["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))

        os.makedirs(os.path.dirname(self._viser_rollout_file), exist_ok=True)
        temp_file = self._viser_rollout_file + ".tmp.npz"
        np.savez_compressed(temp_file, **arrays)
        os.replace(temp_file, self._viser_rollout_file)
        print(
            "Rollout saved: {} ({} frames)".format(
                self._viser_rollout_file, len(self._viser_rollout_frames)
            )
        )
        return self._viser_rollout_file

    def _update_viser_scene(self):
        """Synchronize shared and task nodes from the selected environment."""
        self._update_viser_env_id()
        self._update_character_viser()
        self._update_task_viser()
        self._update_viser_camera()
        return

    def _update_viser_env_id(self):
        self._viser_env_id = 0
        return

    def _update_character_viser(self):
        env_id = self._viser_env_id
        char_id = self._get_char_id()

        if self._viser_char_mode == "g1_urdf":
            root_pos = self._engine.get_root_pos(char_id)[env_id].detach().cpu().numpy()
            root_rot = self._engine.get_root_rot(char_id)[env_id].detach().cpu().numpy()
            dof_pos = self._engine.get_dof_pos(char_id)[env_id].detach().cpu().numpy()
            self._viser_char_root_handle.position = root_pos
            self._viser_char_root_handle.wxyz = _xyzw_to_wxyz(root_rot)
            self._viser_char_urdf.update_cfg(dof_pos)
        else:
            body_pos = self._engine.get_body_pos(char_id)[env_id].detach().cpu().numpy()
            body_rot = self._engine.get_body_rot(char_id)[env_id].detach().cpu().numpy()
            self._viser_char_urdf.update(self._viser_char_handles, body_pos, body_rot)
        return

    @abstractmethod
    def _update_task_viser(self):
        """Synchronize task-specific scene nodes."""
        raise NotImplementedError

    def _reset_task_viser(self, env_ids):
        """Optional hook for task state that must be handled on reset."""
        return

    def _update_viser_camera(self):
        if self._viser_camera_mode.value != "track":
            return

        self._apply_viser_camera()
        return

    def _apply_viser_camera(self):
        cam_delta = np.asarray(
            [self._viser_cam_x.value, self._viser_cam_y.value, self._viser_cam_z.value],
            dtype=np.float32,
        )
        look_delta = np.asarray(
            [self._viser_look_x.value, self._viser_look_y.value, self._viser_look_z.value],
            dtype=np.float32,
        )

        if self._viser_camera_mode.value == "track":
            root_pos = self._engine.get_root_pos(self._get_char_id())[self._viser_env_id]
            root_pos = root_pos.detach().cpu().numpy()
            origin = np.array([root_pos[0], root_pos[1], 0.0], dtype=np.float32)
            camera_position = origin + cam_delta
            look_at = origin + look_delta
        else:
            camera_position = cam_delta
            look_at = look_delta

        for client in self.server.get_clients().values():
            client.camera.position = camera_position
            client.camera.look_at = look_at
            client.camera.fov = np.deg2rad(float(self._viser_cam_fov.value))
            client.camera.up_direction = (0.0, 0.0, 1.0)
        return

    def _capture_viser_frame(self, prefix):
        """Render one PNG from the first connected browser client."""
        clients = list(self.server.get_clients().values())
        if len(clients) == 0:
            return False

        os.makedirs(self._viser_record_dir, exist_ok=True)
        capture_state = self._prepare_viser_capture()
        try:
            if capture_state:
                time.sleep(0.05)
            image = clients[0].camera.get_render(
                width=self._viser_record_width,
                height=self._viser_record_height,
                transport_format="png",
            )
        finally:
            self._restore_viser_capture(capture_state)
        from PIL import Image

        if prefix == "frame":
            filename = "{:04d}.png".format(self._viser_step_id)
        else:
            filename = "{}_{:04d}.png".format(prefix, self._viser_step_id)
        Image.fromarray(image).save(os.path.join(self._viser_record_dir, filename))
        return True

    def _prepare_viser_capture(self):
        """Temporarily adjust scene visibility before capturing an image."""
        return None

    def _restore_viser_capture(self, capture_state):
        """Restore scene visibility after capturing an image."""
        return
