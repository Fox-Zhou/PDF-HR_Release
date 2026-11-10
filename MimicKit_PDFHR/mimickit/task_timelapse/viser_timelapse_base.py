import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import viser
from viser.extras import ViserUrdf
import yourdfpy

PROJECT_DIR = Path(__file__).resolve().parents[2]
MIMICKIT_DIR = PROJECT_DIR / "mimickit"
DEFAULT_URDF = PROJECT_DIR / "data/assets/viser_g1/g1_29dof_rev_1_0.urdf"
GROUND_SIZE = 100.0
GROUND_X = 0.0
GROUND_Y = 40.0
GRID_Z = 0.01
GRID_COLOR = (180, 180, 180)
BOX_COLOR = (170, 170, 150)
if str(MIMICKIT_DIR) not in sys.path:
    sys.path.insert(0, str(MIMICKIT_DIR))

from envs.viser_base_env import _MJCFHumanoidVisual


def _make_tube_segments(points, radius=0.008, sectors=10):
    """Build a small world-space tube mesh for opacity-aware task lines."""
    segments = np.asarray(points, dtype=np.float32).reshape(-1, 2, 3)
    vertices = []
    faces = []
    for start, end in segments:
        axis = end - start
        length = float(np.linalg.norm(axis))
        if length < 1e-7:
            continue
        direction = axis / length
        reference = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        if abs(float(np.dot(direction, reference))) > 0.9:
            reference = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        tangent = np.cross(direction, reference)
        tangent /= np.linalg.norm(tangent)
        bitangent = np.cross(direction, tangent)

        base = len(vertices)
        for center in (start, end):
            for sector in range(sectors):
                theta = 2.0 * np.pi * sector / sectors
                offset = radius * (
                    np.cos(theta) * tangent + np.sin(theta) * bitangent
                )
                vertices.append(center + offset)
        start_center = len(vertices)
        vertices.append(start)
        end_center = len(vertices)
        vertices.append(end)

        for sector in range(sectors):
            next_sector = (sector + 1) % sectors
            start_a = base + sector
            start_b = base + next_sector
            end_a = base + sectors + sector
            end_b = base + sectors + next_sector
            faces.append([start_a, end_a, start_b])
            faces.append([start_b, end_a, end_b])
            faces.append([start_center, start_b, start_a])
            faces.append([end_center, end_a, end_b])

    return (
        np.asarray(vertices, dtype=np.float32).reshape(-1, 3),
        np.asarray(faces, dtype=np.int32).reshape(-1, 3),
    )


def decode_rollout_metadata(metadata):
    """Decode the scalar JSON metadata stored in a rollout NPZ."""
    if isinstance(metadata, np.ndarray):
        metadata = metadata.item()
    if isinstance(metadata, bytes):
        metadata = metadata.decode("utf-8")
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    if not isinstance(metadata, dict):
        raise ValueError("Rollout metadata must decode to a dictionary")
    return dict(metadata)


def load_rollout_metadata(rollout_file):
    """Read only metadata, without decompressing the rollout frame arrays."""
    rollout_file = os.path.abspath(os.path.expanduser(str(rollout_file)))
    if not rollout_file.lower().endswith(".npz"):
        raise ValueError("Rollout file must use the .npz extension")
    with np.load(rollout_file, allow_pickle=False) as archive:
        if "metadata" not in archive.files:
            raise ValueError("Rollout is missing metadata")
        return decode_rollout_metadata(archive["metadata"])


def load_rollout(rollout_file):
    """Read and validate the task-independent rollout NPZ schema."""
    rollout_file = os.path.abspath(os.path.expanduser(str(rollout_file)))
    if not rollout_file.lower().endswith(".npz"):
        raise ValueError("Rollout file must use the .npz extension")
    with np.load(rollout_file, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}

    if "metadata" not in data:
        raise ValueError("Rollout is missing metadata")
    data["metadata"] = decode_rollout_metadata(data["metadata"])

    required = (
        "frame_num",
        "root_pos",
        "root_rot",
        "dof_pos",
        "actions",
        "task",
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError("Rollout is missing keys: {}".format(", ".join(missing)))
    num_frames = len(data["frame_num"])
    for key in required[1:]:
        if len(data[key]) != num_frames:
            raise ValueError("Rollout key '{}' has inconsistent length".format(key))
    task_fields = data["task"].dtype.names
    if task_fields is None:
        raise ValueError("Rollout 'task' must be a structured NumPy array")
    task_schema = data["metadata"].get("task_schema", {})
    if set(task_fields) != set(task_schema.keys()):
        raise ValueError("Rollout task fields do not match metadata.task_schema")
    return data


class ViserTimelapseBase:
    """Task-independent offline timelapse viewer for one recorded rollout."""

    def __init__(self, rollout_file):
        self.rollout_file = self.resolve_rollout_file(rollout_file)
        self.rollout = load_rollout(self.rollout_file)
        self.metadata = self.rollout["metadata"]
        self.frame_nums = np.asarray(self.rollout["frame_num"], dtype=np.int64)
        if self.frame_nums.size == 0:
            raise ValueError("Rollout contains no frames")
        if len(np.unique(self.frame_nums)) != len(self.frame_nums):
            raise ValueError("frame_num values must be unique")
        self._frame_to_index = {
            int(frame_num): index for index, frame_num in enumerate(self.frame_nums)
        }

        self.output_file = self.get_output_file(self.rollout_file)
        self.server = viser.ViserServer()
        self.server.scene.set_up_direction("+z")
        self._robot_frame_slots = {}
        self._task_frame_slots = {}
        self._next_robot_slot_id = 0
        self._next_task_slot_id = 0

        self._load_robot_asset()
        self._build_ground()
        self._build_gui()
        self._build_task_scene()
        first_frame = int(self.frame_nums[0])
        self.create_robot_frame_slot(first_frame)
        self.create_task_frame_slot(first_frame)

        seed = self.metadata.get("seed", "unknown")
        print("Loaded {} frames (seed={})".format(len(self.frame_nums), seed))
        print("Timelapse image: {}".format(self.output_file))

    @staticmethod
    def resolve_rollout_file(rollout_file):
        path = Path(rollout_file).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        if path.suffix.lower() != ".npz":
            raise ValueError("Rollout path must point to a .npz file")
        if not path.is_file():
            raise FileNotFoundError("Rollout file does not exist: {}".format(path))
        return path.resolve()

    @staticmethod
    def get_output_file(rollout_file):
        output_dir = PROJECT_DIR / "record/img/timelapse"
        return output_dir / (Path(rollout_file).stem + ".png")

    def _load_robot_asset(self):
        char_file = str(self.metadata.get("char_file", ""))
        if "g1" in char_file.lower():
            self._robot_mode = "g1_urdf"
            if not DEFAULT_URDF.is_file():
                raise FileNotFoundError(
                    "Missing G1 Viser URDF: {}".format(DEFAULT_URDF)
                )
            self._urdf = yourdfpy.URDF.load(
                str(DEFAULT_URDF), mesh_dir=str(DEFAULT_URDF.parent / "meshes")
            )
            return

        char_path = Path(char_file)
        if not char_path.is_absolute():
            char_path = PROJECT_DIR / char_path
        if char_path.suffix.lower() != ".xml" or not char_path.is_file():
            raise ValueError("Unsupported Humanoid character file: {}".format(char_file))
        for key in ("body_pos", "body_rot"):
            if key not in self.rollout:
                raise ValueError("Humanoid rollout is missing '{}'".format(key))
            if len(self.rollout[key]) != len(self.frame_nums):
                raise ValueError("Humanoid rollout '{}' has inconsistent length".format(key))
        body_names = self.metadata.get("body_names", [])
        if len(body_names) == 0:
            raise ValueError("Humanoid rollout metadata is missing body_names")
        self._robot_mode = "humanoid_mjcf"
        self._body_names = list(body_names)
        self._mjcf_visual = _MJCFHumanoidVisual(str(char_path))

    def _build_ground(self):
        self._grid_handle = self.server.scene.add_grid(
            name="/grid",
            width=GROUND_SIZE,
            height=GROUND_SIZE,
            position=(GROUND_X, GROUND_Y, GRID_Z),
            cell_size=1.0,
            cell_thickness=2.0,
            section_size=1.0,
            cell_color=GRID_COLOR,
            section_color=GRID_COLOR,
            plane="xy",
            shadow_opacity=0.2,
        )
        self._floor_handle = self.server.scene.add_box(
            name="/box",
            dimensions=(GROUND_SIZE, GROUND_SIZE, GROUND_SIZE),
            position=(GROUND_X, GROUND_Y, -GROUND_SIZE // 2),
            color=BOX_COLOR,
            material="standard",
            side="double",
            cast_shadow=False,
            flat_shading=True,
            receive_shadow=True,
        )

    def _build_gui(self):
        self._robot_frames_folder = self.server.gui.add_folder("Robot Frames")
        with self._robot_frames_folder:
            self._add_robot_frame_button = self.server.gui.add_button(
                label="Add robot"
            )

        @self._add_robot_frame_button.on_click
        def _on_add_robot_frame(_event):
            self.create_robot_frame_slot(
                self._get_next_frame_num(self._robot_frame_slots)
            )

        self._task_frames_folder = self.server.gui.add_folder("Task Frames")
        with self._task_frames_folder:
            self._add_task_frame_button = self.server.gui.add_button(
                label="Add task"
            )

        @self._add_task_frame_button.on_click
        def _on_add_task_frame(_event):
            self.create_task_frame_slot(
                self._get_next_frame_num(self._task_frame_slots)
            )

        with self.server.gui.add_folder("Camera"):
            self._free_camera = self.server.gui.add_checkbox(
                label="Free camera", initial_value=True
            )
            self._cam_x = self.server.gui.add_number(
                label="Camera X", initial_value=4.0, step=0.1
            )
            self._cam_y = self.server.gui.add_number(
                label="Camera Y", initial_value=-6.0, step=0.1
            )
            self._cam_z = self.server.gui.add_number(
                label="Camera Z", initial_value=3.0, step=0.1
            )
            self._look_x = self.server.gui.add_number(
                label="Look-at X", initial_value=0.0, step=0.1
            )
            self._look_y = self.server.gui.add_number(
                label="Look-at Y", initial_value=0.0, step=0.1
            )
            self._look_z = self.server.gui.add_number(
                label="Look-at Z", initial_value=0.9, step=0.1
            )
            self._fov = self.server.gui.add_number(
                label="FOV (degrees)", initial_value=45.0, step=1.0
            )
            self._resolution_width = self.server.gui.add_number(
                label="Resolution width", initial_value=1920, step=100
            )
            self._resolution_height = self.server.gui.add_number(
                label="Resolution height", initial_value=1080, step=100
            )
            self._set_camera_button = self.server.gui.add_button(label="Set camera")
            self._save_image_button = self.server.gui.add_button(label="Save image")

        @self._set_camera_button.on_click
        def _on_set_camera(_event):
            self._apply_camera(force=True)

        @self._free_camera.on_update
        def _on_free_camera(_event):
            self._apply_camera()

        @self._save_image_button.on_click
        def _on_save_image(_event):
            self.save_image()

        with self.server.gui.add_folder("Ground"):
            self._grid_thickness = self.server.gui.add_slider(
                label="Grid thickness",
                initial_value=1.0,
                min=0.5,
                max=5.0,
                step=0.1,
            )
            self._grid_size = self.server.gui.add_slider(
                label="Grid size",
                initial_value=2.0,
                min=0.5,
                max=5.0,
                step=0.1,
            )
            self._grid_color = self.server.gui.add_rgb(
                label="Grid color", initial_value=GRID_COLOR
            )
            self._box_color = self.server.gui.add_rgb(
                label="Box color", initial_value=BOX_COLOR
            )
            self._box_x = self.server.gui.add_number(
                label="Box position X", initial_value=GROUND_X, step=5.0
            )
            self._box_y = self.server.gui.add_number(
                label="Box position Y", initial_value=GROUND_Y, step=5.0
            )
            self._show_ground = self.server.gui.add_checkbox(
                label="Visualize background box", initial_value=True
            )

        @self._grid_thickness.on_update
        def _on_grid_thickness(_event):
            value = float(self._grid_thickness.value)
            self._grid_handle.cell_thickness = value
            self._grid_handle.section_thickness = value

        @self._grid_size.on_update
        def _on_grid_size(_event):
            value = float(self._grid_size.value)
            self._grid_handle.cell_size = value
            self._grid_handle.section_size = value

        @self._grid_color.on_update
        def _on_grid_color(_event):
            color = tuple(self._grid_color.value)
            self._grid_handle.cell_color = color
            self._grid_handle.section_color = color

        @self._box_color.on_update
        def _on_box_color(_event):
            self._floor_handle.color = tuple(self._box_color.value)

        @self._box_x.on_update
        def _on_box_x(_event):
            x = float(self._box_x.value)
            self._floor_handle.position = (
                x,
                self._floor_handle.position[1],
                self._floor_handle.position[2],
            )
            self._grid_handle.position = (
                x,
                self._grid_handle.position[1],
                self._grid_handle.position[2],
            )

        @self._box_y.on_update
        def _on_box_y(_event):
            y = float(self._box_y.value)
            self._floor_handle.position = (
                self._floor_handle.position[0],
                y,
                self._floor_handle.position[2],
            )
            self._grid_handle.position = (
                self._grid_handle.position[0],
                y,
                self._grid_handle.position[2],
            )

        @self._show_ground.on_update
        def _on_show_ground(_event):
            self._floor_handle.visible = bool(self._show_ground.value)

    def _get_next_frame_num(self, slots):
        if len(slots) == 0:
            return int(self.frame_nums[0])
        current = max(
            int(round(slot["frame_handle"].value))
            for slot in slots.values()
        )
        later = self.frame_nums[self.frame_nums > current]
        if len(later) > 0:
            return int(later[0])
        return int(self.frame_nums[-1])

    def create_robot_frame_slot(self, frame_num, opacity=1.0):
        slot_id = self._next_robot_slot_id
        self._next_robot_slot_id += 1
        with self._robot_frames_folder:
            frame_handle = self.server.gui.add_number(
                label="Robot frame {}".format(slot_id),
                initial_value=int(frame_num),
                min=int(self.frame_nums[0]),
                max=int(self.frame_nums[-1]),
                step=1,
            )
            opacity_handle = self.server.gui.add_number(
                label="Robot opacity {}".format(slot_id),
                initial_value=float(opacity),
                min=0.0,
                max=1.0,
                step=0.05,
            )
            remove_button = self.server.gui.add_button(
                label="Delete robot {}".format(slot_id)
            )

        robot_prefix = "/timelapse/robot_{:04d}".format(slot_id)
        if self._robot_mode == "g1_urdf":
            root_handle = self.server.scene.add_frame(
                name=robot_prefix, show_axes=False
            )
            robot = ViserUrdf(
                self.server,
                self._urdf,
                root_node_name=robot_prefix,
            )
            body_handles = None
        else:
            root_handle = None
            robot = None
            body_handles = self._mjcf_visual.build(
                self.server, self._body_names, prefix=robot_prefix
            )
        self._robot_frame_slots[slot_id] = {
            "frame_handle": frame_handle,
            "opacity_handle": opacity_handle,
            "remove_button": remove_button,
            "root_handle": root_handle,
            "robot": robot,
            "body_handles": body_handles,
        }

        @frame_handle.on_update
        def _on_frame_update(_event):
            self.update_robot_frame_slot(slot_id)

        @opacity_handle.on_update
        def _on_opacity_update(_event):
            self.update_robot_frame_slot(slot_id)

        @remove_button.on_click
        def _on_remove(_event):
            self.remove_robot_frame_slot(slot_id)

        self.update_robot_frame_slot(slot_id)
        return slot_id

    def update_robot_frame_slot(self, slot_id):
        slot = self._robot_frame_slots.get(slot_id)
        if slot is None:
            return
        frame_num = int(round(slot["frame_handle"].value))
        index = self._frame_to_index.get(frame_num)
        if index is None:
            self._set_robot_visible(slot, False)
            print("Frame {} is not present in this rollout".format(frame_num))
            return

        if self._robot_mode == "g1_urdf":
            root_xyzw = np.asarray(self.rollout["root_rot"][index], dtype=np.float32)
            slot["root_handle"].position = np.asarray(
                self.rollout["root_pos"][index], dtype=np.float32
            )
            slot["root_handle"].wxyz = root_xyzw[[3, 0, 1, 2]]
            slot["robot"].update_cfg(
                np.asarray(self.rollout["dof_pos"][index], dtype=np.float32)
            )
        else:
            self._mjcf_visual.update(
                slot["body_handles"],
                np.asarray(self.rollout["body_pos"][index], dtype=np.float32),
                np.asarray(self.rollout["body_rot"][index], dtype=np.float32),
            )
        self._set_robot_visible(slot, True)
        opacity = float(
            np.clip(slot["opacity_handle"].value, 0.0, 1.0)
        )
        self._set_robot_opacity(slot, opacity)

    def create_task_frame_slot(self, frame_num, opacity=1.0):
        slot_id = self._next_task_slot_id
        self._next_task_slot_id += 1
        with self._task_frames_folder:
            frame_handle = self.server.gui.add_number(
                label="Task frame {}".format(slot_id),
                initial_value=int(frame_num),
                min=int(self.frame_nums[0]),
                max=int(self.frame_nums[-1]),
                step=1,
            )
            opacity_handle = self.server.gui.add_number(
                label="Task opacity {}".format(slot_id),
                initial_value=float(opacity),
                min=0.0,
                max=1.0,
                step=0.05,
            )
            remove_button = self.server.gui.add_button(
                label="Delete task {}".format(slot_id)
            )

        self._task_frame_slots[slot_id] = {
            "frame_handle": frame_handle,
            "opacity_handle": opacity_handle,
            "remove_button": remove_button,
        }

        @frame_handle.on_update
        def _on_frame_update(_event):
            self.update_task_frame_slot(slot_id)

        @opacity_handle.on_update
        def _on_opacity_update(_event):
            self.update_task_frame_slot(slot_id)

        @remove_button.on_click
        def _on_remove(_event):
            self.remove_task_frame_slot(slot_id)

        self._create_task_slot(slot_id)
        self.update_task_frame_slot(slot_id)
        return slot_id

    def update_task_frame_slot(self, slot_id):
        slot = self._task_frame_slots.get(slot_id)
        if slot is None:
            return
        frame_num = int(round(slot["frame_handle"].value))
        index = self._frame_to_index.get(frame_num)
        if index is None:
            self._set_task_slot_visible(slot_id, False)
            print("Frame {} is not present in this rollout".format(frame_num))
            return
        opacity = float(np.clip(slot["opacity_handle"].value, 0.0, 1.0))
        self._update_task_slot(slot_id, index, opacity)

    def _set_robot_visible(self, slot, visible):
        if self._robot_mode == "g1_urdf":
            slot["robot"].show_visual = visible
        else:
            for _body_idx, _spec, handle in slot["body_handles"]:
                handle.visible = visible

    def _set_robot_opacity(self, slot, opacity):
        if self._robot_mode == "g1_urdf":
            handles = getattr(slot["robot"], "_meshes", [])
        else:
            handles = [entry[2] for entry in slot["body_handles"]]
        for handle in handles:
            handle.opacity = opacity

    def _add_task_line_mesh(
        self, name, points, color, opacity, visible, radius=0.008
    ):
        vertices, faces = _make_tube_segments(points, radius=radius)
        if len(faces) == 0:
            return None
        return self.server.scene.add_mesh_simple(
            name=name,
            vertices=vertices,
            faces=faces,
            color=color,
            opacity=float(opacity),
            material="standard",
            side="double",
            cast_shadow=False,
            receive_shadow=False,
            visible=bool(visible),
        )

    def remove_robot_frame_slot(self, slot_id):
        slot = self._robot_frame_slots.pop(slot_id, None)
        if slot is None:
            return
        slot["frame_handle"].remove()
        slot["opacity_handle"].remove()
        slot["remove_button"].remove()
        if self._robot_mode == "g1_urdf":
            slot["robot"].remove()
            slot["root_handle"].remove()
        else:
            for _body_idx, _spec, handle in slot["body_handles"]:
                handle.remove()

    def remove_task_frame_slot(self, slot_id):
        slot = self._task_frame_slots.pop(slot_id, None)
        if slot is None:
            return
        self._remove_task_slot(slot_id)
        slot["frame_handle"].remove()
        slot["opacity_handle"].remove()
        slot["remove_button"].remove()

    def _apply_camera(self, force=False):
        if self._free_camera.value and not force:
            return
        for client in self.server.get_clients().values():
            client.camera.position = np.asarray(
                [self._cam_x.value, self._cam_y.value, self._cam_z.value]
            )
            client.camera.look_at = np.asarray(
                [self._look_x.value, self._look_y.value, self._look_z.value]
            )
            client.camera.fov = np.deg2rad(float(self._fov.value))
            client.camera.up_direction = (0.0, 0.0, 1.0)

    def save_image(self):
        clients = list(self.server.get_clients().values())
        if len(clients) == 0:
            return None
        image = clients[0].camera.get_render(
            width=int(self._resolution_width.value),
            height=int(self._resolution_height.value),
            transport_format="png",
        )
        os.makedirs(self.output_file.parent, exist_ok=True)
        Image.fromarray(image).save(self.output_file)
        print("Image saved: {}".format(self.output_file))
        return self.output_file

    def _build_task_scene(self):
        """Hook for task-specific static scene elements."""
        return

    def _create_task_slot(self, slot_id):
        """Hook for task objects belonging to a newly created frame slot."""
        return

    def _update_task_slot(self, slot_id, frame_index, opacity):
        """Hook for updating task objects at one recorded frame."""
        return

    def _set_task_slot_visible(self, slot_id, visible):
        """Hook for hiding task objects when a requested frame is absent."""
        return

    def _remove_task_slot(self, slot_id):
        """Hook for deleting task objects belonging to one frame slot."""
        return

    def run_forever(self):
        while True:
            time.sleep(0.1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rollout_file",
        required=True,
        help="Path to a rollout NPZ file",
    )
    args = parser.parse_args()
    viewer = ViserTimelapseBase(args.rollout_file)
    viewer.run_forever()


if __name__ == "__main__":
    main()
