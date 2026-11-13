import argparse
import os
import queue
import re
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import open3d as o3d
from open3d.visualization import gui, rendering
import serial
import serial.tools.list_ports


# ============================================================
# Basic settings
# ============================================================

WINDOW_TITLE = "VL53L4CD 3-Sensor Plane Viewer"
LOG_TAG = "VL53L4CD_3D"

# All distances are in mm.
CALIBRATION_COEFFICIENTS = {
    1: (0.9513, -7.7578),
    2: (0.9719, -9.8247),
    3: (0.9631, -8.8797),
}

# Robot front cover center / coordinate origin.
FRONT_PLATE_CENTER_MM = np.array(
    [0.0, 0.0, 0.0],
    dtype=np.float64,
)


# ============================================================
# Sensor geometry
#
# Coordinate system:
#   Origin: center of the robot front plate
#   X: right
#   Y: up
#   Z: forward
#
# IMPORTANT:
# All values below are already in mm.
# Do NOT multiply these values by 10.
# ============================================================

SENSOR_POSITIONS_MM = {
    # S1: upper sensor
    1: np.array([0.0, 14.8, 10.8], dtype=np.float64),

    # S2: robot left side
    2: np.array([6.7, -5.5, 12.3], dtype=np.float64),

    # S3: robot right side
    3: np.array([-8.7, -5.5, 12.3], dtype=np.float64),
}

SENSOR_DIRECTIONS = {
    1: np.array([0.0, 0.259, 0.966], dtype=np.float64),
    2: np.array([0.5, 0.0, 0.866], dtype=np.float64),
    3: np.array([-0.5, 0.0, 0.866], dtype=np.float64),
}

SENSOR_COLORS = {
    1: [1.0, 0.20, 0.20],
    2: [0.20, 1.0, 0.20],
    3: [0.20, 0.50, 1.0],
}

OBSTACLE_COLORS = {
    1: [1.0, 0.0, 0.0],
    2: [0.0, 1.0, 0.0],
    3: [0.0, 0.45, 1.0],
}

PLANE_COLOR = [1.0, 0.75, 0.0]
NORMAL_COLOR = [1.0, 1.0, 0.0]
DISTANCE_LINE_COLOR = [1.0, 1.0, 1.0]


# ============================================================
# Serial log parsing
#
# Expected serial output example:
#
# S1: VALID | distance=356 mm ...
# S2: VALID | distance=420 mm ...
# S3: VALID | distance=500 mm ...
# ============================================================

VALID_PATTERN = re.compile(
    r"S(?P<sensor>[1-3]):\s*VALID\s*\|\s*distance=(?P<distance>\d+)\s*mm",
    re.IGNORECASE,
)

INVALID_PATTERN = re.compile(
    r"S(?P<sensor>[1-3]):\s*INVALID",
    re.IGNORECASE,
)


# ============================================================
# Utility functions
# ============================================================

def normalize(vector):
    """Return a normalized vector."""
    vector = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(vector)

    if norm < 1e-12:
        return vector.copy()

    return vector / norm


def calibrate_distance(sensor_id, raw_distance_mm):
    """
    Apply calibration:

        calibrated_distance_mm = scale * raw_distance_mm + offset
    """
    scale, offset = CALIBRATION_COEFFICIENTS[sensor_id]
    return scale * raw_distance_mm + offset


def format_distance(distance_mm):
    """Format a distance for the information panel."""
    if distance_mm is None:
        return "--"

    return f"{distance_mm:.1f} mm"


def rotation_from_z_axis_to_direction(direction):
    """
    Open3D arrows are created along +Z by default.

    This function generates a rotation matrix that rotates +Z
    toward the requested direction vector.
    """
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    direction = normalize(direction)

    dot_value = np.clip(np.dot(z_axis, direction), -1.0, 1.0)
    angle = np.arccos(dot_value)

    if abs(angle) < 1e-10:
        return np.eye(3)

    rotation_axis = np.cross(z_axis, direction)
    rotation_axis_norm = np.linalg.norm(rotation_axis)

    # Direction is almost exactly -Z.
    if rotation_axis_norm < 1e-10:
        return o3d.geometry.get_rotation_matrix_from_axis_angle(
            np.array([np.pi, 0.0, 0.0], dtype=np.float64)
        )

    rotation_axis = rotation_axis / rotation_axis_norm

    return o3d.geometry.get_rotation_matrix_from_axis_angle(
        rotation_axis * angle
    )


def create_arrow(
    position_mm,
    direction,
    color,
    cylinder_radius_mm,
    cone_radius_mm,
    cylinder_height_mm,
    cone_height_mm,
):
    """Create an arrow starting at position_mm."""
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=cylinder_radius_mm,
        cone_radius=cone_radius_mm,
        cylinder_height=cylinder_height_mm,
        cone_height=cone_height_mm,
    )

    arrow.compute_vertex_normals()
    arrow.paint_uniform_color(color)

    rotation_matrix = rotation_from_z_axis_to_direction(direction)

    arrow.rotate(
        rotation_matrix,
        center=np.array([0.0, 0.0, 0.0], dtype=np.float64),
    )

    arrow.translate(position_mm)

    return arrow


def create_sensor_body(position_mm, color):
    """Create a small sphere representing one sensor."""
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.7)
    sphere.compute_vertex_normals()
    sphere.paint_uniform_color(color)
    sphere.translate(position_mm)

    return sphere


def create_sensor_arrow(position_mm, direction, color):
    """Create a short direction arrow for one sensor."""
    return create_arrow(
        position_mm=position_mm,
        direction=direction,
        color=color,
        cylinder_radius_mm=0.75,
        cone_radius_mm=1.60,
        cylinder_height_mm=8.0,
        cone_height_mm=3.5,
    )


def create_front_plate():
    """
    Create the robot front plate.

    The dimensions are 26 mm × 34 mm.
    """
    width_mm = 26.0
    height_mm = 34.0
    z_mm = 0.0

    vertices = np.array(
        [
            [-width_mm / 2, -height_mm / 2, z_mm],
            [width_mm / 2, -height_mm / 2, z_mm],
            [width_mm / 2, height_mm / 2, z_mm],
            [-width_mm / 2, height_mm / 2, z_mm],
        ],
        dtype=np.float64,
    )

    triangles = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [2, 1, 0],
            [3, 2, 0],
        ],
        dtype=np.int32,
    )

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.32, 0.32, 0.32])

    return mesh


def create_plane_mesh(points_mm):
    """Create a double-sided triangle plane from three points."""
    vertices = np.asarray(points_mm, dtype=np.float64)

    triangles = np.array(
        [
            [0, 1, 2],
            [0, 2, 1],
        ],
        dtype=np.int32,
    )

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(PLANE_COLOR)

    return mesh


def create_line_set(start_point_mm, end_point_mm, color):
    """Create one colored line segment."""
    points = np.asarray(
        [start_point_mm, end_point_mm],
        dtype=np.float64,
    )

    lines = np.asarray([[0, 1]], dtype=np.int32)

    colors = np.asarray([color], dtype=np.float64)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


# ============================================================
# Shared sensor state
# ============================================================

class SharedSensorData:
    def __init__(self):
        self.lock = threading.Lock()

        self.raw_distance_mm = {
            1: None,
            2: None,
            3: None,
        }

        self.calibrated_distance_mm = {
            1: None,
            2: None,
            3: None,
        }

        self.status = {
            1: "WAITING",
            2: "WAITING",
            3: "WAITING",
        }

        self.last_update_time = {
            1: None,
            2: None,
            3: None,
        }

    def update_valid(self, sensor_id, raw_distance_mm):
        calibrated_distance_mm = calibrate_distance(
            sensor_id,
            raw_distance_mm,
        )

        with self.lock:
            self.raw_distance_mm[sensor_id] = raw_distance_mm
            self.calibrated_distance_mm[sensor_id] = calibrated_distance_mm
            self.status[sensor_id] = "VALID"
            self.last_update_time[sensor_id] = time.time()

        return calibrated_distance_mm

    def update_invalid(self, sensor_id):
        with self.lock:
            self.raw_distance_mm[sensor_id] = None
            self.calibrated_distance_mm[sensor_id] = None
            self.status[sensor_id] = "INVALID"

    def get_calibrated_distances(self):
        with self.lock:
            return dict(self.calibrated_distance_mm)

    def get_status(self):
        with self.lock:
            return dict(self.status)


# ============================================================
# Serial reader thread
# ============================================================

class SerialReader(threading.Thread):
    def __init__(self, port, baudrate, shared_data, log_queue):
        super().__init__(daemon=True)

        self.port = port
        self.baudrate = baudrate
        self.shared_data = shared_data
        self.log_queue = log_queue

        self.stop_event = threading.Event()
        self.ser = None

    def stop(self):
        """Stop the serial reader."""
        self.stop_event.set()

        if self.ser is not None and self.ser.is_open:
            try:
                self.ser.close()
            except Exception:
                pass

    def run(self):
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=0.5,
            )

            self.log_queue.put(
                f"Connected: {self.port}, baud rate: {self.baudrate}"
            )

            # ESP32 may reboot after serial connection.
            time.sleep(1.5)

        except serial.SerialException as error:
            self.log_queue.put(f"Serial open failed: {error}")
            return

        while not self.stop_event.is_set():
            try:
                raw_line = self.ser.readline()

                if not raw_line:
                    continue

                line = raw_line.decode(
                    "utf-8",
                    errors="ignore",
                ).strip()

                if not line:
                    continue

                valid_match = VALID_PATTERN.search(line)

                if valid_match is not None:
                    sensor_id = int(valid_match.group("sensor"))
                    raw_distance_mm = int(valid_match.group("distance"))

                    self.shared_data.update_valid(
                        sensor_id,
                        raw_distance_mm,
                    )

                    continue

                invalid_match = INVALID_PATTERN.search(line)

                if invalid_match is not None:
                    sensor_id = int(invalid_match.group("sensor"))

                    self.shared_data.update_invalid(sensor_id)

            except serial.SerialException as error:
                self.log_queue.put(f"Serial communication error: {error}")
                break

            except Exception as error:
                self.log_queue.put(f"Serial parse error: {error}")

        if self.ser is not None and self.ser.is_open:
            try:
                self.ser.close()
            except Exception:
                pass


# ============================================================
# Open3D visualizer
# ============================================================

class VL53L3DVisualizer:
    def __init__(self, shared_data, log_queue):
        self.shared_data = shared_data
        self.log_queue = log_queue

        self.normal_label_3d = None
        self.last_signature = None
        self.latest_plane_data = None
        self.record_queue = queue.Queue()

        self.record_thread = threading.Thread(
            target=self.record_to_excel_worker,
            daemon=True,
        )
        self.record_thread.start()

        self.app = gui.Application.instance

        self.window = self.app.create_window(
            WINDOW_TITLE,
            1400,
            850,
        )

        # ----------------------------------------------------
        # 3D scene
        # ----------------------------------------------------

        self.scene_widget = gui.SceneWidget()

        self.scene_widget.scene = rendering.Open3DScene(
            self.window.renderer
        )

        self.scene_widget.scene.set_background(
            [0.05, 0.05, 0.07, 1.0]
        )

        self.scene_widget.scene.set_lighting(
            rendering.Open3DScene.LightingProfile.MED_SHADOWS,
            np.array(
                [0.577, -0.577, -0.577],
                dtype=np.float32,
            ),
        )

        # ----------------------------------------------------
        # Right-side English information panel
        # ----------------------------------------------------

        em = int(self.window.theme.font_size)

        self.info_panel = gui.Vert(
            int(0.4 * em),
            gui.Margins(
                int(0.8 * em),
                int(0.8 * em),
                int(0.8 * em),
                int(0.8 * em),
            ),
        )

        self.info_panel.background_color = gui.Color(
            0.10,
            0.10,
            0.13,
            1.0,
        )

        self.info_label = gui.Label(
            "SENSOR / PLANE INFO\n\n"
            "S1 Distance: --\n"
            "S2 Distance: --\n"
            "S3 Distance: --\n\n"
            "Plane Normal:\n"
            "--\n\n"
            "Center to Plane:\n"
            "--\n\n"
            "Status: Waiting"
        )

        # White text in right panel.
        self.info_label.text_color = gui.Color(
            1.0,
            1.0,
            1.0,
            1.0,
        )

        self.info_panel.add_child(self.info_label)

        self.window.add_child(self.scene_widget)
        self.window.add_child(self.info_panel)

        self.last_tick_error = None

        self.window.set_on_layout(self.on_layout)
        self.window.set_on_tick_event(self.on_tick)
        self.window.set_on_key(self.on_key_event)

        # ----------------------------------------------------
        # Materials
        # ----------------------------------------------------

        self.mesh_material = rendering.MaterialRecord()
        self.mesh_material.shader = "defaultLit"

        self.point_material = rendering.MaterialRecord()
        self.point_material.shader = "defaultUnlit"
        self.point_material.point_size = 10.0 * self.window.scaling

        self.line_material = rendering.MaterialRecord()
        self.line_material.shader = "unlitLine"
        self.line_material.line_width = 2.5 * self.window.scaling

        # ----------------------------------------------------
        # Static geometry
        # ----------------------------------------------------

        # ----------------------------------------------------
        # Thin XYZ coordinate axes
        # ----------------------------------------------------

        axis_length_mm = 50.0

        # X axis: red
        axis_x = create_arrow(
            position_mm=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            direction=np.array([1.0, 0.0, 0.0], dtype=np.float64),
            color=[1.0, 0.0, 0.0],
            cylinder_radius_mm=0.45,
            cone_radius_mm=1.20,
            cylinder_height_mm=42.0,
            cone_height_mm=8.0,
        )

        # Y axis: green
        axis_y = create_arrow(
            position_mm=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            direction=np.array([0.0, 1.0, 0.0], dtype=np.float64),
            color=[0.0, 1.0, 0.0],
            cylinder_radius_mm=0.45,
            cone_radius_mm=1.20,
            cylinder_height_mm=42.0,
            cone_height_mm=8.0,
        )

        # Z axis: blue
        axis_z = create_arrow(
            position_mm=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            direction=np.array([0.0, 0.0, 1.0], dtype=np.float64),
            color=[0.0, 0.45, 1.0],
            cylinder_radius_mm=0.45,
            cone_radius_mm=1.20,
            cylinder_height_mm=42.0,
            cone_height_mm=8.0,
        )

        front_plate = create_front_plate()

        self.scene_widget.scene.add_geometry(
            "axis_x",
            axis_x,
            self.mesh_material,
        )

        self.scene_widget.scene.add_geometry(
            "axis_y",
            axis_y,
            self.mesh_material,
        )

        self.scene_widget.scene.add_geometry(
            "axis_z",
            axis_z,
            self.mesh_material,
        )

        self.scene_widget.scene.add_geometry(
            "front_plate",
            front_plate,
            self.mesh_material,
        )

        for sensor_id in (1, 2, 3):
            sensor_position_mm = SENSOR_POSITIONS_MM[sensor_id]
            sensor_direction = SENSOR_DIRECTIONS[sensor_id]
            sensor_color = SENSOR_COLORS[sensor_id]

            sensor_body = create_sensor_body(
                sensor_position_mm,
                sensor_color,
            )

            sensor_arrow = create_sensor_arrow(
                sensor_position_mm,
                sensor_direction,
                sensor_color,
            )

            self.scene_widget.scene.add_geometry(
                f"sensor_body_{sensor_id}",
                sensor_body,
                self.mesh_material,
            )

            self.scene_widget.scene.add_geometry(
                f"sensor_arrow_{sensor_id}",
                sensor_arrow,
                self.mesh_material,
            )

        # Add an empty point cloud initially.
        empty_cloud = o3d.geometry.PointCloud()

        self.scene_widget.scene.add_geometry(
            "obstacle_points",
            empty_cloud,
            self.point_material,
        )

        # ----------------------------------------------------
        # Camera setup
        # ----------------------------------------------------

        bounding_box = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=np.array(
                [-100.0, -100.0, -50.0],
                dtype=np.float64,
            ),
            max_bound=np.array(
                [100.0, 100.0, 1200.0],
                dtype=np.float64,
            ),
        )

        self.scene_widget.setup_camera(
            60.0,
            bounding_box,
            np.array(
                [0.0, 0.0, 250.0],
                dtype=np.float32,
            ),
        )

        # Important:
        # Do not use center= / eye= / up= keyword arguments.
        # Your Open3D version expects positional float32 arrays.
        # Camera orbit / drag center is the coordinate origin (0, 0, 0).
        self.scene_widget.look_at(
            np.array(
                [0.0, 0.0, 0.0],
                dtype=np.float32,
            ),
            np.array(
                [500.0, -400.0, 650.0],
                dtype=np.float32,
            ),
            np.array(
                [0.0, 1.0, 0.0],
                dtype=np.float32,
            ),
        )

    def on_layout(self, layout_context):
        """Lay out the 3D scene and right information panel."""
        rect = self.window.content_rect

        panel_width = int(270 * self.window.scaling)

        self.scene_widget.frame = gui.Rect(
            rect.x,
            rect.y,
            rect.width - panel_width,
            rect.height,
        )

        self.info_panel.frame = gui.Rect(
            rect.x + rect.width - panel_width,
            rect.y,
            panel_width,
            rect.height,
        )

    def calculate_obstacle_points(self, distances_mm=None):
        """
        Calculate obstacle points:

            obstacle_point =
                sensor_position_mm +
                calibrated_distance_mm * sensor_direction_unit

        All values are in mm.
        """
        if distances_mm is None:
            distances_mm = self.shared_data.get_calibrated_distances()

        points_mm = []
        colors = []

        for sensor_id in (1, 2, 3):
            distance_mm = distances_mm[sensor_id]

            if distance_mm is None:
                continue

            sensor_position_mm = SENSOR_POSITIONS_MM[sensor_id]

            sensor_direction_unit = normalize(
                SENSOR_DIRECTIONS[sensor_id]
            )

            obstacle_point_mm = (
                sensor_position_mm
                + distance_mm * sensor_direction_unit
            )

            points_mm.append(obstacle_point_mm)
            colors.append(OBSTACLE_COLORS[sensor_id])

        return points_mm, colors

    @staticmethod
    def calculate_plane_normal(points_mm):
        """Calculate a unit plane normal from three points."""
        if len(points_mm) != 3:
            return None

        points_mm = np.asarray(points_mm, dtype=np.float64)

        vector_1 = points_mm[1] - points_mm[0]
        vector_2 = points_mm[2] - points_mm[0]

        normal_vector = np.cross(vector_1, vector_2)
        normal_norm = np.linalg.norm(normal_vector)

        if normal_norm < 1e-9:
            return None

        return normal_vector / normal_norm

    def calculate_plane_metrics(self, points_mm):
        """
        Calculate plane normal, plane distance and projected point.

        Plane equation:

            n dot X + d = 0
        """
        if len(points_mm) != 3:
            return None, None, None, None, None

        points_mm = np.asarray(points_mm, dtype=np.float64)

        normal_vector = self.calculate_plane_normal(points_mm)

        if normal_vector is None:
            return None, None, None, None, None

        plane_point_mm = points_mm[0]
        plane_center_mm = np.mean(points_mm, axis=0)

        # Flip the normal so it points toward the robot center.
        direction_to_robot = (
            FRONT_PLATE_CENTER_MM - plane_point_mm
        )

        if np.dot(normal_vector, direction_to_robot) < 0:
            normal_vector = -normal_vector

        plane_offset_d = -float(
            np.dot(normal_vector, plane_point_mm)
        )

        signed_distance_mm = (
            np.dot(normal_vector, FRONT_PLATE_CENTER_MM)
            + plane_offset_d
        )

        distance_mm = abs(signed_distance_mm)

        projection_point_mm = (
            FRONT_PLATE_CENTER_MM
            - signed_distance_mm * normal_vector
        )

        return (
            normal_vector,
            plane_offset_d,
            distance_mm,
            projection_point_mm,
            plane_center_mm,
        )

    def remove_dynamic_geometry(self):
        """Remove geometries that must be redrawn every update."""
        geometry_names = [
            "obstacle_points",
            "obstacle_plane",
            "plane_normal_arrow",
            "center_to_plane_line",
        ]

        for geometry_name in geometry_names:
            if self.scene_widget.scene.has_geometry(geometry_name):
                self.scene_widget.scene.remove_geometry(geometry_name)

        if self.normal_label_3d is not None:
            try:
                self.scene_widget.remove_3d_label(
                    self.normal_label_3d
                )
            except Exception:
                pass

            self.normal_label_3d = None

    def update_info_panel(self, normal_vector, distance_mm):
        """Update the concise English information panel."""
        distances_mm = self.shared_data.get_calibrated_distances()
        statuses = self.shared_data.get_status()

        if normal_vector is None:
            normal_text = "--"
            plane_distance_text = "--"

            status_text = (
                f"S1={statuses[1]}, "
                f"S2={statuses[2]}, "
                f"S3={statuses[3]}\n"
                "Need 3 valid measurements"
            )
        else:
            normal_text = (
                f"({normal_vector[0]:.4f},\n"
                f" {normal_vector[1]:.4f},\n"
                f" {normal_vector[2]:.4f})"
            )

            plane_distance_text = f"{distance_mm:.1f} mm"
            status_text = "Plane valid"

        self.info_label.text = (
            "SENSOR / PLANE INFO\n\n"
            f"S1 Distance: {format_distance(distances_mm[1])}\n"
            f"S2 Distance: {format_distance(distances_mm[2])}\n"
            f"S3 Distance: {format_distance(distances_mm[3])}\n\n"
            "Plane Normal:\n"
            f"{normal_text}\n\n"
            "Center to Plane:\n"
            f"{plane_distance_text}\n\n"
            f"Status: {status_text}"
        )

    def update_dynamic_scene(self):
        """Update obstacle points, plane, normal arrow and labels."""
        distances_mm = self.shared_data.get_calibrated_distances()
        statuses = self.shared_data.get_status()
        signature = tuple(
            (sensor_id, distances_mm[sensor_id], statuses[sensor_id])
            for sensor_id in (1, 2, 3)
        )

        # Geometry only needs to be rebuilt when sensor data has changed.
        if signature == self.last_signature:
            return

        points_mm, colors = self.calculate_obstacle_points(distances_mm)

        self.remove_dynamic_geometry()

        # ----------------------------------------------------
        # Obstacle point cloud
        # ----------------------------------------------------

        obstacle_cloud = o3d.geometry.PointCloud()

        if len(points_mm) > 0:
            obstacle_cloud.points = o3d.utility.Vector3dVector(
                np.asarray(points_mm, dtype=np.float64)
            )

            obstacle_cloud.colors = o3d.utility.Vector3dVector(
                np.asarray(colors, dtype=np.float64)
            )

        self.scene_widget.scene.add_geometry(
            "obstacle_points",
            obstacle_cloud,
            self.point_material,
        )

        normal_vector = None
        distance_mm = None

        # ----------------------------------------------------
        # Plane geometry
        # ----------------------------------------------------

        if len(points_mm) == 3:
            (
                normal_vector,
                _plane_offset_d,
                distance_mm,
                projection_point_mm,
                plane_center_mm,
            ) = self.calculate_plane_metrics(points_mm)

            if normal_vector is not None:
                self.latest_plane_data = {
                    "distance": float(distance_mm),
                    "nx": float(normal_vector[0]),
                    "ny": float(normal_vector[1]),
                    "nz": float(normal_vector[2]),
                }

                plane_mesh = create_plane_mesh(points_mm)

                self.scene_widget.scene.add_geometry(
                    "obstacle_plane",
                    plane_mesh,
                    self.mesh_material,
                )

                # Normal vector arrow.
                # Thin plane normal arrow.
                normal_arrow = create_arrow(
                    position_mm=plane_center_mm,
                    direction=normal_vector,
                    color=NORMAL_COLOR,
                    cylinder_radius_mm=0.85,
                    cone_radius_mm=2.00,
                    cylinder_height_mm=45.0,
                    cone_height_mm=14.0,
                )

                self.scene_widget.scene.add_geometry(
                    "plane_normal_arrow",
                    normal_arrow,
                    self.mesh_material,
                )

                # Line from robot center to the plane.
                distance_line = create_line_set(
                    FRONT_PLATE_CENTER_MM,
                    projection_point_mm,
                    DISTANCE_LINE_COLOR,
                )

                self.scene_widget.scene.add_geometry(
                    "center_to_plane_line",
                    distance_line,
                    self.line_material,
                )

                # ------------------------------------------------
                # White 3D label in the middle of the screen
                # ------------------------------------------------

                label_position_mm = (
                    plane_center_mm + normal_vector * 80.0
                )

                self.normal_label_3d = (
                    self.scene_widget.add_3d_label(
                        label_position_mm.astype(np.float32),
                        "Plane Normal\n"
                        f"n = ({normal_vector[0]:.3f}, "
                        f"{normal_vector[1]:.3f}, "
                        f"{normal_vector[2]:.3f})\n"
                        f"Distance = {distance_mm:.1f} mm",
                    )
                )

                # Make the center tag white.
                self.normal_label_3d.color = gui.Color(
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                )

                self.normal_label_3d.scale = 1.15
            else:
                self.latest_plane_data = None
        else:
            self.latest_plane_data = None

        self.update_info_panel(
            normal_vector,
            distance_mm,
        )
        self.last_signature = signature

    def on_key_event(self, event):
        """Queue the current plane data when the user presses S."""
        if event.type != gui.KeyEvent.Type.DOWN or event.key != gui.KeyName.S:
            return False

        if self.latest_plane_data is None:
            self.log_queue.put("Cannot record: a valid plane is not available yet.")
            return True

        self.record_queue.put(dict(self.latest_plane_data))
        self.log_queue.put("S pressed: enter a sequence number in the console.")
        return True

    def record_to_excel_worker(self):
        """Ask for a sequence number and append each queued snapshot to Excel."""
        workbook_path = Path(__file__).resolve().with_name("positioning_data.xlsx")

        while True:
            plane_data = self.record_queue.get()
            try:
                while True:
                    sequence_no = input("请输入本条数据的序号: ").strip()
                    if sequence_no:
                        break
                    print("序号不能为空，请重新输入。")

                from openpyxl import Workbook, load_workbook

                if workbook_path.exists():
                    workbook = load_workbook(workbook_path)
                    worksheet = workbook.active
                else:
                    workbook = Workbook()
                    worksheet = workbook.active
                    worksheet.title = "Positioning Data"
                    worksheet.append(["Sequence No.", "distance (mm)", "nx", "ny", "nz"])

                worksheet.append([
                    sequence_no,
                    plane_data["distance"],
                    plane_data["nx"],
                    plane_data["ny"],
                    plane_data["nz"],
                ])
                temporary_path = workbook_path.with_name(
                    f"{workbook_path.stem}.tmp{workbook_path.suffix}"
                )
                workbook.save(temporary_path)
                os.replace(temporary_path, workbook_path)
                self.log_queue.put(
                    f"Saved sequence {sequence_no} to {workbook_path.name}."
                )
            except ImportError:
                self.log_queue.put(
                    "Excel recording requires openpyxl; install it with: pip install openpyxl"
                )
            except Exception as error:
                self.log_queue.put(f"Excel recording failed: {error}")
            finally:
                self.record_queue.task_done()

    def print_pending_logs(self):
        """Print serial messages in the console."""
        while True:
            try:
                message = self.log_queue.get_nowait()
                print(f"[{LOG_TAG}] {message}")
            except queue.Empty:
                break

    def on_tick(self):
        """GUI timer callback."""
        try:
            self.update_dynamic_scene()
            self.last_tick_error = None
        except Exception as error:
            error_message = f"{type(error).__name__}: {error}"
            if error_message != self.last_tick_error:
                self.log_queue.put(
                    f"Viewer update failed: {error_message}; see console traceback."
                )
                traceback.print_exc()
                self.last_tick_error = error_message
        finally:
            self.print_pending_logs()

        return True

    def run(self):
        print("Open3D viewer started.")
        print("All coordinates and distances use mm.")
        print("Mouse: left drag = rotate, wheel = zoom.")

        self.app.run()


# ============================================================
# Serial port helpers
# ============================================================

def print_available_ports():
    """Print all currently detected serial ports."""
    ports = list(serial.tools.list_ports.comports())

    if not ports:
        print("No serial ports found.")
        return

    print("Available serial ports:")

    for port in ports:
        print(f"  {port.device} - {port.description}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="VL53L4CD 3-sensor Open3D real-time viewer"
    )

    parser.add_argument(
        "--port",
        default="COM4",
        help="Serial port, for example COM4 or /dev/ttyUSB0",
    )

    parser.add_argument(
        "--baud",
        type=int,
        default=115200,
        help="Serial baud rate, default: 115200",
    )

    parser.add_argument(
        "--list-ports",
        action="store_true",
        help="List serial ports and exit",
    )

    args = parser.parse_args()

    if args.list_ports:
        print_available_ports()
        return

    shared_data = SharedSensorData()
    log_queue = queue.Queue()

    serial_reader = SerialReader(
        port=args.port,
        baudrate=args.baud,
        shared_data=shared_data,
        log_queue=log_queue,
    )

    try:
        # Initialize Open3D GUI before creating the window.
        gui.Application.instance.initialize()

        serial_reader.start()

        visualizer = VL53L3DVisualizer(
            shared_data=shared_data,
            log_queue=log_queue,
        )

        visualizer.run()

    except KeyboardInterrupt:
        print("Stopped by user.")

    except Exception as error:
        print(f"Program error: {error}")
        traceback.print_exc()

    finally:
        serial_reader.stop()
        serial_reader.join(timeout=1.0)

        print("Program exited.")


if __name__ == "__main__":
    main()
