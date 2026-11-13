"""Coordinate math, serial state and CSV recording, independent of the GUI.

All geometry uses metres. T_A_B maps points expressed in B into A.
W = AprilTag/wing world, R = robot front plate, C = OpenCV camera.
R and C both use X right, Y down, Z forward.
"""
from __future__ import annotations

import csv
import json
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

SENSOR_IDS = (1, 2, 3)
# Express the legacy infrared geometry in R (X right, Y down, Z forward):
# negate X and Y for both positions and beam directions. The confirmed mounting
# is S1 above, S2 left (negative X), S3 right (positive X), looking forward.
# Sensor IDs, measured offsets and their calibration coefficients stay paired.
SENSOR_POSITIONS = np.array([[0, -14.8, 10.8], [-6.7, 5.5, 12.3], [8.7, 5.5, 12.3]]) / 1000
SENSOR_DIRECTIONS = np.array([[0, -.259, .966], [-.5, 0, .866], [.5, 0, .866]])
SENSOR_DIRECTIONS /= np.linalg.norm(SENSOR_DIRECTIONS, axis=1)[:, None]
SERIAL_PATTERN = re.compile(
    r"S(?P<id>[1-3]):\s*(?:(?P<valid>VALID)\s*\|\s*distance=(?P<distance>\d+)\s*mm|INVALID)",
    re.IGNORECASE,
)


def rigid_transform(value, name="transform"):
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{name}: must be a finite 4x4 matrix")
    rotation = matrix[:3, :3]
    if not (np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-7)
            and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            and np.isclose(np.linalg.det(rotation), 1, atol=1e-6)):
        raise ValueError(f"{name}: must be a rigid transform (rotation determinant +1)")
    return matrix


def inverse_transform(matrix):
    result = np.eye(4)
    result[:3, :3] = matrix[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ matrix[:3, 3]
    return result


def transform_points(matrix, points):
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


def load_tag_layout(path):
    """Load T_W_Tag matrices shared by visual localization and the 3D scene."""
    layout = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(layout, dict):
        raise ValueError("tag_layout must map Tag IDs to 4x4 transforms")
    return {tag_id: rigid_transform(matrix, f"tag_layout[{tag_id}]")
            for tag_id, matrix in layout.items()}


def robot_pose(camera_pose, robot_from_camera):
    """T_W_R = T_W_C @ inverse(T_R_C). Unknown extrinsics yield no pose."""
    if camera_pose is None or robot_from_camera is None:
        return None
    return camera_pose @ inverse_transform(robot_from_camera)


def euler_degrees(rotation):
    pitch = np.arcsin(np.clip(-rotation[2, 0], -1, 1))
    if abs(np.cos(pitch)) > 1e-8:
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        roll, yaw = np.arctan2(-rotation[1, 2], rotation[1, 1]), 0.
    return np.rad2deg([roll, pitch, yaw])


def load_infrared_calibration(path):
    """Read calibrated_mm = slope * raw_mm + intercept_mm for S1/S2/S3."""
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("infrared_calibration must contain S1, S2 and S3 parameters")
    calibration = {}
    for sensor_id in SENSOR_IDS:
        name = f"S{sensor_id}"
        parameters = data.get(name)
        if not isinstance(parameters, dict):
            raise ValueError(f"infrared_calibration[{name}]: missing sensor parameters")
        values = [parameters.get(key) for key in ("slope", "intercept_mm")]
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
            raise ValueError(f"infrared_calibration[{name}]: slope and intercept_mm must be numbers")
        slope, intercept = values
        if not np.isfinite(values).all() or slope <= 0:
            raise ValueError(f"infrared_calibration[{name}]: slope must be positive and both values finite")
        calibration[sensor_id] = (float(slope), float(intercept))
    return calibration


def load_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    for key in ("T_R_C", "T_W_wing_mesh", "T_R_robot_mesh"):
        if config[key] is not None:
            config[key] = rigid_transform(config[key], key)
        elif key != "T_R_C":
            raise ValueError(f"{key} cannot be null")
    for key in ("wing_mesh_scale", "robot_mesh_scale", "sensor_timeout_s", "vision_timeout_s",
                "sensor_max_skew_s", "reprojection_error_px", "tag_size_m"):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if not 0 < config["wing_opacity"] <= 1:
        raise ValueError("wing_opacity must be in (0, 1]")
    config.setdefault("snapshot_dir", "camera_snapshots")
    config.setdefault("infrared_calibration_file", "infrared_calibration.json")
    for key, default, minimum in (("pnp_center_weight", 4., 1.), ("pnp_image_center_boost", 3., 0.)):
        config.setdefault(key, default)
        if (isinstance(config[key], bool) or not isinstance(config[key], (int, float))
                or not np.isfinite(config[key]) or config[key] < minimum):
            raise ValueError(f"{key} must be finite and >= {minimum}")
    for key in ("wing_mesh", "robot_mesh", "calibration_file", "tag_layout_file", "csv_file", "snapshot_dir",
                "infrared_calibration_file"):
        config[key] = (path.parent / config[key]).resolve()
    if config.get("focus_map_file"):
        config["focus_map_file"] = (path.parent / config["focus_map_file"]).resolve()
    usb_id = config.get("serial_usb_id")
    if usb_id is not None:
        if not isinstance(usb_id, str) or not re.fullmatch(r"[0-9A-Fa-f]{4}:[0-9A-Fa-f]{4}", usb_id):
            raise ValueError("serial_usb_id 须为十六进制 VID:PID，例如 1A86:7523")
        config["serial_usb_id"] = tuple(int(part, 16) for part in usb_id.split(":"))
    config["infrared_calibration"] = load_infrared_calibration(config["infrared_calibration_file"])
    return config


@dataclass
class SensorReading:
    raw_mm: float | None = None
    calibrated_mm: float | None = None
    status: str = "WAITING"
    monotonic: float | None = None
    timestamp: str = ""


class SensorStore:
    def __init__(self, calibration):
        self.calibration = dict(calibration)
        self.lock = threading.Lock()
        self.readings = {i: SensorReading() for i in SENSOR_IDS}

    def consume(self, line, now=None):
        """Return whether a line contains any recognized firmware reports."""
        now = time.monotonic() if now is None else now
        received = False
        with self.lock:
            for match in SERIAL_PATTERN.finditer(line):
                received = True
                sensor_id = int(match["id"])
                raw = float(match["distance"]) if match["valid"] else None
                slope, intercept = self.calibration[sensor_id]
                calibrated = None if raw is None else slope * raw + intercept
                status = "VALID" if calibrated is not None and calibrated > 0 else "INVALID"
                self.readings[sensor_id] = SensorReading(
                    raw, calibrated, status, now, datetime.now().astimezone().isoformat(timespec="milliseconds")
                )
        return received

    def snapshot(self, timeout, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            result = {i: SensorReading(**vars(r)) for i, r in self.readings.items()}
        for reading in result.values():
            if reading.status == "VALID" and reading.monotonic is not None and now - reading.monotonic > timeout:
                reading.status = "STALE"
        return result

    def disconnect(self):
        with self.lock:
            for reading in self.readings.values():
                reading.status = "DISCONNECTED"


@dataclass
class Plane:
    points: np.ndarray
    normal: np.ndarray
    offset: float
    distance: float
    projection: np.ndarray
    center: np.ndarray


def measured_points(readings):
    return {i: SENSOR_POSITIONS[i - 1] + r.calibrated_mm / 1000 * SENSOR_DIRECTIONS[i - 1]
            for i, r in readings.items() if r.status == "VALID" and r.calibrated_mm is not None}


def fit_plane(readings, max_skew):
    points = measured_points(readings)
    if len(points) != 3:
        return None
    times = [readings[i].monotonic for i in SENSOR_IDS]
    if any(t is None for t in times) or max(times) - min(times) > max_skew:
        return None
    points = np.array([points[i] for i in SENSOR_IDS])
    normal = np.cross(points[1] - points[0], points[2] - points[0])
    length = np.linalg.norm(normal)
    if length < 1e-10:
        return None
    normal /= length
    if np.dot(normal, -points[0]) < 0:
        normal = -normal
    offset = -float(normal @ points[0])
    return Plane(points, normal, offset, abs(offset), -offset * normal, points.mean(axis=0))


@dataclass
class VisionResult:
    frame: np.ndarray | None = None
    camera_pose: np.ndarray | None = None
    detected_ids: list = field(default_factory=list)
    known_ids: list = field(default_factory=list)
    corners: list = field(default_factory=list)
    inlier_indices: list = field(default_factory=list)
    reprojection_rmse_px: float | None = None
    weighted_reprojection_rmse_px: float | None = None
    center_inlier_ids: list = field(default_factory=list)
    tag_weights: list = field(default_factory=list)
    focus_position: float | None = None
    focus_readback: float | None = None
    focus_mode: str = "UNKNOWN"
    centers: list = field(default_factory=list)
    camera_matrix: list | None = None
    dist_coeffs: list | None = None
    timestamp: str = ""
    monotonic: float = 0.
    status: str = "WAITING"


def load_camera_calibration(path, resolution, flip_camera_180):
    """Load original-image OpenCV K/D, with optional capture metadata checks.

    Legacy JSON files without capture metadata remain supported. This loader
    never rescales K or rotates image coordinates implicitly.
    """
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict):
            raise ValueError("Calibration JSON must be an object")
        if data.get("model", "opencv_pinhole") != "opencv_pinhole":
            raise ValueError("Calibration model must be opencv_pinhole")
        if "image_size" in data and data["image_size"] != list(resolution):
            raise ValueError(f"image_size {data['image_size']} does not match camera_resolution {list(resolution)}")
        if "flip_camera_180" in data and (
                type(data["flip_camera_180"]) is not bool
                or data["flip_camera_180"] != flip_camera_180):
            raise ValueError("Calibration flip_camera_180 does not match camera orientation")
        matrix = np.array(data["camera_matrix"], dtype=float)
        distortion = np.array(data["dist_coeffs"], dtype=float)
        if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
                or matrix[0, 0] <= 0 or matrix[1, 1] <= 0
                or not np.allclose(matrix[2], [0., 0., 1.], rtol=0, atol=1e-9)
                or not np.allclose([matrix[0, 1], matrix[1, 0]], 0., rtol=0, atol=1e-9)):
            raise ValueError("camera_matrix must be [[fx,0,cx],[0,fy,cy],[0,0,1]] with positive focal lengths")
        if (not np.isfinite(distortion).all()
                or not (distortion.ndim == 1 or (distortion.ndim == 2 and 1 in distortion.shape))
                or distortion.size not in (4, 5, 8, 12, 14)):
            raise ValueError("dist_coeffs must be a finite OpenCV vector with 4/5/8/12/14 elements")
        if "rms" in data and (isinstance(data["rms"], bool)
                or not isinstance(data["rms"], (int, float))
                or not np.isfinite(data["rms"]) or data["rms"] < 0):
            raise ValueError("rms must be a finite nonnegative number")
        return matrix, distortion.reshape(-1), data
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid camera calibration JSON ({path}): {error}") from error


def tag_image_weight(center, image_size, boost=3.):
    """Smooth radial preference about the image midpoint, not the optical principal point."""
    midpoint = (np.asarray(image_size, dtype=float) - 1.) / 2
    radius = np.clip(np.linalg.norm(np.asarray(center) - midpoint) / max(np.linalg.norm(midpoint), 1.), 0., 1.)
    return float(1. + boost * (1. - radius) ** 2)


def refine_weighted_pnp(world, image, weights, matrix, distortion, rvec, tvec):
    """Minimize sum_i w_i ||project(X_i)-u_i||^2 with damped Gauss-Newton.

    projectPoints supplies the exact first six Jacobian columns for rvec/tvec.
    No point duplication or extra optimization dependency is needed.
    """
    import cv2
    world = np.asarray(world, dtype=float).reshape(-1, 3)
    image = np.asarray(image, dtype=float).reshape(-1, 2)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if (len(world) < 4 or len(image) != len(world) or len(weights) != len(world)
            or not np.isfinite(weights).all() or np.any(weights <= 0)):
        raise ValueError("Weighted PnP requires >=4 correspondences and positive finite weights")
    scale = np.repeat(np.sqrt(weights / weights.max()), 2)
    parameters = np.r_[np.asarray(rvec).reshape(3), np.asarray(tvec).reshape(3)].astype(float)
    damping = 1e-3

    def evaluate(values):
        projected, jacobian = cv2.projectPoints(world, values[:3], values[3:], matrix, distortion)
        residual = (projected.reshape(-1, 2) - image).ravel() * scale
        return residual, jacobian[:, :6] * scale[:, None]

    residual, jacobian = evaluate(parameters)
    for _ in range(30):
        hessian = jacobian.T @ jacobian
        gradient = jacobian.T @ residual
        if np.linalg.norm(gradient, ord=np.inf) < 1e-9:
            break
        try:
            step = np.linalg.solve(hessian + damping * np.diag(np.maximum(np.diag(hessian), 1e-12)), -gradient)
        except np.linalg.LinAlgError:
            break
        candidate = parameters + step
        rotation = cv2.Rodrigues(candidate[:3])[0]
        if (not np.isfinite(candidate).all()
                or np.any((world @ rotation.T + candidate[3:])[:, 2] <= 0)):
            damping *= 10
            continue
        new_residual, new_jacobian = evaluate(candidate)
        old_cost, new_cost = residual @ residual, new_residual @ new_residual
        if np.isfinite(new_cost) and new_cost < old_cost:
            parameters, residual, jacobian = candidate, new_residual, new_jacobian
            damping = max(damping / 3, 1e-12)
            if old_cost - new_cost < 1e-12 * max(old_cost, 1.) or np.linalg.norm(step) < 1e-10:
                break
        else:
            damping = min(damping * 10, 1e12)
    return parameters[:3].reshape(3, 1), parameters[3:].reshape(3, 1)


class VisionEstimator:
    def __init__(self, config):
        import cv2
        from pupil_apriltags import Detector
        self.cv2, self.config = cv2, config
        # Each entry is T_W_Tag: tag-local coordinates -> world coordinates (m).
        self.layout = load_tag_layout(config["tag_layout_file"])
        self.detector = Detector(families=config["tag_family"], nthreads=4, quad_decimate=1., refine_edges=1)
        width, height = config["camera_resolution"]
        self.calibrated = config["calibration_file"].exists()
        self.calibration_data = {}
        self.focus_map = None
        if config.get("focus_map_file"):
            from localization.focus_map import FocusMap
            self.focus_map = FocusMap(config["focus_map_file"], (width, height), config["flip_camera_180"])
            self.K, self.D = self.focus_map.lookup(self.focus_map.focus[0], require_validation=False)
            self.calibration_data = self.focus_map.data
            self.calibrated = True
        elif self.calibrated:
            self.K, self.D, self.calibration_data = load_camera_calibration(
                config["calibration_file"], (width, height), config["flip_camera_180"])
        else:
            self.K = np.array([[600., 0, width / 2], [0, 600., height / 2], [0, 0, 1.]])
            self.D = np.zeros(5)
        s = config["tag_size_m"] / 2
        # pupil-apriltags corner order, retained from the original program.
        self.local = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=float)
        self.center_weight = config.get("pnp_center_weight", 4.)
        self.image_center_boost = config.get("pnp_image_center_boost", 3.)

    def _draw_corner_errors(self, frame, corners, errors, inlier_indices):
        cv2 = self.cv2
        font, scale = cv2.FONT_HERSHEY_SIMPLEX, .38
        height, width = frame.shape[:2]
        inliers = set(inlier_indices)
        for start in range(0, len(corners), 4):
            center = corners[start:start + 4].mean(axis=0)
            for index in range(start, start + 4):
                error = errors[index]
                if not np.isfinite(error):
                    continue
                label = f"{error:.2f}px"
                (text_width, text_height), baseline = cv2.getTextSize(label, font, scale, 1)
                x, y = np.rint(corners[index]).astype(int)
                left = x - text_width - 5 if corners[index, 0] < center[0] else x + 5
                bottom = y - 5 if corners[index, 1] < center[1] else y + text_height + 5
                origin = (max(2, min(left, width - text_width - 2)),
                          max(text_height + 2, min(bottom, height - baseline - 2)))
                color = (0, 255, 0) if index in inliers else (0, 140, 255)
                cv2.circle(frame, (x, y), 2, color, -1, cv2.LINE_AA)
                cv2.putText(frame, label, origin, font, scale, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(frame, label, origin, font, scale, color, 1, cv2.LINE_AA)

    def process(self, frame, focus_position=None):
        cv2 = self.cv2
        if self.config["flip_camera_180"]:
            frame = cv2.flip(frame, -1)
        result = VisionResult(frame=frame, timestamp=datetime.now().astimezone().isoformat(timespec="milliseconds"),
                              monotonic=time.monotonic(), status="No known tags")
        if self.focus_map is not None:
            if focus_position is not None and np.isfinite(focus_position):
                result.focus_position = float(focus_position)
            try:
                self.K, self.D = self.focus_map.lookup(focus_position, require_validation=False)
                result.dist_coeffs = self.D.tolist()
            except ValueError as error:
                result.status = f"FOCUS: {error}"
                return result
        result.camera_matrix = self.K.tolist()
        tags = self.detector.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if result.focus_position is not None:
            cv2.putText(frame, f"focus={focus_position:g}", (8, frame.shape[0]-12),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 220, 255), 1)
        world, image, centers_world, centers_image = [], [], [], []
        for tag in tags:
            result.detected_ids.append(int(tag.tag_id))
            result.corners.append(np.asarray(tag.corners).tolist())
            result.centers.append(np.asarray(tag.center).tolist())
            corners = np.asarray(tag.corners, dtype=np.int32)
            cv2.polylines(frame, [corners], True, (0, 255, 0), 2)
            center = tuple(np.asarray(tag.center, dtype=int))
            cv2.circle(frame, center, 4, (0, 0, 255), -1)
            cv2.putText(frame, f"ID:{tag.tag_id}", (center[0] - 20, center[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1)
            pose = self.layout.get(str(tag.tag_id))
            if pose is None:
                continue
            result.known_ids.append(int(tag.tag_id))
            world.append(transform_points(pose, self.local))
            image.append(np.asarray(tag.corners, dtype=float))
            centers_world.append(pose[:3, 3].copy())
            centers_image.append(np.asarray(tag.center, dtype=float).reshape(2))
            # Use the tag-local plane and retain the detector's corner order.
            ok, rvec, tvec = cv2.solvePnP(self.local, image[-1], self.K, self.D, flags=cv2.SOLVEPNP_ITERATIVE)
            if ok:
                cv2.drawFrameAxes(frame, self.K, self.D, rvec, tvec, self.config["tag_size_m"] * .6)
        if world:
            world, image = np.vstack(world), np.vstack(image)
            result.tag_weights = [tag_image_weight(c, frame.shape[1::-1], self.image_center_boost)
                                  if len(result.known_ids) > 1 and np.isfinite(c).all() else 1.
                                  for c in centers_image]
            if len(result.known_ids) == 1:
                # Solve in the tag's z=0 plane, then apply its world transform.
                ok, rvec, tvec = cv2.solvePnP(self.local, image, self.K, self.D, flags=cv2.SOLVEPNP_ITERATIVE)
                if ok:
                    camera_from_tag = np.eye(4)
                    camera_from_tag[:3, :3] = cv2.Rodrigues(rvec)[0]
                    camera_from_tag[:3, 3] = tvec.reshape(3)
                    tag_pose = self.layout[str(result.known_ids[0])]
                    camera_from_world = camera_from_tag @ inverse_transform(tag_pose)
                    rvec = cv2.Rodrigues(camera_from_world[:3, :3])[0]
                    tvec = camera_from_world[:3, 3].reshape(3, 1).copy()
                indices = np.arange(4) if ok else np.array([], dtype=int)
            else:
                ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                    world, image, self.K, self.D, flags=cv2.SOLVEPNP_SQPNP,
                    reprojectionError=self.config["reprojection_error_px"], iterationsCount=100, confidence=.99)
                indices = inliers.ravel() if inliers is not None else np.array([], dtype=int)
            result.status = "PnP failed"
            if ok and len(indices) >= 4:
                # Keep corner indices in their original 4-per-tag order for CSV compatibility.
                fit_world, fit_image = list(world[indices]), list(image[indices])
                fit_weights = [result.tag_weights[i // 4] for i in indices]
                center_ids = []
                projected_centers, _ = cv2.projectPoints(np.asarray(centers_world), rvec, tvec, self.K, self.D)
                for i, (point, center, predicted) in enumerate(zip(
                        centers_world, centers_image, projected_centers.reshape(-1, 2))):
                    # A rejected tag must not re-enter the fit through its high-weight center.
                    if (np.sum((indices // 4) == i) >= 3 and np.isfinite(center).all()
                            and np.linalg.norm(center - predicted) <= self.config["reprojection_error_px"]):
                        fit_world.append(point)
                        fit_image.append(center)
                        fit_weights.append(self.center_weight * result.tag_weights[i])
                        center_ids.append(result.known_ids[i])
                fit_world, fit_image, fit_weights = map(np.asarray, (fit_world, fit_image, fit_weights))
                rvec, tvec = refine_weighted_pnp(fit_world, fit_image, fit_weights, self.K, self.D, rvec, tvec)
                rotation, _ = cv2.Rodrigues(rvec)
                projected, _ = cv2.projectPoints(fit_world, rvec, tvec, self.K, self.D)
                squared_errors = np.sum((projected.reshape(-1, 2) - fit_image) ** 2, axis=1)
                rmse = float(np.sqrt(np.mean(squared_errors)))
                weighted_rmse = float(np.sqrt(np.average(squared_errors, weights=fit_weights)))
                positive_depth = np.all((fit_world @ rotation.T + tvec.reshape(3))[:, 2] > 0)
                if positive_depth and rmse <= self.config["reprojection_error_px"]:
                    camera_from_world = np.eye(4)
                    camera_from_world[:3, :3], camera_from_world[:3, 3] = rotation, tvec.reshape(3)
                    result.camera_pose = inverse_transform(camera_from_world)
                    result.inlier_indices = indices.tolist()
                    result.reprojection_rmse_px = rmse
                    result.weighted_reprojection_rmse_px = weighted_rmse
                    result.center_inlier_ids = center_ids
                    result.status = "VALID" if self.calibrated else "VALID (default intrinsics, uncalibrated)"
                    cv2.drawFrameAxes(frame, self.K, self.D, rvec, tvec, .05)
                else:
                    result.status = "PnP rejected (depth/reprojection)"
            for index, tag_id in enumerate(result.known_ids):
                count = sum(index * 4 <= i < index * 4 + 4 for i in result.inlier_indices)
                center = image[index * 4:index * 4 + 4].mean(axis=0).astype(int)
                text = f"{count}/4 C:{int(tag_id in result.center_inlier_ids)} W:{result.tag_weights[index]:.1f}"
                cv2.putText(frame, text, (center[0] - 30, center[1] + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 0) if count == 4 else (0, 100, 255), 1)
            if result.camera_pose is not None:
                # Use the joint pose and include corners that RANSAC excluded.
                projected_corners, _ = cv2.projectPoints(world, rvec, tvec, self.K, self.D)
                errors = np.linalg.norm(projected_corners.reshape(-1, 2) - image, axis=1)
                errors[(world @ rotation[2] + tvec[2, 0]) <= 0] = np.nan
                self._draw_corner_errors(frame, image, errors, result.inlier_indices)
        return result


def matrix_fields(prefix):
    return [f"{prefix}_{row}{col}" for row in range(4) for col in range(4)]


LEGACY_CSV_FIELDS = (["group_id", "saved_at", "snapshot_at", "demo", "intrinsics_calibrated",
               "extrinsics_calibrated", "vision_status", "vision_timestamp", "vision_age_s",
               "detected_tag_ids", "known_tag_ids", "tag_corners_px_json", "inlier_indices_json", "reprojection_rmse_px"]
              + matrix_fields("T_W_C") + matrix_fields("T_W_R") + matrix_fields("T_R_C")
              + [f"{body}_{axis}" for body in ("camera", "robot")
                 for axis in ("x_m", "y_m", "z_m", "roll_deg", "pitch_deg", "yaw_deg")]
              + [f"s{i}_{item}" for i in SENSOR_IDS for item in
                 ("raw_mm", "calibrated_mm", "status", "timestamp", "age_s", "point_x_m", "point_y_m", "point_z_m")]
              + ["plane_valid", "plane_distance_m", "plane_distance_mm", "plane_nx_R", "plane_ny_R", "plane_nz_R", "plane_d_R_m"]
              + [f"plane_{item}_{axis}_m" for item in ("projection_R", "center_R") for axis in "xyz"])
CSV_FIELDS = LEGACY_CSV_FIELDS + ["camera_fx_px", "camera_fy_px", "focus_position"]
CSV_WRITE_LOCK = threading.Lock()


def record_snapshot(vision, readings, plane, config, demo=False, now=None):
    """Flatten one displayed sample; absent/expired values stay empty, never zero."""
    now = time.monotonic() if now is None else now
    row = dict.fromkeys(CSV_FIELDS, "")
    row.update(snapshot_at=datetime.now().astimezone().isoformat(timespec="milliseconds"), demo=demo,
               intrinsics_calibrated=bool(config["intrinsics_calibrated"]),
               extrinsics_calibrated=config["T_R_C"] is not None and not demo, vision_status=vision.status,
               vision_timestamp=vision.timestamp, vision_age_s=max(0, now - vision.monotonic) if vision.timestamp else "",
               detected_tag_ids=json.dumps(vision.detected_ids), known_tag_ids=json.dumps(vision.known_ids),
               tag_corners_px_json=json.dumps(vision.corners), inlier_indices_json=json.dumps(vision.inlier_indices),
               reprojection_rmse_px=vision.reprojection_rmse_px if vision.reprojection_rmse_px is not None else "",
               plane_valid=plane is not None)
    if vision.camera_matrix is not None:
        row.update(camera_fx_px=vision.camera_matrix[0][0], camera_fy_px=vision.camera_matrix[1][1])
    if vision.focus_position is not None:
        row["focus_position"] = vision.focus_position
    if not demo and config.get("focus_map_file"):
        row["_camera_context"] = {"focus_map_file": str(config["focus_map_file"]),
            "focus_position": vision.focus_position, "camera_matrix": vision.camera_matrix,
            "dist_coeffs": vision.dist_coeffs, "tag_centers_px": vision.centers,
            "detected_tag_ids": vision.detected_ids, "center_inlier_ids": vision.center_inlier_ids,
            "tag_weights": vision.tag_weights, "vision_status": vision.status,
            "note": "Focus is a stable driver readback; not synchronized per-exposure encoder metadata."}
    camera = vision.camera_pose if now - vision.monotonic <= config["vision_timeout_s"] else None
    robot = robot_pose(camera, config["T_R_C"])
    for prefix, matrix in (("T_W_C", camera), ("T_W_R", robot), ("T_R_C", config["T_R_C"])):
        if matrix is not None:
            row.update(zip(matrix_fields(prefix), matrix.ravel().tolist()))
    for name, matrix in (("camera", camera), ("robot", robot)):
        if matrix is not None:
            values = [*matrix[:3, 3], *euler_degrees(matrix[:3, :3])]
            row.update(zip([f"{name}_{a}" for a in ("x_m", "y_m", "z_m", "roll_deg", "pitch_deg", "yaw_deg")], values))
    points = measured_points(readings)
    for i, reading in readings.items():
        row.update({f"s{i}_raw_mm": reading.raw_mm if reading.raw_mm is not None else "",
                    f"s{i}_calibrated_mm": reading.calibrated_mm if reading.calibrated_mm is not None else "",
                    f"s{i}_status": reading.status, f"s{i}_timestamp": reading.timestamp,
                    f"s{i}_age_s": max(0, now - reading.monotonic) if reading.monotonic is not None else ""})
        if i in points:
            row.update(zip([f"s{i}_point_{a}_m" for a in "xyz"], points[i].tolist()))
    if plane is not None:
        row.update(plane_distance_m=plane.distance, plane_distance_mm=plane.distance * 1000,
                   plane_nx_R=plane.normal[0], plane_ny_R=plane.normal[1], plane_nz_R=plane.normal[2], plane_d_R_m=plane.offset)
        for key, vector in (("projection_R", plane.projection), ("center_R", plane.center)):
            row.update(zip([f"plane_{key}_{a}_m" for a in "xyz"], vector.tolist()))
    return row


def append_csv(path, group_id, row):
    path = Path(path)
    if not group_id.strip():
        raise ValueError("数据组编号不能为空")
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {**row, "group_id": group_id.strip(),
              "saved_at": datetime.now().astimezone().isoformat(timespec="milliseconds")}
    with CSV_WRITE_LOCK:
        with path.open("a+", encoding="utf-8-sig", newline="") as stream:
            stream.seek(0)
            reader = csv.reader(stream)
            header = next(reader, None)
            if header == LEGACY_CSV_FIELDS:
                previous_rows = list(reader)
            elif header is None or header == CSV_FIELDS:
                stream.seek(0, 2)
                writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
                if header is None:
                    writer.writeheader()
                writer.writerow(record)
                return path
            else:
                raise ValueError("CSV 表头与当前版本不一致，请选择新的 CSV 文件")
        _upgrade_csv_and_append(path, previous_rows, record)
    return path


def _upgrade_csv_and_append(path, previous_rows, record):
    """Add focal-length columns and the new row in one atomic replacement."""
    if any(len(row) > len(LEGACY_CSV_FIELDS) for row in previous_rows):
        raise ValueError("旧版 CSV 有数据行超出表头列数，无法追加")
    temporary = NamedTemporaryFile(mode="w", dir=path.parent, encoding="utf-8-sig", newline="", delete=False)
    temporary_path = Path(temporary.name)
    try:
        with temporary:
            writer = csv.writer(temporary)
            writer.writerow(CSV_FIELDS)
            for row in previous_rows:
                if row:
                    row = row + [""] * (len(CSV_FIELDS) - len(row))
                writer.writerow(row)
            csv.DictWriter(temporary, fieldnames=CSV_FIELDS).writerow(record)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def save_record(csv_path, snapshot_dir, group_id, row, frame, captured_at):
    """Save the frozen annotated BGR frame and its CSV row without overwriting images.

    If CSV writing fails, remove this attempt's image and metadata so a rejected
    record does not leave an apparently successful snapshot.
    """
    if not group_id.strip():
        raise ValueError("数据组编号不能为空")
    image_path = None
    if frame is not None:
        import cv2
        try:
            ok, encoded = cv2.imencode(".png", frame)
        except cv2.error as error:
            raise ValueError(f"相机快照编码失败：{error}") from error
        if not ok:
            raise ValueError("相机快照编码失败")
        # Windows filenames cannot contain these characters. Keep the original
        # group ID in CSV, replacing only filename characters with underscores.
        safe_group = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", group_id.strip()).strip(" .")[:80] or "group"
        stem = f"{safe_group}_{captured_at:%Y%m%d_%H%M%S_%f}"
        directory = Path(snapshot_dir)
        directory.mkdir(parents=True, exist_ok=True)
        number = 0
        while True:
            candidate = directory / f"{stem}{'_' + str(number) if number else ''}.png"
            try:
                stream = candidate.open("xb")
            except FileExistsError:
                number += 1
                continue
            image_path = candidate
            break
        try:
            # Python file I/O supports Chinese paths on Windows.
            with stream:
                stream.write(encoded.tobytes())
        except OSError:
            image_path.unlink(missing_ok=True)
            raise
    metadata_path, metadata_created = None, False
    try:
        context = row.get("_camera_context")
        if image_path is not None and context is not None:
            metadata_path = image_path.with_suffix('.json')
            with metadata_path.open('x', encoding='utf-8') as stream:
                metadata_created = True
                json.dump({**context, "group_id": group_id.strip(), "snapshot_at": row.get('snapshot_at')},
                          stream, ensure_ascii=False, indent=2, allow_nan=False)
        csv_row = {key: value for key, value in row.items() if key != "_camera_context"}
        path = append_csv(csv_path, group_id, csv_row)
    except Exception:
        if image_path is not None:
            image_path.unlink(missing_ok=True)
        if metadata_created:
            metadata_path.unlink(missing_ok=True)
        raise
    return path, image_path
