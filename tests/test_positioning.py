import csv
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from localization.positioning_core import (
    SENSOR_POSITIONS, SENSOR_DIRECTIONS, CSV_FIELDS, LEGACY_CSV_FIELDS,
    SensorReading, SensorStore, VisionResult, VisionEstimator, fit_plane, inverse_transform,
    load_config, load_infrared_calibration, record_snapshot, append_csv, save_record, rigid_transform, robot_pose,
    refine_weighted_pnp, tag_image_weight,
)

ROOT = Path(__file__).resolve().parents[1]


class PositioningTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "config/positioning_config.json")
        self.config.pop("focus_map_file", None)
        self.config["intrinsics_calibrated"] = False
        self.config["T_R_C"] = None  # Keep tests independent of the user's installed extrinsics.

    def plane_readings(self, now=10):
        store = SensorStore(self.config["infrared_calibration"])
        # A known z=0.2 m plane, quantized to the actual integer-mm protocol.
        for i in (1, 2, 3):
            distance_mm = (200 - SENSOR_POSITIONS[i - 1, 2] * 1000) / SENSOR_DIRECTIONS[i - 1, 2]
            scale, offset = store.calibration[i]
            store.consume(f"S{i}: VALID | distance={round((distance_mm - offset) / scale)} mm", now)
        return store

    def test_robot_extrinsics_include_rotation_and_inverse(self):
        extrinsic, world_robot = np.eye(4), np.eye(4)
        extrinsic[:3, :3] = cv2.Rodrigues(np.array([.3, -.2, .1]))[0]
        extrinsic[:3, 3] = [.01, -.02, .03]
        world_robot[:3, :3] = cv2.Rodrigues(np.array([-.2, .5, .4]))[0]
        world_robot[:3, 3] = [.35, .06, -.01]
        np.testing.assert_allclose(robot_pose(world_robot @ extrinsic, extrinsic), world_robot, atol=1e-12)
        self.assertIsNone(robot_pose(world_robot, None))
        with self.assertRaises(ValueError):
            rigid_transform(np.diag([2, 1, 1, 1]))

    def test_serial_calibration_invalid_and_timeout(self):
        store = SensorStore({1: (.95, -8.), 2: (1., 0.), 3: (1., 0.)})
        store.consume("debug S1: VALID | distance=356 mm extra S2: INVALID", 10)
        readings = store.snapshot(1, 10.5)
        self.assertAlmostEqual(readings[1].calibrated_mm, 330.2)
        self.assertEqual(readings[2].status, "INVALID")
        self.assertEqual(store.snapshot(1, 11.1)[1].status, "STALE")
        store.consume("S1: VALID | distance=0 mm", 12)
        self.assertEqual(store.snapshot(1, 12)[1].status, "INVALID")
        store.disconnect()
        self.assertTrue(all(r.status == "DISCONNECTED" for r in store.snapshot(1, 12).values()))

    def test_plane_distance_normal_projection_and_freshness(self):
        store = self.plane_readings()
        plane = fit_plane(store.snapshot(1, 10), .25)
        self.assertIsNotNone(plane)
        self.assertAlmostEqual(plane.distance, .2, delta=.0005)
        self.assertLess(plane.normal[2], -.99)
        np.testing.assert_allclose(plane.points @ plane.normal + plane.offset, 0, atol=1e-12)
        self.assertAlmostEqual(plane.normal @ plane.projection + plane.offset, 0)
        self.assertIsNone(fit_plane(store.snapshot(1, 12), .25))
        store.consume("S1: VALID | distance=200 mm", 10.5)
        self.assertIsNone(fit_plane(store.snapshot(1, 10.5), .25))
        store.consume("S2: INVALID", 10.5)
        self.assertIsNone(fit_plane(store.snapshot(1, 10.5), .25))

    def test_infrared_geometry_uses_y_down(self):
        np.testing.assert_allclose(SENSOR_POSITIONS * 1000,
                                   [[0, -14.8, 10.8], [-6.7, 5.5, 12.3], [8.7, 5.5, 12.3]])
        self.assertLess(SENSOR_DIRECTIONS[0, 1], 0)  # Upper sensor points upward: negative Y.
        self.assertLess(SENSOR_DIRECTIONS[1, 0], 0)  # S2 points left, toward negative X.
        self.assertGreater(SENSOR_DIRECTIONS[2, 0], 0)  # S3 points right, toward positive X.
        np.testing.assert_allclose(np.linalg.norm(SENSOR_DIRECTIONS, axis=1), 1)
        self.assertTrue(np.all(SENSOR_DIRECTIONS[:, 2] > 0))

    def test_tilted_plane_normal_and_csv_use_y_down(self):
        # Physical target plane: 0.15*x - 0.25*y - z + 0.2 = 0.
        # Use independent mounting geometry to catch Y-up or reversed left/right.
        positions = np.array([[0, -.0148, .0108], [-.0067, .0055, .0123], [.0087, .0055, .0123]])
        directions = np.array([[0, -.259, .966], [-.5, 0, .866], [.5, 0, .866]])
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        normal = np.array([.15, -.25, -1.])
        distances = -(positions @ normal + .2) / (directions @ normal)
        readings = {i: SensorReading(calibrated_mm=distances[i - 1] * 1000, status="VALID", monotonic=10)
                    for i in (1, 2, 3)}
        plane = fit_plane(readings, .25)
        length = np.linalg.norm(normal)
        np.testing.assert_allclose(plane.normal, normal / length, atol=1e-12)
        self.assertAlmostEqual(plane.distance, .2 / length)
        np.testing.assert_allclose(plane.projection, -(.2 / length) * normal / length, atol=1e-12)
        row = record_snapshot(VisionResult(), readings, plane, self.config, now=10)
        self.assertLess(row["s1_point_y_m"], 0)
        self.assertLess(row["s2_point_x_m"], 0)
        self.assertGreater(row["s3_point_x_m"], 0)
        self.assertGreater(row["plane_nx_R"], 0)
        self.assertLess(row["plane_ny_R"], 0)
        self.assertGreater(row["plane_projection_R_y_m"], 0)

    def test_csv_snapshot_and_schema_validation(self):
        store = self.plane_readings()
        readings = store.snapshot(1, 10)
        vision = VisionResult(camera_pose=np.eye(4), timestamp="captured", monotonic=10, status="VALID")
        row = record_snapshot(vision, readings, fit_plane(readings, .25), self.config, now=10)
        self.assertEqual(row["T_W_C_00"], 1)
        self.assertEqual(row["T_W_R_00"], "")
        self.assertTrue(row["plane_valid"])
        self.assertEqual(row["s1_status"], "VALID")
        store.consume("S1: INVALID", 11)
        self.assertEqual(row["s1_status"], "VALID")  # frozen S-press data
        expired = record_snapshot(vision, readings, None, self.config, now=12)
        self.assertEqual(expired["T_W_C_00"], "")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            append_csv(path, "试验组,01", row)
            append_csv(path, "02", row)
            with path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(reader.fieldnames, CSV_FIELDS)
                rows = list(reader)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["group_id"], "试验组,01")
            self.assertEqual(rows[0]["T_W_R_00"], "")
            path.write_text("wrong,header\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                append_csv(path, "03", row)
            with self.assertRaises(ValueError):
                append_csv(path, "  ", row)

    def test_camera_snapshot_chinese_path_pixels_and_duplicate_names(self):
        frame = np.arange(18 * 24 * 3, dtype=np.uint8).reshape(18, 24, 3)
        stamp = datetime(2026, 10, 20, 15, 30, 12, 123456)
        row = dict.fromkeys(CSV_FIELDS, "")
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "data.csv"
            images = Path(directory) / "相机快照"
            _, first = save_record(csv_path, images, "组01/试验:1", row, frame, stamp)
            self.assertEqual(first.name, "组01_试验_1_20261020_153012_123456.png")
            np.testing.assert_array_equal(cv2.imdecode(np.fromfile(first, dtype=np.uint8), cv2.IMREAD_COLOR), frame)
            _, second = save_record(csv_path, images, "组01/试验:1", row, np.zeros_like(frame), stamp)
            self.assertNotEqual(first, second)
            np.testing.assert_array_equal(cv2.imdecode(np.fromfile(first, dtype=np.uint8), cv2.IMREAD_COLOR), frame)
            with csv_path.open(encoding="utf-8-sig", newline="") as stream:
                self.assertEqual([r["group_id"] for r in csv.DictReader(stream)], ["组01/试验:1"] * 2)

    def test_fixed_intrinsics_save_extends_legacy_csv(self):
        estimator = VisionEstimator(self.config)
        estimator.detector = SimpleNamespace(detect=lambda gray: [])
        width, height = self.config["camera_resolution"]
        vision = estimator.process(np.zeros((height, width, 3), dtype=np.uint8))
        row = record_snapshot(vision, {}, None, self.config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            with path.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=LEGACY_CSV_FIELDS)
                writer.writeheader()
                writer.writerow({"group_id": "试验组,01", "camera_z_m": "0.123000"})
            original = path.read_bytes()
            with patch.object(Path, "replace", side_effect=PermissionError("CSV locked")):
                with self.assertRaises(PermissionError):
                    append_csv(path, "02", row)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(path.parent.iterdir()), [path])
            append_csv(path, "02", row)
            append_csv(path, "03", row)
            with path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(reader.fieldnames, CSV_FIELDS)
                old, new, appended = list(reader)
            self.assertEqual(old["group_id"], "试验组,01")
            self.assertEqual(old["camera_z_m"], "0.123000")
            for field in ("camera_fx_px", "camera_fy_px", "focus_position"):
                self.assertEqual(old[field], "")
            self.assertEqual(float(new["camera_fx_px"]), estimator.K[0, 0])
            self.assertEqual(float(new["camera_fy_px"]), estimator.K[1, 1])
            self.assertEqual(new["focus_position"], "")
            self.assertEqual(appended["camera_fx_px"], new["camera_fx_px"])

    def test_camera_snapshot_failure_and_missing_frame(self):
        row = dict.fromkeys(CSV_FIELDS, "")
        stamp = datetime.now()
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "data.csv"
            images = Path(directory) / "images"
            with patch("cv2.imencode", return_value=(False, None)):
                with self.assertRaises(ValueError):
                    save_record(csv_path, images, "01", row, np.zeros((8, 8, 3), dtype=np.uint8), stamp)
            self.assertFalse(csv_path.exists())
            with patch("localization.positioning_core.append_csv", side_effect=PermissionError("CSV locked")):
                with self.assertRaises(PermissionError):
                    save_record(csv_path, images, "01", row, np.zeros((8, 8, 3), dtype=np.uint8), stamp)
            self.assertEqual(list(images.glob("*.png")), [])
            _, image = save_record(csv_path, images, "02", row, None, stamp)
            self.assertIsNone(image)
            self.assertTrue(csv_path.exists())

    def synthetic_estimator(self, ids, tag_rvec=None):
        # Exercise the real detector wrapper and PnP code using exactly projected
        # known corners, independently of a camera or printed tags.
        rotation = cv2.Rodrigues(np.array([np.pi, 0., 0.] if tag_rvec is None else tag_rvec))[0]
        layout = {}
        for index, tag_id in enumerate(ids):
            matrix = np.eye(4)
            matrix[:3, :3] = rotation
            matrix[:3, 3] = [.06 + .04 * index, .08, -.049]
            layout[str(tag_id)] = matrix.tolist()
        with tempfile.TemporaryDirectory() as directory:
            layout_path = Path(directory) / "test_tags.json"
            layout_path.write_text(json.dumps(layout), encoding="utf-8")
            estimator = VisionEstimator({**self.config, "tag_layout_file": layout_path})
        estimator.config = {**self.config, "flip_camera_180": False}
        estimator.K = np.array([[600., 0, 320], [0, 600., 240], [0, 0, 1.]])
        estimator.D = np.zeros(5)
        camera_from_world = np.eye(4)
        camera_from_world[:3, 3] = [-.15, -.06, .5]
        rvec = np.zeros(3)
        tvec = camera_from_world[:3, 3]
        tags = []
        for tag_id in ids:
            pose = estimator.layout[str(tag_id)]
            homogeneous = np.column_stack((estimator.local, np.ones(4)))
            world = (pose @ homogeneous.T).T[:, :3]
            corners = cv2.projectPoints(world, rvec, tvec, estimator.K, estimator.D)[0].reshape(4, 2)
            center = cv2.projectPoints(pose[:3, 3].reshape(1, 3), rvec, tvec,
                                       estimator.K, estimator.D)[0].reshape(2)
            tags.append(SimpleNamespace(tag_id=tag_id, corners=corners, center=center))
        estimator.detector = SimpleNamespace(detect=lambda gray: tags)
        return estimator, inverse_transform(camera_from_world)

    def test_single_tag_pose_recovers_known_camera(self):
        estimator, expected = self.synthetic_estimator([33])
        result = estimator.process(np.zeros((480, 640, 3), dtype=np.uint8))
        self.assertIsNotNone(result.camera_pose, result.status)
        np.testing.assert_allclose(result.camera_pose, expected, atol=1e-6)
        self.assertEqual(len(result.inlier_indices), 4)

    def test_multiple_tag_pose_and_unknown_tag(self):
        estimator, expected = self.synthetic_estimator([33, 24, 7, 15])
        result = estimator.process(np.zeros((480, 640, 3), dtype=np.uint8))
        self.assertIsNotNone(result.camera_pose, result.status)
        np.testing.assert_allclose(result.camera_pose, expected, atol=1e-5)
        self.assertGreaterEqual(len(result.inlier_indices), 12)
        estimator.detector = SimpleNamespace(detect=lambda gray: [SimpleNamespace(
            tag_id=999, corners=np.array([[0, 10], [10, 10], [10, 0], [0, 0.]]), center=np.array([5, 5]))])
        result = estimator.process(np.zeros((480, 640, 3), dtype=np.uint8))
        self.assertIsNone(result.camera_pose)
        self.assertEqual(result.detected_ids, [999])

    def test_custom_tag_configuration_is_used_for_pose(self):
        # Tag IDs/layouts are user-configurable, not tied to the legacy 26 tags.
        estimator, expected = self.synthetic_estimator([123])
        self.assertEqual(set(estimator.layout), {"123"})
        result = estimator.process(np.zeros((480, 640, 3), dtype=np.uint8))
        self.assertEqual(result.known_ids, [123])
        self.assertIsNotNone(result.camera_pose, result.status)
        np.testing.assert_allclose(result.camera_pose, expected, atol=1e-6)

    def test_rotated_tag_matrix_recovers_known_camera(self):
        estimator, expected = self.synthetic_estimator([123], tag_rvec=[2.7, .25, -.35])
        result = estimator.process(np.zeros((480, 640, 3), dtype=np.uint8))
        self.assertIsNotNone(result.camera_pose, result.status)
        np.testing.assert_allclose(result.camera_pose, expected, atol=1e-6)

    def test_weighted_pnp_uses_true_center_and_two_levels_of_weights(self):
        estimator, _ = self.synthetic_estimator([33, 24], tag_rvec=[2.7, .25, -.35])
        tags = estimator.detector.detect(None)
        # Perspective center is not generally the arithmetic average of corners.
        self.assertGreater(np.linalg.norm(tags[0].center - tags[0].corners.mean(0)), 1e-4)
        with patch("localization.positioning_core.refine_weighted_pnp", wraps=refine_weighted_pnp) as solver:
            result = estimator.process(np.zeros((480, 640, 3), dtype=np.uint8))
        self.assertIsNotNone(result.camera_pose, result.status)
        world, image, weights = solver.call_args.args[:3]
        self.assertEqual(world.shape, (10, 3))
        np.testing.assert_allclose(image[-2:], [tag.center for tag in tags])
        np.testing.assert_allclose(weights[:4], result.tag_weights[0])
        np.testing.assert_allclose(weights[4:8], result.tag_weights[1])
        np.testing.assert_allclose(weights[-2:], np.array(result.tag_weights) * 4)
        self.assertEqual(result.center_inlier_ids, [33, 24])
        self.assertLess(result.weighted_reprojection_rmse_px, 1e-5)

    def test_image_center_preference_is_radial_and_bounded(self):
        midpoint = np.array([319.5, 239.5])
        self.assertAlmostEqual(tag_image_weight(midpoint, (640, 480)), 4.)
        self.assertAlmostEqual(tag_image_weight([0, 0], (640, 480)), 1.)
        self.assertGreater(tag_image_weight(midpoint + [50, 0], (640, 480)),
                           tag_image_weight(midpoint + [200, 0], (640, 480)))
        self.assertEqual(tag_image_weight(midpoint + [50, 0], (640, 480)),
                         tag_image_weight(midpoint - [50, 0], (640, 480)))

    def test_high_center_weight_reduces_center_residual(self):
        obj = np.array([[-.1, .1, 0], [.1, .1, 0], [.1, -.1, 0], [-.1, -.1, 0], [0, 0, 0.]])
        K = np.array([[600., 0, 320], [0, 600., 240], [0, 0, 1.]])
        r0, t0 = np.array([.2, -.3, .1]), np.array([.01, -.02, .7])
        observed = cv2.projectPoints(obj, r0, t0, K, np.zeros(5))[0].reshape(-1, 2)
        observed[:4] += [2., -1.]
        errors = []
        for center_weight in (1., 16.):
            rv, tv = refine_weighted_pnp(obj, observed, [1., 1., 1., 1., center_weight], K, np.zeros(5), r0, t0)
            projected = cv2.projectPoints(obj, rv, tv, K, np.zeros(5))[0].reshape(-1, 2)
            errors.append(np.linalg.norm(projected[-1] - observed[-1]))
        self.assertLess(errors[1], errors[0] * .7)

    def test_equal_weights_agree_with_opencv_refinement(self):
        rng = np.random.default_rng(87)
        world = rng.uniform(-.1, .1, (30, 3))
        K = np.array([[600., 0, 320], [0, 600., 240], [0, 0, 1.]])
        rv, tv = np.array([.1, -.2, .05]), np.array([.01, .02, .8])
        observed = cv2.projectPoints(world, rv, tv, K, np.zeros(5))[0].reshape(-1, 2)
        observed += rng.normal(0, .3, observed.shape)
        initial_rv, initial_tv = rv + .02, tv + .01
        a, b = cv2.solvePnPRefineLM(world, observed, K, np.zeros(5), initial_rv.copy(), initial_tv.copy())
        c, d = refine_weighted_pnp(world, observed, np.ones(30), K, np.zeros(5), initial_rv, initial_tv)
        np.testing.assert_allclose(c.ravel(), a.ravel(), atol=1e-6)
        np.testing.assert_allclose(d.ravel(), b.ravel(), atol=1e-6)

    def test_image_center_tag_has_more_influence_on_conflicting_observations(self):
        local = np.array([[-.04, .04, 0], [.04, .04, 0], [.04, -.04, 0], [-.04, -.04, 0], [0, 0, 0.]])
        world = np.vstack([local, local + [.3, 0., 0.]])
        K = np.array([[600., 0, 319.5], [0, 600., 239.5], [0, 0, 1.]])
        rv, tv = np.array([.2, -.1, .03]), np.array([0., 0., .8])
        observed = cv2.projectPoints(world, rv, tv, K, np.zeros(5))[0].reshape(-1, 2)
        observed[5:] += [4., -2.]
        base = np.tile([1., 1., 1., 1., 4.], 2)
        spatial = base * np.repeat([tag_image_weight(observed[4], (640, 480)),
                                    tag_image_weight(observed[9], (640, 480))], 5)
        costs = []
        for weights in (base, spatial):
            r, t = refine_weighted_pnp(world, observed, weights, K, np.zeros(5), rv, tv)
            projected = cv2.projectPoints(world, r, t, K, np.zeros(5))[0].reshape(-1, 2)
            costs.append(np.mean(np.sum((projected[:5] - observed[:5]) ** 2, axis=1)))
        self.assertLess(costs[1], costs[0])

    def test_rejected_tag_cannot_reenter_through_its_center(self):
        estimator, expected = self.synthetic_estimator([33, 24, 7, 15])
        estimator.config["reprojection_error_px"] = 2.
        tags = estimator.detector.detect(None)
        tags[-1].corners += [100., 100.]
        tags[-1].center += [100., 100.]
        cv2.setRNGSeed(1)
        with patch.object(cv2, "putText", wraps=cv2.putText) as draw_text:
            result = estimator.process(np.zeros((480, 640, 3), dtype=np.uint8))
        self.assertIsNotNone(result.camera_pose, result.status)
        self.assertNotIn(15, result.center_inlier_ids)
        self.assertTrue(all(i < 12 for i in result.inlier_indices))
        np.testing.assert_allclose(result.camera_pose, expected, atol=1e-5)
        labels = [(call.args[1], call.args[5]) for call in draw_text.call_args_list
                  if call.args[1].endswith("px") and call.args[6] == 1]
        self.assertEqual(labels, [("0.00px", (0, 255, 0))] * 12 + [("141.42px", (0, 140, 255))] * 4)

    def test_bad_center_is_excluded_while_good_corners_still_solve(self):
        estimator, expected = self.synthetic_estimator([33])
        estimator.detector.detect(None)[0].center += [150., 0.]
        result = estimator.process(np.zeros((480, 640, 3), dtype=np.uint8))
        self.assertIsNotNone(result.camera_pose, result.status)
        self.assertEqual(result.center_inlier_ids, [])
        np.testing.assert_allclose(result.camera_pose, expected, atol=1e-6)

    def test_tag_layout_rejects_invalid_matrices(self):
        invalid_bottom = np.eye(4)
        invalid_bottom[3, 0] = 1
        nonfinite = np.eye(4)
        nonfinite[0, 3] = np.nan
        for matrix in (np.eye(3), np.diag([2, 1, 1, 1]), np.diag([-1, 1, 1, 1]),
                       invalid_bottom, nonfinite):
            with self.subTest(matrix=matrix.tolist()), tempfile.TemporaryDirectory() as directory:
                layout_path = Path(directory) / "invalid_tags.json"
                layout_path.write_text(json.dumps({"123": matrix.tolist()}), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, r"tag_layout\[123\]"):
                    VisionEstimator({**self.config, "tag_layout_file": layout_path})

    def calibration_estimator(self, path):
        # Intrinsics tests use a valid fixture, independent of the user's layout.
        layout_path = path.parent / "calibration_test_tags.json"
        layout_path.write_text(json.dumps({"123": np.eye(4).tolist()}), encoding="utf-8")
        return VisionEstimator({**self.config, "calibration_file": path, "tag_layout_file": layout_path})

    def test_camera_calibration_json_loads_matrix_and_distortion(self):
        matrix = [[710., 0., 312.], [0., 705., 238.], [0., 0., 1.]]
        distortion = [.12, -.25, .001, -.002, .08]
        for values in (distortion, [distortion], [[value] for value in distortion]):
            with self.subTest(distortion=values), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "相机内参.json"
                path.write_text(json.dumps({"camera_matrix": matrix, "dist_coeffs": values, "rms": .5}),
                                encoding="utf-8-sig")
                estimator = self.calibration_estimator(path)
                self.assertTrue(estimator.calibrated)
                np.testing.assert_array_equal(estimator.K, matrix)
                np.testing.assert_array_equal(estimator.D, distortion)

    def test_missing_camera_calibration_uses_default_intrinsics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.json"
            estimator = self.calibration_estimator(path)
        self.assertFalse(estimator.calibrated)
        width, height = self.config["camera_resolution"]
        np.testing.assert_array_equal(estimator.K, [[600., 0., width / 2], [0., 600., height / 2], [0., 0., 1.]])
        np.testing.assert_array_equal(estimator.D, np.zeros(5))

    def test_camera_calibration_json_rejects_invalid_data(self):
        valid = {"camera_matrix": [[600., 0., 320.], [0., 600., 240.], [0., 0., 1.]],
                 "dist_coeffs": [0., 0., 0., 0., 0.]}
        invalid = ["{", "[]", json.dumps({"camera_matrix": valid["camera_matrix"]}),
                   json.dumps({**valid, "camera_matrix": np.eye(2).tolist()}),
                   json.dumps({**valid, "dist_coeffs": [0., 0., 0.]}),
                   json.dumps({**valid, "dist_coeffs": [[0., 0.], [0., 0.]]}),
                   json.dumps({**valid, "dist_coeffs": [np.nan, 0., 0., 0., 0.]})]
        for contents in invalid:
            with self.subTest(contents=contents), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "invalid.json"
                path.write_text(contents, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Invalid camera calibration"):
                    self.calibration_estimator(path)

    def test_custom_infrared_json_is_used_by_acquisition(self):
        from localization.positioning import Acquisition
        parameters = {"S1": {"slope": 2., "intercept_mm": -10.},
                      "S2": {"slope": 3., "intercept_mm": -20.},
                      "S3": {"slope": 4., "intercept_mm": -30.}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration_path = root / "标定" / "infrared.json"
            calibration_path.parent.mkdir()
            calibration_path.write_text(json.dumps(parameters), encoding="utf-8-sig")
            config_data = json.loads((ROOT / "config/positioning_config.json").read_text(encoding="utf-8"))
            config_data["infrared_calibration_file"] = "标定/infrared.json"
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config_data), encoding="utf-8")
            config = load_config(config_path)
            self.assertEqual(config["infrared_calibration_file"], calibration_path.resolve())
            source = Acquisition(config)
            source.sensors.consume("S1: VALID | distance=100 mm S2: VALID | distance=200 mm S3: VALID | distance=300 mm", 10)
            readings = source.sensors.snapshot(1, 10)
            self.assertEqual([readings[i].calibrated_mm for i in (1, 2, 3)], [190., 580., 1170.])
            source.sensors.consume("S1: VALID | distance=1 mm", 11)
            self.assertEqual(source.sensors.snapshot(1, 11)[1].status, "INVALID")

    def test_infrared_json_rejects_missing_or_invalid_parameters(self):
        valid = {f"S{i}": {"slope": 1., "intercept_mm": 0.} for i in (1, 2, 3)}
        invalid = ["{", "[]", json.dumps({"S1": valid["S1"]}),
                   json.dumps({**valid, "S2": {"slope": 1.}})]
        for parameters in ({"slope": 0., "intercept_mm": 0.},
                           {"slope": -1., "intercept_mm": 0.},
                           {"slope": 1., "intercept_mm": np.nan},
                           {"slope": np.inf, "intercept_mm": 0.},
                           {"slope": "1", "intercept_mm": 0.},
                           {"slope": True, "intercept_mm": 0.}):
            invalid.append(json.dumps({**valid, "S3": parameters}))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "infrared.json"
            with self.assertRaises(FileNotFoundError):
                load_infrared_calibration(path)
            for contents in invalid:
                with self.subTest(contents=contents):
                    path.write_text(contents, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_infrared_calibration(path)


if __name__ == "__main__":
    unittest.main()
