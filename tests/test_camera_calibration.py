import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from calibration.camera_calibration import (
    board_spec, calibrate_points, detect_corners, main, object_points,
    read_image, resolve_dataset, undistort_image, write_image, write_json,
    lock_focus, verify_focus,
    set_manual_focus, FocusControlError,
)
from localization.positioning_core import VisionEstimator, load_config, load_camera_calibration

ROOT = Path(__file__).resolve().parents[1]


class CameraCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.spec = board_spec(9, 6, 20)
        self.matrix = np.array([[550., 0, 319.2], [0, 558., 241.5], [0, 0, 1.]])
        self.dist = np.array([.1, -.22, .001, -.002, .08])

    def projected_corners(self):
        obj = object_points(self.spec)
        rng = np.random.default_rng(1234)
        points = []
        for _ in range(22):
            rv = rng.uniform([-.55, -.6, -.3], [.55, .6, .3])
            tv = np.array([rng.uniform(-.13, .02), rng.uniform(-.09, .01), rng.uniform(.35, .65)])
            projected, _ = cv2.projectPoints(obj, rv, tv, self.matrix, self.dist)
            points.append(projected.astype(np.float32))
        return points

    def test_known_intrinsics_recovered_from_projected_points(self):
        result = calibrate_points(self.projected_corners(), self.spec, (640, 480))
        np.testing.assert_allclose(result["camera_matrix"], self.matrix, atol=.01)
        np.testing.assert_allclose(result["dist_coeffs"], self.dist, atol=.002)
        self.assertLess(result["rms"], .001)
        self.assertEqual(len(result["per_view_rmse_px"]), 22)
        self.assertAlmostEqual(result["rms"], np.sqrt(np.mean(np.square(result["per_view_rmse_px"]))), delta=1e-5)

    def test_board_points_use_inner_corners_and_meters(self):
        points = object_points(self.spec)
        self.assertEqual(points.shape, (54, 3))
        np.testing.assert_allclose(points[-1], [.16, .10, 0], atol=1e-7)
        for value in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                board_spec(9, 6, value)
        with self.assertRaises(ValueError):
            calibrate_points(self.projected_corners()[:9], self.spec, (640, 480))

    def test_generated_pattern_detects_54_corners_and_chinese_path(self):
        image = np.full((450, 600, 3), 255, np.uint8)
        for row in range(7):
            for col in range(10):
                if (row + col) % 2 == 0:
                    image[50 + row*50:100 + row*50, 50 + col*50:100 + col*50] = 0
        with tempfile.TemporaryDirectory(prefix="标定_") as temp:
            file = Path(temp) / "棋盘图.png"
            write_image(file, image)
            restored = read_image(file)
            np.testing.assert_array_equal(restored, image)
            corners = detect_corners(restored, (9, 6))
            self.assertIsNotNone(corners)
            self.assertEqual(corners.shape, (54, 1, 2))

    def test_capture_metadata_prevents_double_rotation_and_conflicts(self):
        with tempfile.TemporaryDirectory() as temp:
            write_json(Path(temp) / "capture.json", {
                **self.spec, "flip_camera_180": True, "images_already_oriented": True,
                "image_size": [640, 480], "camera_index": 1})
            args = SimpleNamespace(images=temp, cols=None, rows=None, square_mm=None, flip_180=None)
            _, _, spec, orientation, rotate = resolve_dataset(args)
            self.assertEqual(spec, self.spec)
            self.assertTrue(orientation)
            self.assertFalse(rotate)
            args.flip_180 = False
            with self.assertRaises(ValueError):
                resolve_dataset(args)
            args.flip_180 = None
            args.square_mm = 19.9
            self.assertEqual(resolve_dataset(args)[2]["square_size_mm"], 19.9)

    def test_capture_saves_oriented_clean_image_and_releases_camera(self):
        frame = np.arange(480 * 640 * 3, dtype=np.uint8).reshape(480, 640, 3)
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.read.side_effect = [(True, frame.copy()), (True, frame.copy())]
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / "config.json"
            write_json(config, {"camera_index": 1, "camera_resolution": [640, 480],
                               "camera_fps": 15, "flip_camera_180": True})
            argv = ["capture", "--config", str(config), "--output-dir", temp, "--min-sharpness", "0"]
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch("calibration.camera_calibration.cv2.VideoCapture", return_value=cap))
                stack.enter_context(patch("calibration.camera_calibration.detect_corners", return_value=self.projected_corners()[0]))
                for name in ("namedWindow", "imshow", "destroyAllWindows"):
                    stack.enter_context(patch("calibration.camera_calibration.cv2." + name))
                stack.enter_context(patch("calibration.camera_calibration.cv2.waitKey", side_effect=[ord("s"), ord("q")]))
                stack.enter_context(patch("calibration.camera_calibration.cv2.getWindowProperty", return_value=1))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                self.assertEqual(main(argv), 0)
            session = next(Path(temp).glob("session_*"))
            np.testing.assert_array_equal(read_image(session / "frame_0001.png"), cv2.flip(frame, -1))
            metadata = json.loads((session / "capture.json").read_text(encoding="utf-8"))
            self.assertTrue(metadata["images_already_oriented"])
            self.assertTrue(metadata["flip_camera_180"])
            cap.release.assert_called_once()

    def test_fixed_focus_checks_dshow_mode_and_drift(self):
        cap = MagicMock()
        cap.getBackendName.return_value = 'DSHOW'
        current = {'focus': 259., 'af': 1.}
        def setting(prop, value):
            if prop == cv2.CAP_PROP_AUTOFOCUS: current['af'] = 1. if value else 2.
            if prop == cv2.CAP_PROP_FOCUS: current['focus'] = value
            return True
        cap.set.side_effect = setting
        cap.get.side_effect = lambda prop: current['af'] if prop == cv2.CAP_PROP_AUTOFOCUS else current['focus']
        cap.read.return_value = (True, np.zeros((480,640,3), np.uint8))
        state = lock_focus(cap, 259., 0)
        self.assertEqual(state['autofocus_mode'], 'MANUAL')
        self.assertEqual(state['focus_readback'], 259.)
        current['focus'] = 264.
        with self.assertRaises(ValueError): verify_focus(cap, 259.)
        current.update(focus=259., af=1.)
        with self.assertRaises(ValueError): verify_focus(cap, 259.)
        cap.set.side_effect = None
        cap.set.return_value = False
        with self.assertRaises(ValueError): lock_focus(cap, 259., 0)

    def test_manual_focus_retries_ignored_write_without_resetting_manual_mode(self):
        class Camera:
            def __init__(self):
                self.q,self.writes,self.af_writes=144.,0,0
                self.pending=None
                self.pending_frames=0
            def getBackendName(self):return 'DSHOW'
            def get(self,prop):return 2. if prop==cv2.CAP_PROP_AUTOFOCUS else self.q
            def set(self,prop,value):
                if prop==cv2.CAP_PROP_AUTOFOCUS:self.af_writes+=1
                if prop==cv2.CAP_PROP_FOCUS:
                    self.writes+=1
                    if self.writes>=2:self.pending=value
                return True
            def read(self):
                if self.pending is not None:
                    self.pending_frames+=1
                    if self.pending_frames>=2:self.q=self.pending
                return True,np.zeros((4,4,3),np.uint8)
        camera=Camera();events=[];tick=[0.]
        def clock():tick[0]+=.1;return tick[0]
        with patch('calibration.camera_calibration.time.monotonic',side_effect=clock):
            state=set_manual_focus(camera,272,settle=.1,timeout=.3,on_event=events.append)
        self.assertEqual(state['focus_readback'],272)
        self.assertEqual(camera.writes,2)
        self.assertEqual(camera.af_writes,0)
        self.assertIn('focus_timeout',[e['action'] for e in events])
        self.assertEqual(events[-1]['action'],'verified')

    def test_manual_focus_waits_for_mode_and_never_accepts_stuck_readback(self):
        cap=MagicMock();cap.getBackendName.return_value='DSHOW';mode={'af':1.,'frames':0};writes=[]
        def read():
            mode['frames']+=1
            if mode['frames']>=3:mode['af']=2.
            return True,np.zeros((4,4,3),np.uint8)
        def setting(prop,value):
            if prop==cv2.CAP_PROP_FOCUS:writes.append(mode['frames'])
            return True
        cap.read.side_effect=read;cap.set.side_effect=setting
        cap.get.side_effect=lambda prop:mode['af'] if prop==cv2.CAP_PROP_AUTOFOCUS else 144.
        tick=[0.]
        def clock():tick[0]+=.1;return tick[0]
        with patch('calibration.camera_calibration.time.monotonic',side_effect=clock):
            with self.assertRaises(FocusControlError):
                set_manual_focus(cap,272,settle=.1,timeout=1.,retries=3)
        self.assertEqual(len(writes),3)
        self.assertTrue(all(n>=3 for n in writes))

    def test_external_images_require_explicit_orientation(self):
        with tempfile.TemporaryDirectory() as temp:
            args = SimpleNamespace(images=temp, cols=None, rows=None, square_mm=None, flip_180=None)
            with self.assertRaises(ValueError):
                resolve_dataset(args)
            args.flip_180 = True
            self.assertEqual(resolve_dataset(args)[3:], (True, True))

    def test_undistortion_checks_resolution_and_preserves_dimensions(self):
        result = {"image_size": [640, 480], "camera_matrix": self.matrix, "dist_coeffs": self.dist}
        corrected, matrix, roi = undistort_image(np.zeros((480, 640, 3), np.uint8), result)
        self.assertEqual(corrected.shape, (480, 640, 3))
        self.assertEqual(matrix.shape, (3, 3))
        self.assertEqual(len(roi), 4)
        with self.assertRaises(ValueError):
            undistort_image(np.zeros((240, 320, 3), np.uint8), result)

    def test_localization_accepts_metadata_and_rejects_mismatches(self):
        with tempfile.TemporaryDirectory() as temp:
            config = load_config(ROOT / "config/positioning_config.json")
            config.pop("focus_map_file", None)
            config["calibration_file"] = Path(temp) / "intrinsics.json"
            data = {"camera_matrix": self.matrix.tolist(), "dist_coeffs": self.dist.tolist(),
                    "model": "opencv_pinhole", "image_size": config["camera_resolution"],
                    "flip_camera_180": config["flip_camera_180"]}
            write_json(config["calibration_file"], data)
            estimator = VisionEstimator(config)
            np.testing.assert_allclose(estimator.K, self.matrix)
            for key, value in [("image_size", [1280, 720]), ("flip_camera_180", False),
                                ("flip_camera_180", "true"), ("model", "opencv_fisheye")]:
                write_json(config["calibration_file"], {**data, key: value})
                with self.assertRaises(ValueError):
                    VisionEstimator(config)

    def test_loader_rejects_noncanonical_matrix_and_invalid_rms(self):
        data = {"camera_matrix": self.matrix.tolist(), "dist_coeffs": self.dist.tolist()}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "内参.json"
            for row, col, value in [(2, 2, 2), (0, 1, 1), (1, 0, 1), (0, 0, -1)]:
                matrix = self.matrix.copy()
                matrix[row, col] = value
                write_json(path, {**data, "camera_matrix": matrix.tolist()})
                with self.assertRaisesRegex(ValueError, "Invalid camera calibration"):
                    load_camera_calibration(path, (640, 480), True)
            for rms in [-1, True, "0.5"]:
                write_json(path, {**data, "rms": rms})
                with self.assertRaisesRegex(ValueError, "rms"):
                    load_camera_calibration(path, (640, 480), True)

    def test_project_configuration_loads_selected_intrinsics_exactly(self):
        config = load_config(ROOT / "config/positioning_config.json")
        data = json.loads(config["calibration_file"].read_text(encoding="utf-8"))
        matrix, distortion, info = load_camera_calibration(
            config["calibration_file"], config["camera_resolution"], config["flip_camera_180"])
        np.testing.assert_array_equal(matrix, data["camera_matrix"])
        np.testing.assert_array_equal(distortion, data["dist_coeffs"])
        self.assertEqual(info["image_size"], [640, 480])

    def test_cli_rejects_mixed_sizes_without_writing_calibration(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            write_image(folder / "a.png", np.zeros((480, 640, 3), np.uint8))
            write_image(folder / "b.png", np.zeros((240, 320, 3), np.uint8))
            output = folder / "result.json"
            with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                code = main(["calibrate", "--images", temp, "--no-flip-180", "--output", str(output)])
            self.assertEqual(code, 1)
            self.assertFalse(output.exists())

    def test_full_calibration_workflow_writes_project_compatible_outputs(self):
        points = iter(self.projected_corners())
        with tempfile.TemporaryDirectory(prefix="标定流程_") as temp:
            folder = Path(temp) / "images"
            folder.mkdir()
            write_json(folder / "capture.json", {
                **self.spec, "flip_camera_180": True, "images_already_oriented": True,
                "image_size": [640, 480], "camera_index": 1,
                "focus_position": 259., "focus_lock_verified": True, "backend": "DSHOW",
                "focus_samples": {f"frame_{i:04d}.png": {"focus_readback": 259., "autofocus_mode": "MANUAL"} for i in range(22)}})
            for i in range(22):
                write_image(folder / f"frame_{i:04d}.png", np.full((480, 640, 3), 160, np.uint8))
            output = Path(temp) / "result.json"
            argv = ["calibrate", "--images", str(folder), "--output", str(output)]
            with patch("calibration.camera_calibration.detect_corners", side_effect=lambda *_: next(points)), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(argv), 0)
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["image_size"], [640, 480])
            self.assertTrue(result["flip_camera_180"])
            self.assertEqual(result["num_images"], 22)
            self.assertEqual(result['focus_position'], 259.)
            np.testing.assert_allclose(result["camera_matrix"], self.matrix, atol=.01)
            diagnostics = output.parent / "result_diagnostics"
            self.assertEqual(len(list((diagnostics / "corners").glob("*.png"))), 22)
            self.assertTrue((diagnostics / "per_view_errors.csv").exists())
            self.assertEqual(read_image(diagnostics / "undistortion_comparison.png").shape, (480, 1280, 3))
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(argv), 1)  # Do not overwrite without explicit flag.


if __name__ == "__main__":
    unittest.main()
