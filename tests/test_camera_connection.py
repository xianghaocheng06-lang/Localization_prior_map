import queue
import threading
import time
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from localization.positioning import Acquisition
from localization.positioning_core import VisionResult


class FakeCapture:
    def __init__(self, index):
        self.index = index
        self.released = False
        self.frames = queue.Queue()
        self.focus = 200.
        self.focus_mode = 1

    def isOpened(self):
        return self.index != 9

    def getBackendName(self):
        return "DSHOW"

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_AUTOFOCUS:
            self.focus_mode = 1 if value else 2
        elif prop == cv2.CAP_PROP_FOCUS:
            self.focus = float(value)
        return True

    def get(self, prop):
        return self.focus_mode if prop == cv2.CAP_PROP_AUTOFOCUS else self.focus

    def read(self):
        try:
            frame = self.frames.get(timeout=.01)
        except queue.Empty:
            frame = np.full((2, 3, 3), self.index, dtype=np.uint8)
        if isinstance(frame, Exception):
            raise frame
        return True, frame

    def release(self):
        self.released = True


class FakeEstimator:
    calibrated = True

    def __init__(self):
        self.block_next = False
        self.processing = threading.Event()
        self.resume = threading.Event()

    def process(self, frame, focus_position=None):
        if self.block_next:
            self.block_next = False
            self.processing.set()
            self.resume.wait(3)
        pose = np.eye(4)
        pose[0, 3] = int(frame[0, 0, 0])
        return VisionResult(frame=frame, camera_pose=pose, status="VALID", focus_position=focus_position,
                            timestamp="test", monotonic=time.monotonic())


class CameraConnectionTests(unittest.TestCase):
    def setUp(self):
        self.captures = []
        self.estimator = FakeEstimator()
        self.source = Acquisition({"infrared_calibration": {i: (1., 0.) for i in (1, 2, 3)},
                                   "serial_port": "COM4", "serial_baud": 115200,
                                   "camera_index": 1, "camera_resolution": [3, 2], "camera_fps": 15})
        self.source.serial_loop = Mock()
        self.capture_patch = patch("cv2.VideoCapture", side_effect=self.open_camera)
        self.estimator_patch = patch("localization.positioning.VisionEstimator", return_value=self.estimator)
        self.capture_patch.start()
        self.estimator_patch.start()
        self.source.start()
        self.wait_for(lambda: self.source.get_vision().frame is not None)

    def tearDown(self):
        self.estimator.resume.set()
        self.source.stop()
        self.estimator_patch.stop()
        self.capture_patch.stop()
        self.assertTrue(all(not thread.is_alive() for thread in self.source.threads))
        self.assertTrue(all(capture.released for capture in self.captures))

    def open_camera(self, index, *backend):
        if any(not capture.released for capture in self.captures):
            raise AssertionError("previous camera was not released before switching")
        capture = FakeCapture(index)
        self.captures.append(capture)
        return capture

    def test_dynamic_focus_readback_is_passed_to_estimator_after_stabilizing(self):
        self.source.config['focus_map_file']='mock-focus-map.json'
        self.source.config['flip_camera_180']=False
        self.source.connect_camera(2)
        self.wait_for(lambda: self.source.get_vision().focus_position == 200.)
        self.assertEqual(self.source.get_vision().camera_pose[0,3],2)

    def test_manual_focus_and_auto_resume_on_same_capture(self):
        self.source.config.update(focus_map_file="mock-focus-map.json", flip_camera_180=False)
        self.source.connect_camera(2)
        self.wait_for(lambda: self.source.get_vision().focus_position == 200.)
        capture = self.captures[-1]
        self.source.request_camera_focus("304")
        self.assertIsNone(self.source.get_vision().camera_pose)
        self.wait_for(lambda: self.source.get_vision().focus_position == 304.)
        manual = self.source.get_vision()
        self.assertEqual((manual.focus_mode, manual.focus_readback), ("MANUAL", 304.))
        self.source.request_camera_focus()
        self.wait_for(lambda: self.source.get_vision().focus_mode == "AUTO")
        capture.focus = 288.
        self.wait_for(lambda: self.source.get_vision().focus_position == 288.)
        self.assertIs(self.captures[-1], capture)

    def wait_for(self, condition):
        deadline = time.monotonic() + 3
        while not condition():
            if time.monotonic() > deadline:
                self.fail(f"camera operation timed out: {self.source.get_camera_status()}")
            time.sleep(.01)

    def pose_is(self, index):
        pose = self.source.get_vision().camera_pose
        return pose is not None and pose[0, 3] == index

    def test_switch_discards_inflight_pose_and_preserves_infrared(self):
        self.source.sensors.consume("S1: VALID | distance=123 mm")
        self.estimator.block_next = True
        self.assertTrue(self.estimator.processing.wait(2))
        self.source.connect_camera(2)
        self.assertIsNone(self.source.get_vision().frame)
        self.assertIsNone(self.source.get_vision().camera_pose)
        self.estimator.resume.set()
        self.wait_for(lambda: self.pose_is(2))
        self.assertTrue(self.captures[0].released)
        self.assertEqual(self.source.sensors.snapshot(5)[1].calibrated_mm, 123.)
        self.assertEqual(self.source.sensors.snapshot(5)[1].status, "VALID")
        self.source.serial_loop.assert_called_once()
        self.source.disconnect_camera()
        self.wait_for(lambda: self.captures[-1].released)
        self.assertIsNone(self.source.get_vision().frame)
        self.assertIsNone(self.source.get_vision().camera_pose)
        self.assertEqual(self.source.get_camera_status(), "未连接")

    def test_unplug_and_reconnect_to_same_camera(self):
        self.captures[0].frames.put(OSError("USB removed"))
        self.wait_for(lambda: "USB removed" in self.source.get_camera_status())
        self.assertIsNone(self.source.get_vision().camera_pose)
        self.source.connect_camera(1)
        self.wait_for(lambda: len(self.captures) == 2 and self.pose_is(1))

    def test_failed_open_can_be_retried(self):
        self.source.connect_camera(9)
        self.wait_for(lambda: "无法打开摄像头 9" in self.source.get_camera_status())
        self.assertIsNone(self.source.get_vision().frame)
        self.source.connect_camera(3)
        self.wait_for(lambda: self.pose_is(3))

    def test_resolution_mismatch_rejects_frame(self):
        self.captures[0].frames.put(np.zeros((4, 5, 3), dtype=np.uint8))
        self.wait_for(lambda: "相机分辨率 5x4" in self.source.get_camera_status())
        self.assertIsNone(self.source.get_vision().frame)
        self.assertIsNone(self.source.get_vision().camera_pose)

    def test_invalid_index_keeps_current_camera(self):
        for index in (-1, "bad", "1.5"):
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.source.connect_camera(index)
        self.assertEqual(self.source.camera_request[0], 1)
        self.assertTrue(self.pose_is(1))


if __name__ == "__main__":
    unittest.main()
