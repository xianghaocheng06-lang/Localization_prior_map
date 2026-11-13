import queue
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from localization.positioning import Acquisition


class FakePort:
    def __init__(self, device, baud, factory):
        self.port, self.baudrate = device, baud
        self.factory = factory
        self.chunks = queue.Queue()
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    @property
    def in_waiting(self):
        return self.chunks.qsize()

    def read(self, size):
        try:
            chunk = self.chunks.get(timeout=.02)
        except queue.Empty:
            return b""
        if isinstance(chunk, Exception):
            raise chunk
        return chunk

    def cancel_read(self):
        self.chunks.put(b"")


class SerialConnectionTests(unittest.TestCase):
    def setUp(self):
        self.ports = []
        self.source = Acquisition({"infrared_calibration": {i: (1., 0.) for i in (1, 2, 3)},
                                   "serial_port": "COM4", "serial_baud": 115200, "camera_index": 1,
                                   "sensor_timeout_s": .1})
        self.source.camera_loop = Mock()
        for setting, value in (("SERIAL_RETRY_S", .15), ("SERIAL_SILENCE_S", .75)):
            timer = patch(f"localization.positioning.{setting}", value)
            timer.start()
            self.addCleanup(timer.stop)
        self.serial_patch = patch("serial.Serial", side_effect=self.open_port)
        self.serial_patch.start()
        self.source.start()
        self.wait_for(lambda: len(self.ports) == 1)

    def tearDown(self):
        self.source.stop()
        self.serial_patch.stop()
        self.assertTrue(all(not thread.is_alive() for thread in self.source.threads))
        self.assertTrue(all(port.closed for port in self.ports))

    def open_port(self, device, baud, **kwargs):
        if device == "COM_BAD":
            raise OSError("port busy")
        if any(not port.closed for port in self.ports):
            raise AssertionError("previous port was not closed before reconnecting")
        port = FakePort(device, baud, self)
        self.ports.append(port)
        return port

    def wait_for(self, condition):
        deadline = time.monotonic() + 3
        while not condition():
            if time.monotonic() > deadline:
                self.fail(f"serial operation timed out: {self.source.get_serial_status()}")
            time.sleep(.01)

    def test_switch_port_discards_old_buffer_and_keeps_camera_running(self):
        self.ports[0].chunks.put(b"S1: VAL")
        self.ports[0].chunks.put(b"ID | distance=100 mm\nS2: INVALID\n")
        self.wait_for(lambda: self.source.sensors.snapshot(5)[1].status == "VALID")
        self.assertEqual(self.source.sensors.snapshot(5)[1].calibrated_mm, 100.)
        self.ports[0].chunks.put(b"S3: VALID | distance=")
        self.wait_for(lambda: self.ports[0].chunks.empty())
        self.source.connect_serial("COM8", 57600)
        self.wait_for(lambda: len(self.ports) == 2)
        self.assertEqual((self.ports[1].port, self.ports[1].baudrate), ("COM8", 57600))
        self.ports[1].chunks.put(b"999 mm\nS2: VALID | distance=200 mm\n")
        self.wait_for(lambda: self.source.sensors.snapshot(5)[2].status == "VALID")
        self.assertEqual(self.source.sensors.snapshot(5)[3].status, "DISCONNECTED")
        self.source.disconnect_serial()
        self.wait_for(lambda: self.ports[1].closed)
        self.assertEqual(self.source.get_serial_status(), "未连接")
        self.assertTrue(all(reading.status == "DISCONNECTED" for reading in self.source.sensors.snapshot(5).values()))
        self.source.camera_loop.assert_called_once()

    def test_unplug_automatically_reconnects_to_same_port(self):
        self.ports[0].chunks.put(OSError("USB removed"))
        self.wait_for(lambda: len(self.ports) == 2)
        self.assertTrue(self.ports[0].closed)
        self.assertEqual(self.ports[1].port, "COM4")
        self.ports[1].chunks.put(b"S3: VALID | distance=321 mm\n")
        self.wait_for(lambda: self.source.sensors.snapshot(5)[3].status == "VALID")
        self.assertEqual(self.source.sensors.snapshot(5)[3].calibrated_mm, 321.)

    def test_stream_without_firmware_reports_reconnects(self):
        original = self.ports[0]
        original.chunks.put(b"S1: VALID | distance=100 mm\n")
        self.wait_for(lambda: self.source.sensors.snapshot(5)[1].status == "VALID")
        original.chunks.put(b"debug without measurements\n")
        self.wait_for(lambda: self.source.sensors.snapshot(.1)[1].status == "STALE")
        self.wait_for(lambda: len(self.ports) == 2)
        self.assertTrue(original.closed)
        self.ports[1].chunks.put(b"S1: VALID | distance=200 mm\n")
        self.wait_for(lambda: self.source.sensors.snapshot(5)[1].calibrated_mm == 200.)

    def test_usb_device_returns_under_a_different_com_number(self):
        usb_id = (0x1A86, 0x7523)
        unrelated = SimpleNamespace(device="COM6", vid=None, pid=None)
        adapter = SimpleNamespace(device="COM3", vid=usb_id[0], pid=usb_id[1])
        with patch("serial.tools.list_ports.comports", return_value=[unrelated, adapter]) as comports:
            self.source.connect_serial("COM4", 115200, usb_id=usb_id)
            self.wait_for(lambda: len(self.ports) == 2)
            self.assertEqual(self.ports[1].port, "COM3")
            comports.return_value = [unrelated]
            self.ports[1].chunks.put(OSError("USB removed"))
            self.wait_for(lambda: self.ports[1].closed and "未检测到" in self.source.get_serial_status())
            adapter.device = "COM10"
            comports.return_value = [unrelated, adapter]
            self.wait_for(lambda: len(self.ports) == 3)
            self.assertEqual(self.ports[2].port, "COM10")
            self.source.disconnect_serial()
            self.wait_for(lambda: self.ports[2].closed)
            self.assertEqual(self.source.get_serial_status(), "未连接")

    def test_failed_open_can_be_retried_without_restart(self):
        self.source.connect_serial("COM_BAD", 115200)
        self.wait_for(lambda: "port busy" in self.source.get_serial_status())
        self.assertTrue(self.ports[0].closed)
        self.source.connect_serial("COM9", 9600)
        self.wait_for(lambda: len(self.ports) == 2)
        self.wait_for(lambda: self.source.get_serial_status() == "已连接 COM9 (9600)")

    def test_invalid_connection_parameters_leave_current_connection_intact(self):
        for device, baud in (("", 115200), ("COM8", 0), ("COM8", "bad")):
            with self.subTest(device=device, baud=baud), self.assertRaises(ValueError):
                self.source.connect_serial(device, baud)
        self.assertEqual(self.source.serial_request[:2], ("COM4", 115200))


if __name__ == "__main__":
    unittest.main()
