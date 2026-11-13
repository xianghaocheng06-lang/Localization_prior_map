#!/usr/bin/env python3
"""Unified AprilTag / infrared positioning window. Run with --demo without hardware."""
from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from localization.positioning_core import (
    SENSOR_IDS, SENSOR_POSITIONS, SENSOR_DIRECTIONS,
    SensorStore, VisionResult, VisionEstimator, load_config, load_tag_layout, robot_pose,
    fit_plane, measured_points, transform_points, record_snapshot, save_record,
)

ROOT = Path(__file__).resolve().parents[1]
SERIAL_RETRY_S = 2.0
SERIAL_SILENCE_S = 5.0


def resolve_serial_device(preferred, usb_id):
    if usb_id is None:
        return preferred
    from serial.tools.list_ports import comports

    matches = [port.device for port in comports() if (port.vid, port.pid) == usb_id]
    if preferred in matches:
        return preferred
    if len(matches) == 1:
        return matches[0]
    if matches:
        raise OSError("检测到多个匹配的红外 USB 串口，请手动选择")
    raise OSError(f"未检测到红外 USB 串口 {usb_id[0]:04X}:{usb_id[1]:04X}")


class Acquisition:
    """Hardware I/O and detection run outside the Qt render/event thread."""
    def __init__(self, config, demo=False):
        self.config, self.demo = config, demo
        self.sensors = SensorStore(config["infrared_calibration"])
        self.logs = queue.Queue()
        self.lock = threading.Lock()
        self.vision = VisionResult()
        self.stop_event = threading.Event()
        self.threads = []
        self.serial_port = None
        self.serial_lock = threading.Lock()
        self.serial_wakeup = threading.Event()
        self.serial_request = None
        self.serial_version = 0
        self.serial_status = "模拟模式" if demo else "未连接"
        self.camera_wakeup = threading.Event()
        self.camera_request = None
        self.camera_version = 0
        self.camera_status = "模拟模式" if demo else "未连接"
        self.focus_request = None

    def start(self):
        if not self.demo:
            self.connect_serial(self.config["serial_port"], self.config["serial_baud"],
                                usb_id=self.config.get("serial_usb_id"))
            self.connect_camera(self.config["camera_index"])
        for task in ([self.demo_loop] if self.demo else [self.camera_loop, self.serial_loop]):
            thread = threading.Thread(target=task, daemon=True)
            self.threads.append(thread)
            thread.start()

    def get_vision(self):
        with self.lock:
            return self.vision

    def set_vision(self, result):
        with self.lock:
            self.vision = result

    def connect_camera(self, index):
        index = int(index)
        if index < 0:
            raise ValueError("相机编号须为非负整数")
        if self.demo:
            return
        with self.lock:
            self.camera_version += 1
            self.camera_request = (index, self.camera_version)
            self.camera_status = f"正在连接相机 {index}…"
            self.vision = VisionResult(status=f"CAMERA CONNECTING: {index}")
            self.focus_request = None
        self.camera_wakeup.set()

    def disconnect_camera(self):
        if self.demo:
            return
        with self.lock:
            self.camera_request = None
            self.camera_status = "未连接"
            self.vision = VisionResult(status="CAMERA DISCONNECTED")
            self.focus_request = None
        self.camera_wakeup.set()

    def get_camera_status(self):
        with self.lock:
            return self.camera_status

    def can_request_focus(self):
        with self.lock:
            return self.camera_request is not None

    def request_camera_focus(self, target=None):
        """Send one focus request to the current camera's capture thread."""
        if target is not None:
            target = int(target)
            if target < 0:
                raise ValueError("请输入非负整数对焦读数")
        with self.lock:
            if self.camera_request is None:
                raise ValueError("请先连接相机")
            self.focus_request = ("auto" if target is None else "manual", target)
            self.vision = VisionResult(status="FOCUSING (applying focus request)")

    def focus_preview(self, frame, status):
        import cv2
        if self.config["flip_camera_180"]:
            frame = cv2.flip(frame, -1)
        return VisionResult(frame=frame, timestamp=datetime.now().astimezone().isoformat(timespec="milliseconds"),
                            monotonic=time.monotonic(), status=status)

    def apply_focus_request(self, cap, camera_request, focus_request):
        import cv2
        from calibration.camera_calibration import set_manual_focus

        mode, target = focus_request

        def preview(frame, phase, state):
            result = self.focus_preview(frame, f"FOCUSING ({phase})")
            readback = state["focus_readback"]
            result.focus_readback = readback if readback is not None and readback >= 0 else None
            result.focus_mode = state["autofocus_mode"]
            with self.lock:
                if (self.camera_request != camera_request or self.focus_request is not focus_request
                        or self.stop_event.is_set()):
                    raise InterruptedError
                self.vision = result

        try:
            if mode == "auto":
                if not cap.set(cv2.CAP_PROP_AUTOFOCUS, 1):
                    raise ValueError("设备拒绝开启自动对焦")
                message = "已请求开启自动对焦，当前模式见相机读数。"
            else:
                state = set_manual_focus(cap, target, settle=.3, on_frame=preview)
                message = f"手动调焦已确认，回读 {state['focus_readback']:g}。"
        except InterruptedError:
            return
        except (ValueError, cv2.error) as error:
            message = f"调焦失败：{error}"
        with self.lock:
            if self.camera_request == camera_request and self.focus_request is focus_request:
                self.focus_request = None
                self.logs.put(message)

    def camera_loop(self):
        import cv2
        from calibration.camera_calibration import autofocus_mode
        from localization.focus_map import FocusStability

        while not self.stop_event.is_set():
            self.camera_wakeup.clear()
            with self.lock:
                request = self.camera_request
            if request is None:
                self.camera_wakeup.wait(.2)
                continue
            index, _ = request
            cap = None
            try:
                estimator = VisionEstimator(self.config)
                if not estimator.calibrated:
                    self.logs.put("未找到相机标定文件，使用默认内参；当前视觉结果仅供调试。")
                elif "calibration_file" in self.config:
                    info = getattr(estimator, "calibration_data", {})
                    rms = info.get("rms")
                    detail = f"，RMS={rms:.4f} px" if rms is not None else ""
                    source_file = self.config.get('focus_map_file') or self.config['calibration_file']
                    self.logs.put(f"已加载相机内参：{source_file}{detail}")
                with self.lock:
                    if self.camera_request != request or self.stop_event.is_set():
                        continue
                backend = self.config.get("camera_backend")
                if backend is None and self.config.get("focus_map_file"):
                    backend = "dshow"
                if backend:
                    cap = cv2.VideoCapture(index, {"dshow":cv2.CAP_DSHOW,"msmf":cv2.CAP_MSMF,"any":cv2.CAP_ANY}[backend])
                else:
                    cap = cv2.VideoCapture(index)
                if not cap.isOpened():
                    raise RuntimeError(f"无法打开摄像头 {index}")
                backend_name = cap.getBackendName()
                active_map = getattr(estimator, 'focus_map', None)
                if active_map is not None:
                    expected_backend = active_map.data.get('focus_control_backend')
                    if expected_backend and expected_backend != backend_name:
                        raise RuntimeError(f"焦点表基于 {expected_backend}，当前后端为 {backend_name}，请使用相同后端")
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                width, height = self.config["camera_resolution"]
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                cap.set(cv2.CAP_PROP_FPS, self.config["camera_fps"])
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                focus_tracker = None
                if self.config.get("focus_map_file"):
                    if not cap.set(cv2.CAP_PROP_AUTOFOCUS, 1):
                        self.logs.put("设备拒绝开启自动对焦，可在相机工具栏尝试手动调焦。")
                    focus_tracker = FocusStability()
                    self.logs.put("动态内参已启用：焦点回读连续3帧稳定后求解，档位之间线性插值，范围外线性外推。")
                with self.lock:
                    if self.camera_request != request or self.stop_event.is_set():
                        continue
                    self.camera_status = f"已连接相机 {index} ({width}×{height})"
                self.logs.put(f"相机已连接：{index}，{width}×{height}，{self.config['camera_fps']} FPS")
                while not self.stop_event.is_set():
                    with self.lock:
                        if self.camera_request != request:
                            break
                        focus_request = self.focus_request
                    if focus_request is not None:
                        self.apply_focus_request(cap, request, focus_request)
                        if focus_tracker is not None:
                            focus_tracker = FocusStability()
                        continue
                    focus_before = float(cap.get(cv2.CAP_PROP_FOCUS)) if focus_tracker else None
                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError("摄像头读取失败")
                    if frame.shape[:2] != (height, width):
                        raise RuntimeError(f"相机分辨率 {frame.shape[1]}x{frame.shape[0]} 与标定配置 {width}x{height} 不符")
                    focus_after = (float(cap.get(cv2.CAP_PROP_FOCUS))
                                   if focus_tracker or backend_name == "DSHOW" else None)
                    if focus_tracker:
                        try:
                            stable_focus = focus_tracker.update(focus_before, focus_after)
                        except ValueError as error:
                            result = self.focus_preview(frame, f"FOCUS: {error}")
                        else:
                            if stable_focus is None:
                                result = self.focus_preview(frame, "FOCUSING (waiting for stable focus)")
                            else:
                                result = estimator.process(frame, focus_position=stable_focus)
                    else:
                        result = estimator.process(frame)
                    if focus_after is not None and np.isfinite(focus_after) and focus_after >= 0:
                        result.focus_readback = focus_after
                    if backend_name == "DSHOW":
                        result.focus_mode = autofocus_mode(cap.get(cv2.CAP_PROP_AUTOFOCUS), backend_name)
                    with self.lock:
                        # Do not publish a frame/pose produced before a camera switch.
                        if self.camera_request != request or self.stop_event.is_set():
                            break
                        if self.focus_request is None:
                            self.vision = result
            except Exception as error:
                with self.lock:
                    if self.camera_request == request and not self.stop_event.is_set():
                        self.camera_request = None
                        self.focus_request = None
                        self.camera_status = f"相机 {index} 连接失败 / 已断开：{error}"
                        self.vision = VisionResult(status=f"CAMERA ERROR: {error}")
                        self.logs.put(f"视觉采集停止：{type(error).__name__}: {error}；可在窗口中重新连接。")
            finally:
                if cap is not None:
                    cap.release()

    def connect_serial(self, device, baud, *, usb_id=None):
        """Queue a connection change; only the serial worker opens/closes ports."""
        device, baud = str(device).strip(), int(baud)
        if not device or baud <= 0:
            raise ValueError("请选择串口并输入正整数波特率")
        if self.demo:
            return
        with self.serial_lock:
            self.serial_version += 1
            self.serial_request = (device, baud, self.serial_version, usb_id)
            self.serial_status = f"正在连接 {device} ({baud})…"
            self.sensors.disconnect()
        self.serial_wakeup.set()

    def disconnect_serial(self):
        if self.demo:
            return
        with self.serial_lock:
            self.serial_request = None
            self.serial_status = "未连接"
            self.sensors.disconnect()
        self.serial_wakeup.set()

    def get_serial_status(self):
        with self.serial_lock:
            return self.serial_status

    def serial_loop(self):
        silence_timeout = max(SERIAL_SILENCE_S, self.config.get("sensor_timeout_s", 1.0))
        while not self.stop_event.is_set():
            self.serial_wakeup.clear()
            with self.serial_lock:
                request = self.serial_request
            if request is None:
                self.serial_wakeup.wait(.2)
                continue
            device, baud, _, usb_id = request
            error_text = None
            retry = False
            try:
                import serial
                device = resolve_serial_device(device, usb_id)
                with serial.Serial(device, baud, timeout=.2) as port:
                    with self.serial_lock:
                        if self.serial_request != request or self.stop_event.is_set():
                            continue
                        self.serial_port = port
                        self.serial_status = f"已连接 {device} ({baud})"
                    self.logs.put(f"串口已连接：{device}，{baud} baud")
                    pending = b""
                    last_report = time.monotonic()
                    while not self.stop_event.is_set():
                        with self.serial_lock:
                            if self.serial_request != request:
                                break
                        chunk = port.read(min(port.in_waiting or 1, 4096))
                        now = time.monotonic()
                        with self.serial_lock:
                            # A reconnect must never consume bytes from the previous port.
                            if self.serial_request != request or self.stop_event.is_set():
                                break
                            *lines, pending = (pending + chunk).split(b"\n")
                            for line in lines:
                                if self.sensors.consume(line.decode("utf-8", errors="replace"), now):
                                    last_report = now
                            if len(pending) > 16384:
                                pending = b""
                                self.logs.put("丢弃过长串口行，请检查波特率和固件输出格式。")
                            if now - last_report >= silence_timeout:
                                raise TimeoutError(f"{silence_timeout:g} 秒未收到可识别的红外报文")
            except Exception as error:
                error_text = f"{device}: {type(error).__name__}: {error}"
            finally:
                with self.serial_lock:
                    self.serial_port = None
                    self.sensors.disconnect()
                    if error_text and self.serial_request == request and not self.stop_event.is_set():
                        status = f"红外重连中：{error_text}"
                        if self.serial_status != status:
                            self.logs.put(status)
                        self.serial_status = status
                        retry = True
            if retry:
                self.serial_wakeup.wait(SERIAL_RETRY_S)

    def demo_loop(self):
        import cv2
        start = time.monotonic()
        from datetime import datetime
        while not self.stop_event.is_set():
            t = time.monotonic() - start
            robot = np.eye(4)
            robot[:3, 3] = [.35 + .03 * np.sin(t * .4), .06, -.01]
            camera = robot @ self.config["T_R_C"]
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "DEMO - SIMULATED DATA", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, .85, (0, 220, 255), 2)
            cv2.rectangle(frame, (250, 180), (350, 280), (0, 255, 0), 2)
            cv2.putText(frame, "ID:33 (synthetic)", (220, 160), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 1)
            self.set_vision(VisionResult(frame=frame, camera_pose=camera, detected_ids=[33], known_ids=[33],
                                        timestamp=datetime.now().astimezone().isoformat(timespec="milliseconds"),
                                        monotonic=time.monotonic(), status="DEMO (simulated pose)"))
            for i in SENSOR_IDS:
                distance_m = (.16 + .015 * np.sin(t * .5) - SENSOR_POSITIONS[i - 1, 2]) / SENSOR_DIRECTIONS[i - 1, 2]
                scale, offset = self.sensors.calibration[i]
                self.sensors.consume(f"S{i}: VALID | distance={round((distance_m * 1000 - offset) / scale)} mm")
            self.stop_event.wait(.1)

    def stop(self):
        self.stop_event.set()
        self.disconnect_serial()
        self.serial_wakeup.set()
        self.disconnect_camera()
        self.camera_wakeup.set()
        port = self.serial_port
        if port is not None:
            try:
                port.cancel_read()
            except (AttributeError, OSError):
                pass
        for thread in self.threads:
            thread.join(timeout=1.)


def make_window(config, source, demo):
    # Import GUI dependencies only after parsing --help / --list-ports.
    from PyQt5 import QtCore, QtGui, QtWidgets
    import cv2
    import pyvista as pv
    from pyvistaqt import QtInteractor

    class Window(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("机翼内软体机器人定位 · AprilTag + 红外" + (" [模拟演示]" if demo else ""))
            self.resize(1500, 920)
            self.paused = False
            self.snapshot = None
            self.saving = False
            self.scene_signature = None
            self.dynamic_names = set()
            self.last_frame = None
            self.robot_actors = []
            self.world_mode = None
            self.tabs = QtWidgets.QTabWidget()

            panel = QtWidgets.QWidget()
            outer = QtWidgets.QVBoxLayout(panel)
            banner = ("模拟数据：图像、位姿及红外读数均为演示数据。" if demo else
                      "T_R_C 未配置：机器人及红外在局部坐标页显示；机翼页等待外参。" if config["T_R_C"] is None else
                      "已加载相机外参。请确认 STL 对齐矩阵与实际装配一致。")
            if not config["intrinsics_calibrated"] and not demo:
                banner += " 相机内参未标定，视觉结果仅供调试。"
            self.banner = QtWidgets.QLabel(banner)
            self.banner.setWordWrap(True)
            self.banner.setStyleSheet("background:#fff2ce; color:#624500; padding:8px")
            outer.addWidget(self.banner)
            splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
            outer.addWidget(splitter, 1)
            left = QtWidgets.QWidget()
            left_layout = QtWidgets.QVBoxLayout(left)
            self.image = QtWidgets.QLabel("等待相机画面…")
            self.image.setAlignment(QtCore.Qt.AlignCenter)
            self.image.setMinimumSize(400, 360)
            self.image.setMaximumHeight(480)
            self.image.setStyleSheet("background:#151b25;color:#ddd")
            left_layout.addWidget(self.image, 4)
            self.focus_readout = QtWidgets.QLabel("对焦读数 --  |  fx -- px  |  fy -- px")
            self.focus_readout.setToolTip("对焦读数来自相机驱动。fx、fy 是当前画面用于定位的有效焦距，单位为像素。")
            left_layout.addWidget(self.focus_readout)
            self.info = QtWidgets.QPlainTextEdit()
            self.info.setReadOnly(True)
            self.info.setFont(QtGui.QFont("Consolas", 10))
            left_layout.addWidget(self.info, 3)
            self.log = QtWidgets.QPlainTextEdit()
            self.log.setReadOnly(True)
            self.log.setMaximumBlockCount(120)
            self.log.setMaximumHeight(80)
            left_layout.addWidget(self.log)
            splitter.addWidget(left)

            # Both views are embedded in the same desktop window. The local page
            # remains useful before extrinsic calibration or during visual loss.
            self.world_plot = QtInteractor(self)
            self.local_plot = QtInteractor(self)
            self.tabs.addTab(self.world_plot.interactor, "机翼世界坐标 W")
            self.tabs.addTab(self.local_plot.interactor, "机器人局部坐标 R / 红外")
            splitter.addWidget(self.tabs)
            splitter.setSizes([660, 840])
            self.setCentralWidget(panel)

            toolbar = self.addToolBar("操作")
            for label, shortcut, callback in (("保存数据 (S)", "S", self.save_sample),
                                               ("暂停 / 继续 (Space)", "Space", self.toggle_pause),
                                               ("重置视角 (R)", "R", self.reset_views),
                                               ("退出 (Q)", "Q", self.close)):
                action = QtWidgets.QAction(label, self)
                action.setShortcut(shortcut)
                action.setShortcutContext(QtCore.Qt.ApplicationShortcut)
                action.triggered.connect(callback)
                toolbar.addAction(action)
            self.statusBar().showMessage(f"CSV: {config['csv_file']}")
            self.setup_serial_controls(QtWidgets)
            self.setup_camera_controls(QtWidgets)
            self.setup_scene(pv)
            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self.refresh)
            self.timer.start(100)
            source.start()

        def setup_serial_controls(self, widgets):
            self.addToolBarBreak()
            toolbar = self.addToolBar("红外串口")
            toolbar.addWidget(widgets.QLabel(" 红外串口："))
            self.serial_combo = widgets.QComboBox()
            self.serial_combo.setEditable(True)
            self.serial_combo.setMinimumWidth(260)
            self.serial_combo.setToolTip("选择设备或直接输入串口名，例如 COM4")
            toolbar.addWidget(self.serial_combo)
            toolbar.addWidget(widgets.QLabel(" 波特率："))
            self.baud_combo = widgets.QComboBox()
            self.baud_combo.setEditable(True)
            self.baud_combo.addItems(["9600", "19200", "38400", "57600", "115200", "230400"])
            self.baud_combo.setCurrentText(str(config["serial_baud"]))
            toolbar.addWidget(self.baud_combo)
            self.serial_refresh_action = toolbar.addAction("刷新串口", self.refresh_serial_ports)
            self.serial_connect_action = toolbar.addAction("连接 / 重连", self.connect_selected_serial)
            self.serial_disconnect_action = toolbar.addAction("断开", source.disconnect_serial)
            self.serial_status_label = widgets.QLabel(" " + source.get_serial_status())
            self.serial_status_label.setMaximumWidth(500)
            toolbar.addWidget(self.serial_status_label)
            for control in (self.serial_combo, self.baud_combo, self.serial_refresh_action,
                            self.serial_connect_action, self.serial_disconnect_action):
                control.setEnabled(not demo)
            if not demo:
                self.refresh_serial_ports()
            else:
                self.serial_combo.addItem(str(config["serial_port"]))

        def selected_serial_device(self):
            return self.serial_combo.currentText().split(" — ", 1)[0].strip()

        def refresh_serial_ports(self):
            selected = self.selected_serial_device() or str(config["serial_port"])
            try:
                from serial.tools.list_ports import comports
                ports = sorted(comports(), key=lambda port: port.device)
            except Exception as error:
                self.log.appendPlainText(f"串口扫描失败: {error}")
                return
            self.serial_combo.clear()
            for port in ports:
                self.serial_combo.addItem(f"{port.device} — {port.description}", port.device)
            index = self.serial_combo.findData(selected)
            if index < 0:
                self.serial_combo.addItem(selected, selected)
                index = self.serial_combo.count() - 1
            self.serial_combo.setCurrentIndex(index)
            self.log.appendPlainText(f"已扫描到 {len(ports)} 个串口。" if ports else
                                     "未检测到串口，请连接红外设备后刷新；也可手动输入端口名。")

        def connect_selected_serial(self):
            try:
                source.connect_serial(self.selected_serial_device(), int(self.baud_combo.currentText()))
            except (TypeError, ValueError):
                self.log.appendPlainText("串口参数无效：请选择串口并输入正整数波特率。")

        def setup_camera_controls(self, widgets):
            self.addToolBarBreak()
            toolbar = self.addToolBar("相机")
            toolbar.addWidget(widgets.QLabel(" 相机编号："))
            self.camera_combo = widgets.QComboBox()
            self.camera_combo.setEditable(True)
            self.camera_combo.addItems([str(index) for index in range(10)])
            self.camera_combo.setCurrentText(str(config["camera_index"]))
            self.camera_combo.setToolTip("OpenCV 相机编号，从 0 开始；可手动输入其他非负整数编号")
            self.camera_combo.setMinimumWidth(100)
            toolbar.addWidget(self.camera_combo)
            self.camera_connect_action = toolbar.addAction("连接 / 重连相机", self.connect_selected_camera)
            self.camera_disconnect_action = toolbar.addAction("断开相机", source.disconnect_camera)
            self.focus_input = widgets.QLineEdit()
            self.focus_input.setPlaceholderText("对焦读数")
            self.focus_input.setToolTip("输入相机驱动支持的整数对焦读数")
            self.focus_input.setValidator(QtGui.QIntValidator(0, 2147483647, self.focus_input))
            self.focus_input.setMaximumWidth(96)
            self.focus_input.returnPressed.connect(self.request_manual_focus)
            toolbar.addWidget(self.focus_input)
            self.manual_focus_action = toolbar.addAction("手动调焦", self.request_manual_focus)
            self.auto_focus_action = toolbar.addAction("自动调焦", self.request_auto_focus)
            self.focus_controls = (self.focus_input, self.manual_focus_action, self.auto_focus_action)
            for control in self.focus_controls:
                control.setEnabled(False)
            self.camera_status_label = widgets.QLabel(" " + source.get_camera_status())
            self.camera_status_label.setMaximumWidth(600)
            toolbar.addWidget(self.camera_status_label)
            for control in (self.camera_combo, self.camera_connect_action, self.camera_disconnect_action):
                control.setEnabled(not demo)

        def connect_selected_camera(self):
            try:
                source.connect_camera(self.camera_combo.currentText())
            except (TypeError, ValueError):
                self.log.appendPlainText("相机编号无效：请输入从 0 开始的非负整数编号。")

        def request_manual_focus(self):
            if not self.focus_input.hasAcceptableInput():
                self.log.appendPlainText("请先输入相机驱动支持的整数对焦读数。")
                return
            try:
                source.request_camera_focus(self.focus_input.text())
            except ValueError as error:
                self.log.appendPlainText(str(error))

        def request_auto_focus(self):
            try:
                source.request_camera_focus()
            except ValueError as error:
                self.log.appendPlainText(str(error))

        def setup_scene(self, pv):
            for plot in (self.world_plot, self.local_plot):
                plot.set_background("#192331")
                plot.add_axes()
                plot.enable_depth_peeling()
            wing = pv.read(str(config["wing_mesh"]))
            wing.points *= config["wing_mesh_scale"]
            wing.transform(config["T_W_wing_mesh"], inplace=True)
            self.world_plot.add_mesh(wing, name="wing", color="lightsteelblue", opacity=config["wing_opacity"])
            self.world_plot.add_text("World W (m)", font_size=11, color="white", name="world_title")
            self.setup_tag_frames(pv)
            robot = pv.read(str(config["robot_mesh"]))
            robot.points *= config["robot_mesh_scale"]
            robot.transform(config["T_R_robot_mesh"], inplace=True)
            for plot in (self.world_plot, self.local_plot):
                actors = [plot.add_mesh(robot, color="silver", name="robot", render=False)]
                # Retain the original front plate outline and origin.
                plate = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=.026, j_size=.034,
                                 i_resolution=1, j_resolution=1)
                actors.append(plot.add_mesh(plate, color="gray", opacity=.6, render=False))
                actors.append(plot.add_mesh(pv.Sphere(radius=.002, center=(0, 0, 0)), color="white", render=False))
                for axis, color in zip(np.eye(3), ("red", "lime", "deepskyblue")):
                    actors.append(plot.add_mesh(pv.Arrow(start=(0, 0, 0), direction=axis, scale=.025), color=color, render=False))
                for pos, direction, color in zip(SENSOR_POSITIONS, SENSOR_DIRECTIONS, ("tomato", "lime", "dodgerblue")):
                    actors.append(plot.add_mesh(pv.Sphere(radius=.0017, center=pos), color=color, render=False))
                    actors.append(plot.add_mesh(pv.Arrow(start=pos, direction=direction, scale=.0115), color=color, render=False))
                if plot is self.world_plot:
                    self.robot_actors = actors
                    for actor in actors:
                        actor.visibility = False
            self.local_plot.add_text("Robot R (m): X right, Y down, Z forward", font_size=10, color="white")
            self.local_plot.add_mesh(pv.Line((-.06, -.04, 0), (.06, .04, .25)), opacity=0, name="local_bounds")
            self.reset_views()

        def setup_tag_frames(self, pv):
            self.tag_actors = {}
            self.tag_labels_actor = None
            try:
                self.tag_layout = load_tag_layout(config["tag_layout_file"])
            except (OSError, ValueError, TypeError) as error:
                self.tag_layout = {}
                self.log.appendPlainText(f"Tag 坐标系加载失败: {error}")
                return
            # Reuse local arrow meshes; T_W_Tag places each frame in world W.
            length = config["tag_size_m"] * 1.5
            axes = [(name, pv.Arrow(direction=axis, scale=length), color)
                    for name, axis, color in zip("xyz", np.eye(3), ("red", "lime", "deepskyblue"))]
            for tag_id, pose in self.tag_layout.items():
                actors = []
                for name, mesh, color in axes:
                    actor = self.world_plot.add_mesh(mesh, name=f"tag_{tag_id}_{name}", color=color,
                                                     lighting=False, reset_camera=False, render=False, pickable=False)
                    actor.user_matrix = pose
                    actors.append(actor)
                self.tag_actors[tag_id] = actors
            if self.tag_layout:
                label_points = np.array([pose[:3, 3] + pose[:3, 2] * length * 1.3
                                         for pose in self.tag_layout.values()])
                self.tag_labels_actor = self.world_plot.add_point_labels(
                    label_points, [f"Tag {tag_id}" for tag_id in self.tag_layout], name="tag_ids",
                    font_size=11, text_color="white", shape=None, shadow=True, show_points=False,
                    always_visible=True, reset_camera=False, render=False, pickable=False)
                self.world_plot.add_text("Tag axes: X red / Y green / Z blue", position="lower_left",
                                         font_size=9, color="white", name="tag_axes_legend")

        def reset_views(self):
            for plot in (self.world_plot, self.local_plot):
                plot.view_isometric()
                plot.reset_camera()

        def toggle_pause(self):
            self.paused = not self.paused
            self.statusBar().showMessage("已暂停，S 保存当前冻结数据" if self.paused else "采集中")

        def refresh(self):
            while True:
                try:
                    self.log.appendPlainText(source.logs.get_nowait())
                except queue.Empty:
                    break
            serial_status = source.get_serial_status()
            self.serial_status_label.setText(" " + serial_status)
            self.serial_status_label.setToolTip(serial_status)
            camera_status = source.get_camera_status()
            self.camera_status_label.setText(" " + camera_status)
            self.camera_status_label.setToolTip(camera_status)
            focus_enabled = source.can_request_focus()
            for control in self.focus_controls:
                control.setEnabled(focus_enabled)
            if self.paused:
                return
            now = time.monotonic()
            vision = source.get_vision()
            readings = source.sensors.snapshot(config["sensor_timeout_s"], now)
            plane = fit_plane(readings, config["sensor_max_skew_s"])
            camera = vision.camera_pose if now - vision.monotonic <= config["vision_timeout_s"] else None
            robot = robot_pose(camera, config["T_R_C"])
            row = record_snapshot(vision, readings, plane, config, demo, now)
            if vision.timestamp and camera is None and vision.camera_pose is not None:
                row["vision_status"] = "STALE"
            self.snapshot = row
            focus = vision.focus_readback if vision.focus_readback is not None else vision.focus_position
            focus_text = "--" if focus is None else f"{focus:g}"
            mode = {"AUTO": "自动对焦", "MANUAL": "手动调焦"}.get(vision.focus_mode, "模式未知")
            fx = f"{row['camera_fx_px']:.3f}" if row["camera_fx_px"] != "" else "--"
            fy = f"{row['camera_fy_px']:.3f}" if row["camera_fy_px"] != "" else "--"
            self.focus_readout.setText(f"{mode}  |  对焦读数 {focus_text}  |  fx {fx} px  |  fy {fy} px")
            if vision.frame is not None:
                self.last_frame = vision.frame
                rgb = cv2.cvtColor(vision.frame, cv2.COLOR_BGR2RGB)
                height, width = rgb.shape[:2]
                image = QtGui.QImage(rgb.data, width, height, rgb.strides[0], QtGui.QImage.Format_RGB888).copy()
                self.image.setPixmap(QtGui.QPixmap.fromImage(image).scaled(
                    self.image.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
            elif self.last_frame is not None:
                self.image.clear()
                self.image.setText(vision.status)
                self.last_frame = None
            else:
                self.image.setText(vision.status)
            matrix_text = "--（缺少视觉位姿或 T_R_C 外参）" if robot is None else np.array2string(
                robot, precision=6, suppress_small=True, formatter={"float_kind": lambda x: f"{x: .6f}"})
            sensor_text = []
            for i, reading in readings.items():
                raw = "--" if reading.raw_mm is None else f"{reading.raw_mm:.0f}"
                calibrated = "--" if reading.calibrated_mm is None else f"{reading.calibrated_mm:.2f}"
                sensor_text.append(f"S{i}: raw={raw}, calibrated={calibrated} mm [{reading.status}]")
            plane_text = ("--（需三路有效、未超时且时间接近的读数）" if plane is None else
                          f"距离 = {plane.distance * 1000:.3f} mm\n"
                          f"n_R = ({plane.normal[0]:.6f}, {plane.normal[1]:.6f}, {plane.normal[2]:.6f})\n"
                          f"n_R · X_R + {plane.offset:.6f} = 0 (m)")
            camera_text = "--" if camera is None else np.array2string(camera[:3, 3], precision=5)
            self.info.setPlainText(
                f"视觉: {row['vision_status']}\n已知 Tag: {vision.known_ids} | 内点: {len(vision.inlier_indices)}\n"
                f"相机位置 W (m): {camera_text}\n\n机器人位姿 T_W_R（平移单位 m）:\n{matrix_text}\n\n"
                + "\n".join(sensor_text) + f"\n\n红外平面（机器人前端盖坐标 R）:\n{plane_text}\n"
                "法向量朝向机器人原点。")
            signature = (None if robot is None else robot.tobytes(),
                         tuple((r.status, r.calibrated_mm) for r in readings.values()), plane is not None)
            if signature != self.scene_signature:
                try:
                    self.update_scene(pv, robot, readings, plane)
                    self.scene_signature = signature
                except Exception as error:
                    self.log.appendPlainText(f"三维更新失败: {error}")

        def draw_measurements(self, pv, plot, transform, readings, plane):
            suffix = "world" if plot is self.world_plot else "local"
            for name in list(self.dynamic_names):
                if name.endswith(suffix):
                    plot.remove_actor(name, reset_camera=False, render=False)
                    self.dynamic_names.remove(name)

            def add(mesh, name, **kwargs):
                name = name + "_" + suffix
                plot.add_mesh(mesh, name=name, reset_camera=False, render=False, **kwargs)
                self.dynamic_names.add(name)

            points = measured_points(readings)
            for i, point in points.items():
                start, end = transform_points(transform, [SENSOR_POSITIONS[i - 1], point])
                color = ("tomato", "lime", "dodgerblue")[i - 1]
                add(pv.Line(start, end), f"ray{i}", color=color, line_width=2)
                add(pv.Sphere(radius=.0025, center=end), f"point{i}", color=color)
            if plane is not None:
                points = transform_points(transform, plane.points)
                add(pv.PolyData(points, faces=[3, 0, 1, 2]), "plane", color="gold", opacity=.5)
                origin, projection, center = transform_points(transform, [[0, 0, 0], plane.projection, plane.center])
                normal = transform[:3, :3] @ plane.normal
                add(pv.Arrow(start=center, direction=normal, scale=.059), "normal", color="yellow")
                if plane.distance > 1e-9:
                    add(pv.Line(origin, projection), "distance", color="white", line_width=3)

        def update_scene(self, pv, robot, readings, plane):
            for actor in self.robot_actors:
                actor.visibility = robot is not None
                if robot is not None:
                    actor.user_matrix = robot
            self.draw_measurements(pv, self.local_plot, np.eye(4), readings, plane)
            if robot is not None:
                self.draw_measurements(pv, self.world_plot, robot, readings, plane)
            else:
                for name in list(self.dynamic_names):
                    if name.endswith("world"):
                        self.world_plot.remove_actor(name, reset_camera=False, render=False)
                        self.dynamic_names.remove(name)
            mode = "localized" if robot is not None else "unlocalized"
            if mode != self.world_mode:
                self.world_plot.add_text("World W (m)" if robot is not None else "World W (m) - awaiting robot pose / T_R_C",
                                         font_size=11, color="white", name="world_title")
                self.world_mode = mode
            self.world_plot.render()
            self.local_plot.render()

        def save_sample(self):
            if self.saving:
                return
            # Freeze precisely the sample visible when S is pressed. Live capture
            # continues during the modal dialog, without changing this row/frame.
            if self.snapshot is None:
                self.refresh()
            if self.snapshot is None:
                return
            self.saving = True
            row = dict(self.snapshot)
            frame = None if self.last_frame is None else self.last_frame.copy()
            captured_at = datetime.now().astimezone()
            try:
                while True:
                    group, ok = QtWidgets.QInputDialog.getText(self, "保存联合数据", "数据组编号（保存按 S 时的画面与读数）：")
                    if not ok:
                        return
                    if group.strip():
                        break
                    QtWidgets.QMessageBox.information(self, "编号为空", "请输入非空的数据组编号。")
                try:
                    path, image_path = save_record(config["csv_file"], config["snapshot_dir"],
                                                   group, row, frame, captured_at)
                    self.log.appendPlainText(f"已保存组 {group.strip()} → {path}")
                    if image_path is not None:
                        self.log.appendPlainText(f"相机快照 → {image_path}")
                    else:
                        self.log.appendPlainText("当前无相机画面，仅保存 CSV 数据。")
                    self.statusBar().showMessage(f"已保存组 {group.strip()}（CSV + 相机快照）" if image_path is not None else
                                                 f"已保存组 {group.strip()}（仅 CSV：当前无相机画面）")
                except (OSError, ValueError) as error:
                    QtWidgets.QMessageBox.critical(self, "保存失败", f"{error}\n请检查文件是否被占用及 CSV 表头。")
            finally:
                self.saving = False

        def closeEvent(self, event):
            self.timer.stop()
            source.stop()
            self.world_plot.close()
            self.local_plot.close()
            event.accept()

    return Window()


def main(argv=None):
    parser = argparse.ArgumentParser(description="机翼内 AprilTag + 三红外联合定位")
    parser.add_argument("--config", type=Path, default=ROOT / "config/positioning_config.json")
    parser.add_argument("--camera", type=int, help="覆盖摄像头编号")
    parser.add_argument("--camera-backend", choices=["dshow","msmf","any"], help="指定相机采集后端")
    parser.add_argument("--focus-map", type=Path, help="覆盖配置中的动态焦点内参表")
    parser.add_argument("--port", help="覆盖串口，例如 COM4")
    parser.add_argument("--baud", type=int, help="覆盖串口波特率")
    parser.add_argument("--csv", type=Path, help="覆盖 CSV 路径（相对于当前工作目录）")
    parser.add_argument("--list-ports", action="store_true")
    parser.add_argument("--demo", action="store_true", help="无需相机或串口的模拟演示")
    args = parser.parse_args(argv)
    if args.list_ports:
        try:
            from serial.tools.list_ports import comports
            ports = list(comports())
            for port in ports:
                print(f"{port.device}: {port.description}")
            if not ports:
                print("未检测到串口。")
            return 0
        except ImportError:
            print("请安装 pyserial: python -m pip install -r requirements.txt", file=sys.stderr)
            return 1
    try:
        config = load_config(args.config)
        if args.focus_map:
            config["focus_map_file"] = args.focus_map.resolve()
        if args.camera_backend:
            config["camera_backend"] = args.camera_backend
        for argument, key in ((args.camera, "camera_index"), (args.port, "serial_port"), (args.baud, "serial_baud")):
            if argument is not None:
                config[key] = argument
        if args.port is not None:
            config.pop("serial_usb_id", None)
        if args.csv:
            config["csv_file"] = args.csv.resolve()
        calibration_file = config.get("focus_map_file") or config["calibration_file"]
        config["intrinsics_calibrated"] = calibration_file.exists() and not args.demo
        if args.demo and config["T_R_C"] is None:
            config["T_R_C"] = np.eye(4)
        from PyQt5 import QtWidgets
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
        source = Acquisition(config, args.demo)
        window = make_window(config, source, args.demo)
        window.show()
        try:
            return app.exec_()
        finally:
            source.stop()
    except Exception as error:
        print(f"启动失败: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
