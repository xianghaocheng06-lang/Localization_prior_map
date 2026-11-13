"""Actual Qt/VTK integration check: python tests/gui_smoke.py.

Uses synthetic data only. Saves a screenshot in .verification/ for inspection.
"""
import csv
import json
import sys
import tempfile
import traceback
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PyQt5 import QtCore, QtWidgets
from localization.positioning import Acquisition, make_window, ROOT
from localization.positioning_core import load_config
from localization.positioning_core import VisionResult


def main():
    app = QtWidgets.QApplication(sys.argv[:1])
    config = load_config(ROOT / "config/positioning_config.json")
    config["T_R_C"] = np.eye(4)
    config["intrinsics_calibrated"] = False
    errors = []
    with tempfile.TemporaryDirectory() as directory:
        config["csv_file"] = Path(directory) / "smoke.csv"
        config["snapshot_dir"] = Path(directory) / "相机快照"
        source = Acquisition(config, demo=True)
        window = make_window(config, source, demo=True)
        window.show()

        def verify():
            try:
                window.refresh()
                assert window.snapshot["plane_valid"], window.info.toPlainText()
                assert window.snapshot["T_W_R_00"] == 1.
                assert all(actor.visibility for actor in window.robot_actors)
                assert "plane_world" in window.dynamic_names
                layout = json.loads(config["tag_layout_file"].read_text(encoding="utf-8"))
                assert set(window.tag_actors) == set(layout)
                assert window.tag_labels_actor is not None and window.tag_labels_actor.visibility
                for tag_id, actors in window.tag_actors.items():
                    assert len(actors) == 3 and all(actor.visibility for actor in actors)
                    for actor in actors:
                        np.testing.assert_allclose(actor.user_matrix, layout[tag_id])
                window.toggle_pause()
                frozen = dict(window.snapshot)
                window.refresh()
                assert window.snapshot == frozen
                frozen_frame = window.last_frame.copy()
                original_dialog = QtWidgets.QInputDialog.getText
                def enter_group(*args, **kwargs):
                    # Even an in-place frame change while entering the group ID
                    # must not affect the image frozen by the S shortcut.
                    window.last_frame[:] = 0
                    return "GUI组01", True
                QtWidgets.QInputDialog.getText = staticmethod(enter_group)
                try:
                    # Trigger the real S action even when the VTK widget has focus.
                    from PyQt5.QtTest import QTest
                    window.activateWindow()
                    window.world_plot.interactor.setFocus()
                    QTest.qWait(100)
                    QTest.keyClick(window.world_plot.interactor, QtCore.Qt.Key_S)
                    app.processEvents()
                finally:
                    QtWidgets.QInputDialog.getText = original_dialog
                with config["csv_file"].open(encoding="utf-8-sig", newline="") as stream:
                    row = next(csv.DictReader(stream))
                assert row["group_id"] == "GUI组01"
                assert row["snapshot_at"] == frozen["snapshot_at"]
                assert row["demo"] == "True"
                import cv2
                images = list(config["snapshot_dir"].glob("*.png"))
                assert len(images) == 1
                assert images[0].name.startswith("GUI组01_")
                np.testing.assert_array_equal(
                    cv2.imdecode(np.fromfile(images[0], dtype=np.uint8), cv2.IMREAD_COLOR), frozen_frame)
                QtWidgets.QInputDialog.getText = staticmethod(lambda *args, **kwargs: ("", False))
                try:
                    window.save_sample()
                finally:
                    QtWidgets.QInputDialog.getText = original_dialog
                assert len(list(config["snapshot_dir"].glob("*.png"))) == 1
                with config["csv_file"].open(encoding="utf-8-sig", newline="") as stream:
                    assert len(list(csv.DictReader(stream))) == 1
                output = ROOT / ".verification"
                output.mkdir(exist_ok=True)
                window.grab().save(str(output / "demo_window.png"))
                # Visual localization loss must remove all world robot/IR actors.
                window.update_scene(__import__("pyvista"), None, source.sensors.snapshot(1), None)
                assert all(not actor.visibility for actor in window.robot_actors)
                assert not any(name.endswith("world") for name in window.dynamic_names)
                assert all(actor.visibility for actors in window.tag_actors.values() for actor in actors)
                assert window.tag_labels_actor.visibility
                # Unknown extrinsics still allow the local infrared plane.
                source.stop()  # Avoid changing worker configuration while it is in use.
                config["T_R_C"] = None
                window.paused = False
                window.refresh()
                assert window.snapshot["T_W_R_00"] == ""
                assert "plane_local" in window.dynamic_names
                assert not window.serial_connect_action.isEnabled()
                assert not window.camera_connect_action.isEnabled()
                assert not window.manual_focus_action.isEnabled()
                # Exercise real controls with enumerated fixtures; no hardware threads.
                controls_source = Acquisition(config)
                controls_source.start = lambda: None
                ports = [SimpleNamespace(device="COM4", description="USB Serial A"),
                         SimpleNamespace(device="COM8", description="Infrared USB")]
                with patch("serial.tools.list_ports.comports", return_value=ports):
                    controls = make_window(config, controls_source, demo=False)
                    try:
                        controls.show()
                        assert controls.serial_combo.findData("COM8") >= 0
                        controls.serial_combo.setCurrentIndex(controls.serial_combo.findData("COM8"))
                        controls.baud_combo.setCurrentText("57600")
                        controls.serial_connect_action.trigger()
                        assert controls_source.serial_request[:2] == ("COM8", 57600)
                        controls.paused = True
                        controls.refresh()
                        assert "COM8" in controls.serial_status_label.text()
                        controls.serial_refresh_action.trigger()
                        assert controls.selected_serial_device() == "COM8"
                        controls.serial_disconnect_action.trigger()
                        assert controls_source.serial_request is None
                        controls.serial_combo.setEditText("COM12")
                        controls.serial_connect_action.trigger()
                        assert controls_source.serial_request[:2] == ("COM12", 57600)
                        controls.baud_combo.setCurrentText("bad")
                        controls.serial_connect_action.trigger()
                        assert "串口参数无效" in controls.log.toPlainText()
                        controls.baud_combo.setCurrentText("57600")
                        controls.serial_disconnect_action.trigger()
                        controls.refresh()
                        # Camera choices queue changes independently of the serial port.
                        controls_source.set_vision(VisionResult(frame=np.zeros((2, 3, 3), dtype=np.uint8),
                                                               camera_pose=np.eye(4), status="VALID"))
                        controls.camera_combo.setCurrentText("3")
                        controls.camera_connect_action.trigger()
                        assert controls_source.camera_request[0] == 3
                        assert controls_source.get_vision().camera_pose is None
                        assert controls_source.get_vision().frame is None
                        controls.refresh()
                        assert "相机 3" in controls.camera_status_label.text()
                        assert controls.manual_focus_action.isEnabled()
                        controls.focus_input.setText("304")
                        controls.manual_focus_action.trigger()
                        assert controls_source.focus_request == ("manual", 304)
                        controls.auto_focus_action.trigger()
                        assert controls_source.focus_request == ("auto", None)
                        controls_source.set_vision(VisionResult(
                            focus_position=304, focus_readback=304, focus_mode="MANUAL",
                            camera_matrix=[[540.42, 0, 318], [0, 537.40, 224], [0, 0, 1]]))
                        controls.paused = False
                        controls.refresh()
                        assert "手动调焦" in controls.focus_readout.text()
                        assert "对焦读数 304" in controls.focus_readout.text()
                        assert "fx 540.420 px" in controls.focus_readout.text()
                        controls.camera_combo.setCurrentText("12")
                        controls.camera_connect_action.trigger()
                        assert controls_source.camera_request[0] == 12
                        controls.camera_combo.setCurrentText("-1")
                        controls.camera_connect_action.trigger()
                        assert controls_source.camera_request[0] == 12
                        assert "相机编号无效" in controls.log.toPlainText()
                        controls.camera_combo.setCurrentText("3")
                        controls.camera_disconnect_action.trigger()
                        assert controls_source.camera_request is None
                        controls.refresh()
                        assert not controls.auto_focus_action.isEnabled()
                        app.processEvents()
                        controls.grab().save(str(output / "serial_controls.png"))
                        controls.grab().save(str(output / "device_controls.png"))
                    finally:
                        controls.close()
                print("GUI PASS: demo, all Tag frames, snapshots, pose loss, local infrared, serial and camera selection/reconnection")
            except BaseException:
                errors.append(traceback.format_exc())
            finally:
                window.close()
                app.quit()

        QtCore.QTimer.singleShot(3000, verify)
        app.exec_()
        source.stop()
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
