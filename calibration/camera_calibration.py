"""Printable checkerboard, camera capture, pinhole calibration and validation.

Run ``python camera_calibration.py --help``; see calibration/README.md.
Image coordinates always refer to the full, uncropped, oriented image.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
import struct
import sys
import time
import zlib

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-9)


def autofocus_mode(raw, backend):
    """Decode DSHOW CameraControlFlags separately from boolean backends."""
    if raw is None or not np.isfinite(raw):
        return "UNKNOWN"
    if str(backend).upper() == "DSHOW":
        if raw < 0 or float(raw) != int(raw):
            return "UNKNOWN"
        return {1: "AUTO", 2: "MANUAL"}.get(int(raw) & 3, "UNKNOWN")
    return {0.: "MANUAL", 1.: "AUTO"}.get(raw, "UNKNOWN")


def verify_focus(cap, target):
    raw = float(cap.get(cv2.CAP_PROP_AUTOFOCUS))
    actual = float(cap.get(cv2.CAP_PROP_FOCUS))
    mode = autofocus_mode(raw, cap.getBackendName())
    if mode != "MANUAL" or not np.isfinite(actual) or abs(actual - target) > 1e-6:
        raise ValueError(f"固定焦点核验失败：请求 {target:g}，回读 {actual}，模式 {mode} (raw={raw})；停止采集，避免混入不同状态")
    return {"focus_readback": actual, "autofocus_readback": raw, "autofocus_mode": mode}


class FocusControlError(ValueError):
    """A focus request could not be verified; never capture under the requested label."""


def set_manual_focus(cap, target, settle=1.5, timeout=2.5, retries=3, on_frame=None, on_event=None):
    """Separate AF shutdown from the focus write, then verify delayed readback.

    Some devices reset/ignore the first lens write while leaving AF. Rewriting
    AF=off at every bracket also risks resetting an already manual lens.
    """
    def state():
        raw = float(cap.get(cv2.CAP_PROP_AUTOFOCUS))
        q = float(cap.get(cv2.CAP_PROP_FOCUS))
        return {'focus_readback': q if np.isfinite(q) else None,
                'autofocus_readback': raw if np.isfinite(raw) else None,
                'autofocus_mode': autofocus_mode(raw, cap.getBackendName())}

    def event(action, **extra):
        if on_event:
            on_event({'time': datetime.now().astimezone().isoformat(), 'action': action,
                      'target': target, 'readback': state(), **extra})

    def read(phase):
        ok, image = cap.read()
        if not ok:
            raise ValueError('等待对焦控制期间相机读帧失败')
        actual = state()
        if on_frame:
            on_frame(image, phase, actual)
        return actual

    initial = state()
    event('begin')
    changed_mode = initial['autofocus_mode'] != 'MANUAL'
    if changed_mode:
        accepted = bool(cap.set(cv2.CAP_PROP_AUTOFOCUS, 0))
        event('disable_autofocus', accepted=accepted)
        if not accepted:
            raise FocusControlError('设备拒绝关闭自动对焦')
    started, manual_frames = time.monotonic(), 0
    while True:
        actual = read('waiting for MANUAL')
        manual_frames = manual_frames+1 if actual['autofocus_mode']=='MANUAL' else 0
        elapsed = time.monotonic()-started
        if manual_frames >= 3 and (not changed_mode or elapsed >= .5):
            break
        if elapsed > max(timeout, .5):
            event('manual_timeout')
            raise FocusControlError(f'手动模式未稳定：{actual}')

    for attempt in range(1,retries+1):
        accepted = bool(cap.set(cv2.CAP_PROP_FOCUS,target))
        event('write_focus',attempt=attempt,accepted=accepted)
        if not accepted:
            raise FocusControlError(f'设备拒绝焦点 {target:g}；请核对范围/步长')
        started, stable_since, stable_frames = time.monotonic(), None, 0
        while True:
            actual = read(f'focus request attempt {attempt}/{retries}')
            now = time.monotonic()
            q = actual['focus_readback']
            matching = actual['autofocus_mode']=='MANUAL' and q is not None and abs(q-target)<=1e-6
            if matching:
                stable_since = now if stable_since is None else stable_since
                stable_frames += 1
                if stable_frames >= 3 and now-stable_since >= settle:
                    event('verified',attempt=attempt)
                    return verify_focus(cap,target)
            else:
                stable_since, stable_frames = None, 0
            if now-started >= settle+timeout:
                event('focus_timeout',attempt=attempt)
                break
    raise FocusControlError(f'焦点写入重试{retries}次仍未到位：请求 {target:g}，实际 {actual}；该档不保存图片')


def lock_focus(cap, target, settle):
    return set_manual_focus(cap,target,settle=settle)


def read_json(path):
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON 顶层必须是对象：{path}")
    return data


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_image(path):
    """imread replacement that also handles Chinese Windows paths."""
    encoded = np.fromfile(path, dtype=np.uint8)
    if not encoded.size:
        return None
    try:
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    except cv2.error:
        return None


def write_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise OSError(f"图像编码失败：{path}")
    encoded.tofile(path)


def board_spec(cols, rows, square_mm):
    if cols < 3 or rows < 3 or not math.isfinite(square_mm) or square_mm <= 0:
        raise ValueError("内角点行列数至少为 3，方格边长须为有限正数")
    return {"inner_corners": [cols, rows], "square_size_mm": float(square_mm)}


def object_points(spec):
    cols, rows = spec["inner_corners"]
    board_spec(cols, rows, spec["square_size_mm"])
    points = np.zeros((cols * rows, 3), dtype=np.float32)
    points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    points *= spec["square_size_mm"] / 1000.0
    return points


def detect_corners(image, pattern):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    # SB already provides subpixel coordinates; do not refine them with cornerSubPix.
    ok, corners = cv2.findChessboardCornersSB(gray, tuple(pattern), cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        ok, corners = cv2.findChessboardCorners(
            gray, tuple(pattern), cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ok:
            corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), CRITERIA)
    return corners.astype(np.float32) if ok else None


def generate_board(args):
    try:
        from reportlab.lib.units import mm
        from reportlab.pdfgen import canvas
    except ImportError as error:
        raise ValueError("生成 PDF 需要 reportlab：python -m pip install -r requirements-calibration.txt") from error
    spec = board_spec(args.cols, args.rows, args.square_mm)
    # A4 landscape; keep a white border of at least one square around the pattern.
    page_w, page_h = 297., 210.
    bw, bh = (args.cols + 1) * args.square_mm, (args.rows + 1) * args.square_mm
    margin_x, margin_y = (page_w - bw) / 2, (page_h - bh) / 2
    if margin_x < max(args.square_mm, 10) or margin_y < max(args.square_mm, 30):
        raise ValueError("棋盘格超出 A4 横向可用区域，请减小 --square-mm 或内角点数量")
    pdf = Path(args.pdf)
    png = Path(args.png)
    if any(p.exists() for p in (pdf, png)) and not args.overwrite:
        raise ValueError("棋盘格文件已存在；请更换输出路径或使用 --overwrite")
    pdf.parent.mkdir(parents=True, exist_ok=True)
    title = f"CAMERA CALIBRATION  /  {args.cols} x {args.rows} INNER CORNERS"
    subtitle = (f"{args.cols + 1} x {args.rows + 1} squares   |   "
                f"Square: {args.square_mm:g} mm   |   Pattern: {bw:g} x {bh:g} mm")
    c = canvas.Canvas(str(pdf), pagesize=(page_w * mm, page_h * mm))
    c.setTitle("Camera calibration checkerboard - A4 landscape - actual size")
    c.setAuthor("Camera calibration toolkit")
    c.setFont("Helvetica-Bold", 14)
    c.drawCentredString(page_w / 2 * mm, 198 * mm, title)
    c.setFont("Helvetica", 10)
    c.drawCentredString(page_w / 2 * mm, 191 * mm, subtitle)
    c.setFillColorRGB(0, 0, 0)
    for row in range(args.rows + 1):
        for col in range(args.cols + 1):
            if (row + col) % 2 == 0:
                c.rect((margin_x + col * args.square_mm) * mm,
                       (margin_y + (args.rows - row) * args.square_mm) * mm,
                       args.square_mm * mm, args.square_mm * mm, stroke=0, fill=1)
    c.setFont("Helvetica", 9)
    c.drawCentredString(page_w / 2 * mm, 24 * mm,
                        "Print at 100% / Actual size. Disable Fit to page. Mount flat on a rigid board.")
    c.drawCentredString(page_w / 2 * mm, 18 * mm,
                        "Measure the squares and the 100 mm ruler below before capturing images.")
    rx, ry = (page_w - 100) / 2, 10
    c.setLineWidth(.6)
    c.line(rx * mm, ry * mm, (rx + 100) * mm, ry * mm)
    for i in range(11):
        c.line((rx + i * 10) * mm, (ry - 1) * mm, (rx + i * 10) * mm, (ry + 1) * mm)
    c.setFont("Helvetica", 7)
    c.drawCentredString(page_w / 2 * mm, 5 * mm, "100 mm - physical scale check")
    c.showPage()
    c.save()

    # 254 dpi = exactly 10 px/mm. pHYs embeds the physical print scale in PNG.
    scale = 10
    image = np.full((210 * scale, 297 * scale), 255, dtype=np.uint8)
    for row in range(args.rows + 1):
        for col in range(args.cols + 1):
            if (row + col) % 2 == 0:
                x0, x1 = [round((margin_x + v * args.square_mm) * scale) for v in (col, col + 1)]
                y0, y1 = [round((margin_y + v * args.square_mm) * scale) for v in (row, row + 1)]
                image[y0:y1, x0:x1] = 0
    for text, y in [(title, 120), (subtitle, 190),
                    ("Print at 100% / Actual size. Disable Fit to page.", 1880),
                    ("Mount flat. Check square size with a ruler before capture.", 1940)]:
        size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, .8, 1)[0]
        cv2.putText(image, text, ((image.shape[1] - size[0]) // 2, y),
                    cv2.FONT_HERSHEY_SIMPLEX, .8, 0, 1, cv2.LINE_AA)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise OSError("PNG 编码失败")
    payload = b"pHYs" + struct.pack(">IIB", 10000, 10000, 1)
    chunk = struct.pack(">I", 9) + payload + struct.pack(">I", zlib.crc32(payload) & 0xffffffff)
    png.parent.mkdir(parents=True, exist_ok=True)
    png.write_bytes(encoded.tobytes()[:33] + chunk + encoded.tobytes()[33:])
    print(f"PDF：{pdf.resolve()}\nPNG：{png.resolve()}\n内角点 {args.cols}x{args.rows}，每格 {args.square_mm:g} mm")


def capture_settings(args):
    config = read_json(args.config)
    width, height = config.get("camera_resolution", [640, 480])
    settings = {"camera_index": config.get("camera_index", 1),
                "image_size": [width, height], "camera_fps": config.get("camera_fps", 15),
                "flip_camera_180": config.get("flip_camera_180", False)}
    if args.camera is not None:
        settings["camera_index"] = args.camera
    if (args.width is None) != (args.height is None):
        raise ValueError("--width 和 --height 必须一起指定")
    if args.width is not None:
        settings["image_size"] = [args.width, args.height]
    if args.flip_180 is not None:
        settings["flip_camera_180"] = args.flip_180
    if settings["camera_index"] < 0 or min(settings["image_size"]) <= 0:
        raise ValueError("相机编号须非负，分辨率须为正")
    return settings


def capture_images(args):
    settings = capture_settings(args)
    spec = board_spec(args.cols, args.rows, args.square_mm)
    parent = Path(args.output_dir)
    parent.mkdir(parents=True, exist_ok=True)
    # Always create a new session to avoid mixing cameras, orientation or patterns.
    session = parent / datetime.now().strftime("session_%Y%m%d_%H%M%S_%f")
    session.mkdir()
    metadata = {**settings, **spec, "images_already_oriented": True,
                "created_at": datetime.now().astimezone().isoformat()}
    write_json(session / "capture.json", metadata)
    backend = {"any": cv2.CAP_ANY, "dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF}[args.backend]
    cap = cv2.VideoCapture(settings["camera_index"], backend)
    count = 0
    try:
        if not cap.isOpened():
            raise ValueError("无法打开相机，请关闭定位程序后检查相机编号")
        width, height = settings["image_size"]
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, settings["camera_fps"])
        if args.focus is not None:
            state = lock_focus(cap, args.focus, args.focus_settle)
            metadata.update(focus_position=args.focus, focus_lock_verified=True,
                            focus_initial_readback=state, focus_samples={}, backend=cap.getBackendName())
            write_json(session / "capture.json", metadata)
            print(f"已请求并核验手动焦点 {args.focus:g}（驱动回读），采集期间持续检查。")
        print(f"采集目录：{session.resolve()}\n设置：{settings}\nS/空格保存，Q/ESC 退出；仅保存检测成功且清晰的原始画面。")
        cv2.namedWindow("Calibration capture", cv2.WINDOW_NORMAL)
        while True:
            if args.focus is not None:
                verify_focus(cap, args.focus)
            ok, frame = cap.read()
            if not ok:
                raise ValueError("相机读取失败")
            if frame.shape[1::-1] != (width, height):
                raise ValueError(f"实际分辨率 {frame.shape[1::-1]} 与设置 {(width, height)} 不符")
            focus_state = verify_focus(cap, args.focus) if args.focus is not None else None
            if settings["flip_camera_180"]:
                frame = cv2.flip(frame, -1)
            corners = detect_corners(frame, spec["inner_corners"])
            sharpness = None
            if corners is not None:
                x, y, w, h = cv2.boundingRect(corners)
                gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
                sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            preview = frame.copy()
            found = corners is not None
            if found:
                cv2.drawChessboardCorners(preview, tuple(spec["inner_corners"]), corners, True)
            label = f"saved={count} | {'FOUND' if found else 'NOT FOUND'} | S/Space: save  Q: quit"
            cv2.putText(preview, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .47,
                        (0, 220, 0) if found else (0, 180, 255), 1, cv2.LINE_AA)
            if sharpness is not None:
                cv2.putText(preview, f"board sharpness={sharpness:.0f}", (8, 44),
                            cv2.FONT_HERSHEY_SIMPLEX, .47, (0, 220, 0), 1, cv2.LINE_AA)
            if args.focus is not None:
                cv2.putText(preview, f"MANUAL focus={args.focus:g} (readback checked)", (8, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, .47, (0, 220, 255), 1, cv2.LINE_AA)
            cv2.imshow("Calibration capture", preview)
            key = cv2.waitKey(1) & 0xff
            if key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty("Calibration capture", cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("s"), ord("S"), 32):
                if not found:
                    print("未找到完整棋盘格，请调整角度和距离。")
                elif sharpness < args.min_sharpness:
                    print(f"图像较模糊 ({sharpness:.0f} < {args.min_sharpness:g})，请固定棋盘并检查对焦。")
                else:
                    count += 1
                    filename = f"frame_{count:04d}.png"
                    write_image(session / filename, frame)
                    if focus_state is not None:
                        metadata["focus_samples"][filename] = focus_state
                        write_json(session / "capture.json", metadata)
                    print(f"已保存 {count} 张")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print(f"共保存 {count} 张；建议 20-30 张，覆盖中心、四角及不同倾角。\n"
          f'下一步：python camera_calibration.py calibrate --images "{session.resolve()}"')


def calibrate_points(points, spec, image_size):
    """Five-coefficient OpenCV pinhole model; point coordinates and poses use m."""
    if len(points) < 10:
        raise ValueError("至少需要 10 张成功检测的图像，建议采集 20-30 张不同姿态")
    obj = object_points(spec)
    rms, matrix, dist, rvecs, tvecs = cv2.calibrateCamera(
        [obj.copy() for _ in points], points, tuple(image_size), None, None,
        flags=0, criteria=CRITERIA)
    if not np.isfinite(rms) or not np.isfinite(matrix).all() or not np.isfinite(dist).all():
        raise ValueError("标定结果包含非有限值，请检查角点、棋盘尺寸和姿态多样性")
    if min(matrix[0, 0], matrix[1, 1]) <= 0:
        raise ValueError("标定焦距无效")
    errors = []
    for corners, rv, tv in zip(points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, rv, tv, matrix, dist)
        residual = projected.reshape(-1, 2) - corners.reshape(-1, 2)
        errors.append(float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1)))))
    return {"camera_matrix": matrix.tolist(), "dist_coeffs": dist.reshape(-1).tolist(),
            "rms": float(rms), "per_view_rmse_px": errors}


def resolve_dataset(args):
    folder = Path(args.images)
    if not folder.is_dir():
        raise ValueError(f"图像目录不存在：{folder}")
    metadata = read_json(folder / "capture.json") if (folder / "capture.json").exists() else None
    if metadata:
        if metadata.get("images_already_oriented") is not True:
            raise ValueError("capture.json 未明确 images_already_oriented=true")
        cols, rows = metadata["inner_corners"]
        square = metadata["square_size_mm"] if args.square_mm is None else args.square_mm
        spec = board_spec(cols, rows, square)
        for value, expected in [(args.cols, cols), (args.rows, rows),
                                (args.flip_180, metadata["flip_camera_180"])]:
            if value is not None and value != expected:
                raise ValueError("命令行参数与 capture.json 不符；请按原会话设置标定")
        orientation = metadata["flip_camera_180"]
        rotate = False  # Saved captures are already oriented; never rotate twice.
    else:
        if args.flip_180 is None:
            raise ValueError("外部图片须指定 --flip-180（旋转原图）或 --no-flip-180（保持原图）")
        spec = board_spec(args.cols if args.cols is not None else 9,
                          args.rows if args.rows is not None else 6,
                          args.square_mm if args.square_mm is not None else 20)
        orientation, rotate = args.flip_180, args.flip_180
    return folder, metadata, spec, orientation, rotate


def calibration_warnings(result, points, image_size):
    warnings = []
    width, height = image_size
    matrix = np.asarray(result["camera_matrix"])
    if result["rms"] > 1:
        warnings.append("总体重投影 RMS 超过 1 px，建议检查模糊、反光及棋盘平整度")
    if len(points) < 20:
        warnings.append("有效图像不足 20 张，建议补充边缘位置及不同倾角")
    all_points = np.concatenate(points).reshape(-1, 2)
    low, high = all_points.min(axis=0), all_points.max(axis=0)
    if low[0] > .15 * width or high[0] < .85 * width or low[1] > .15 * height or high[1] < .85 * height:
        warnings.append("角点对图像边缘覆盖不足，建议补拍四周区域")
    if not (0 < matrix[0, 2] < width and 0 < matrix[1, 2] < height):
        warnings.append("估计主点位于画面外，请检查内角点行列和拍摄姿态")
    return warnings


def undistort_image(image, result, alpha=1.):
    size = tuple(result["image_size"])
    if image.shape[1::-1] != size:
        raise ValueError(f"图像尺寸 {image.shape[1::-1]} 与标定尺寸 {size} 不符")
    matrix = np.asarray(result["camera_matrix"], dtype=float)
    dist = np.asarray(result["dist_coeffs"], dtype=float)
    new_matrix, roi = cv2.getOptimalNewCameraMatrix(matrix, dist, size, alpha, size)
    corrected = cv2.undistort(image, matrix, dist, None, new_matrix)
    return corrected, new_matrix, roi


def comparison_image(image, corrected):
    left, right = image.copy(), corrected.copy()
    cv2.putText(left, "ORIGINAL", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 220, 255), 2)
    cv2.putText(right, "UNDISTORTED", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 220, 255), 2)
    return np.hstack([left, right])


def calibrate_images(args):
    folder, metadata, spec, orientation, rotate = resolve_dataset(args)
    output = Path(args.output)
    diagnostics = output.parent / f"{output.stem}_diagnostics"
    if output.exists() and not args.overwrite:
        raise ValueError("结果已存在，请更换 --output 或使用 --overwrite")
    if diagnostics.exists() and any(diagnostics.iterdir()) and not args.overwrite:
        raise ValueError("诊断目录已存在，请更换 --output 或使用 --overwrite")
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file())
    if not files:
        raise ValueError("目录中没有支持的图像")
    image_size = None
    points, accepted, records = [], [], []
    for file in files:
        image = read_image(file)
        record = {"file": file.name, "status": "unreadable", "rmse_px": ""}
        records.append(record)
        if image is None:
            print(f"无法读取，跳过：{file.name}")
            continue
        size = image.shape[1::-1]
        if image_size is None:
            image_size = size
        if size != image_size:
            raise ValueError(f"混合分辨率：{file.name}={size}，预期 {image_size}；请分开标定")
        if metadata and list(size) != metadata["image_size"]:
            raise ValueError(f"{file.name} 与 capture.json 的分辨率不符")
        if rotate:
            image = cv2.flip(image, -1)
        corners = detect_corners(image, spec["inner_corners"])
        if corners is None:
            record["status"] = "not_found"
            print(f"未找到棋盘格，跳过：{file.name}")
            continue
        points.append(corners)
        if metadata and "focus_position" in metadata:
            state = metadata.get("focus_samples", {}).get(file.name, {})
            if (metadata.get("focus_lock_verified") is not True or state.get("autofocus_mode") != "MANUAL"
                    or state.get("focus_readback") != metadata["focus_position"]):
                raise ValueError(f"{file.name} 缺少有效固定焦点记录，请使用完整原始采集会话")
        accepted.append(record)
        record["status"] = "used"
        cv2.drawChessboardCorners(image, tuple(spec["inner_corners"]), corners, True)
        write_image(diagnostics / "corners" / (file.stem + ".png"), image)
        print(f"检测成功：{file.name}")
    result = calibrate_points(points, spec, image_size)
    for record, error in zip(accepted, result.pop("per_view_rmse_px")):
        record["rmse_px"] = error
    # Optional explicit rejection: never silently discard views based on fit error.
    if args.max_view_error is not None:
        keep = [i for i, record in enumerate(accepted) if record["rmse_px"] <= args.max_view_error]
        if len(keep) < 10:
            raise ValueError("按逐张误差阈值筛选后不足 10 张；请补拍或调整阈值")
        if len(keep) < len(points):
            for i, record in enumerate(accepted):
                if i not in keep:
                    record["status"] = "excluded_by_initial_rmse"
            points = [points[i] for i in keep]
            accepted = [accepted[i] for i in keep]
            result = calibrate_points(points, spec, image_size)
            for record, error in zip(accepted, result.pop("per_view_rmse_px")):
                record["rmse_px"] = error
    warnings = calibration_warnings(result, points, image_size)
    result.update({"model": "opencv_pinhole", "image_size": list(image_size),
                   "flip_camera_180": orientation, **spec, "num_images": len(points),
                   "created_at": datetime.now().astimezone().isoformat(), "opencv_version": cv2.__version__,
                   "source_directory": str(folder.resolve()), "views": records, "warnings": warnings})
    if metadata:
        result["camera_index"] = metadata["camera_index"]
        if "focus_position" in metadata:
            result.update(focus_position=metadata["focus_position"], focus_lock_verified=True,
                          focus_control_backend=metadata.get("backend"),
                          focus_note="Manual control and driver readback checked; not per-exposure encoder metadata.")
    diagnostics.mkdir(parents=True, exist_ok=True)
    with (diagnostics / "per_view_errors.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["file", "status", "rmse_px"])
        writer.writeheader()
        writer.writerows(records)
    # Use a final inlier for the comparison even when the first detection was excluded.
    first = read_image(folder / accepted[0]["file"])
    if rotate:
        first = cv2.flip(first, -1)
    corrected, new_matrix, roi = undistort_image(first, result)
    write_image(diagnostics / "undistortion_comparison.png", comparison_image(first, corrected))
    write_json(diagnostics / "undistortion_info.json", {
        "camera_matrix": new_matrix.tolist(), "dist_coeffs": [0.] * 5,
        "image_size": list(image_size), "valid_roi_xywh": list(map(int, roi)),
        "note": "Only for full-frame undistorted images; localization uses original images and original K,D."})
    write_json(output, result)
    print(f"\n有效图像：{len(points)}/{len(files)}，RMS：{result['rms']:.4f} px\n"
          f"K =\n{np.asarray(result['camera_matrix'])}\nD = {result['dist_coeffs']}\n"
          f"结果：{output.resolve()}\n诊断：{diagnostics.resolve()}")
    for warning in warnings:
        print(f"提示：{warning}")


def undistort_file(args):
    result = read_json(args.calibration)
    if result.get("model", "opencv_pinhole") != "opencv_pinhole":
        raise ValueError("只支持 OpenCV 普通针孔模型")
    source = Path(args.image)
    output = Path(args.output)
    if output.resolve() == source.resolve():
        raise ValueError("输出不能覆盖输入图像")
    if output.exists() and not args.overwrite:
        raise ValueError("输出已存在，请更换路径或使用 --overwrite")
    image = read_image(source)
    if image is None:
        raise ValueError("图像无法读取")
    # Capture-session PNGs already have the rotation applied.
    if args.raw and result["flip_camera_180"]:
        image = cv2.flip(image, -1)
    corrected, new_matrix, roi = undistort_image(image, result, args.alpha)
    write_image(output, comparison_image(image, corrected))
    print(f"对比图：{output.resolve()}\n去畸变图内参：\n{new_matrix}\n有效区域 xywh：{roi}")


def add_pattern(parser, optional=False):
    parser.add_argument("--cols", type=int, default=None if optional else 9, help="内角点列数")
    parser.add_argument("--rows", type=int, default=None if optional else 6, help="内角点行数")
    parser.add_argument("--square-mm", type=float, default=None if optional else 20., help="实测单格边长 mm")


def add_orientation(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--flip-180", dest="flip_180", action="store_true", help="将原始图像旋转 180 度")
    group.add_argument("--no-flip-180", dest="flip_180", action="store_false", help="保持原始图像方向")
    parser.set_defaults(flip_180=None)


def main(argv=None):
    parser = argparse.ArgumentParser(description="相机内参标定工具：棋盘格 / 采集 / 标定 / 去畸变")
    commands = parser.add_subparsers(dest="command", required=True)
    board = commands.add_parser("board", help="生成 A4 横向矢量 PDF 和 PNG 棋盘格")
    add_pattern(board)
    board.add_argument("--pdf", type=Path, default=ROOT / "output/pdf/chessboard_9x6_20mm_A4.pdf")
    board.add_argument("--png", type=Path, default=ROOT / "calibration/boards/chessboard_9x6_20mm_A4.png")
    board.add_argument("--overwrite", action="store_true")
    board.set_defaults(run=generate_board)

    capture = commands.add_parser("capture", help="按定位程序的分辨率和方向采集图像")
    capture.add_argument("--config", type=Path, default=ROOT / "config/positioning_config.json")
    capture.add_argument("--camera", type=int)
    capture.add_argument("--width", type=int)
    capture.add_argument("--height", type=int)
    capture.add_argument("--backend", choices=["any", "dshow", "msmf"], default="any")
    capture.add_argument("--focus", type=float, help="固定对焦位置；正式分焦点标定应明确指定")
    capture.add_argument("--focus-settle", type=float, default=2., help="固定焦点后读帧等待秒数，默认 2")
    add_orientation(capture)
    add_pattern(capture)
    capture.add_argument("--min-sharpness", type=float, default=50., help="棋盘区域 Laplacian 方差阈值，0 关闭")
    capture.add_argument("--output-dir", type=Path, default=ROOT / "calibration/images")
    capture.set_defaults(run=capture_images)

    calibrate = commands.add_parser("calibrate", help="离线标定并生成诊断文件")
    calibrate.add_argument("--images", type=Path, required=True, help="单次会话的图像目录")
    calibrate.add_argument("--output", type=Path, default=ROOT / "calibration/camera_intrinsics.json")
    add_pattern(calibrate, optional=True)
    add_orientation(calibrate)
    calibrate.add_argument("--max-view-error", type=float, help="可选，按初次拟合的逐张 RMS 筛除一次，单位 px")
    calibrate.add_argument("--overwrite", action="store_true")
    calibrate.set_defaults(run=calibrate_images)

    undistort = commands.add_parser("undistort", help="生成原图和去畸变图的左右对比")
    undistort.add_argument("--calibration", type=Path, default=ROOT / "calibration/camera_intrinsics.json")
    undistort.add_argument("--image", type=Path, required=True)
    undistort.add_argument("--output", type=Path, default=ROOT / "calibration/undistortion_comparison.png")
    undistort.add_argument("--raw", action="store_true", help="输入来自相机原始方向，按标定设置旋转；采集会话图片不加此项")
    undistort.add_argument("--alpha", type=float, default=1., help="0 裁去黑边，1 保留更多视场，输出尺寸保持不变")
    undistort.add_argument("--overwrite", action="store_true")
    undistort.set_defaults(run=undistort_file)
    args = parser.parse_args(argv)
    try:
        if hasattr(args, "focus"):
            if args.focus is not None and not math.isfinite(args.focus):
                raise ValueError("--focus 须为有限数值")
            if not math.isfinite(args.focus_settle) or args.focus_settle < 0:
                raise ValueError("--focus-settle 须为有限非负数")
        if hasattr(args, "alpha") and not 0 <= args.alpha <= 1:
            raise ValueError("--alpha 须介于 0 和 1")
        if hasattr(args, "max_view_error") and args.max_view_error is not None:
            if not math.isfinite(args.max_view_error) or args.max_view_error <= 0:
                raise ValueError("--max-view-error 须为有限正数")
        if hasattr(args, "min_sharpness") and (not math.isfinite(args.min_sharpness) or args.min_sharpness < 0):
            raise ValueError("--min-sharpness 须为有限非负数")
        args.run(args)
    except (OSError, ValueError, KeyError, TypeError, cv2.error) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("已停止。", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
