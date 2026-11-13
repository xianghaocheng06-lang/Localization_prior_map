"""Interactive focus control probe and stationary-checkerboard repeatability check.

No calibration or camera configuration files are modified. Device settings may
persist after exit; all control requests and readbacks are logged separately.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import time

import cv2
import numpy as np

from calibration.camera_calibration import detect_corners, read_json, write_image, write_json, autofocus_mode

ROOT = Path(__file__).resolve().parents[1]


def focus_state(cap):
    def get(prop):
        value = float(cap.get(prop))
        return value if np.isfinite(value) else None
    raw = get(cv2.CAP_PROP_AUTOFOCUS)
    return {"focus_readback": get(cv2.CAP_PROP_FOCUS), "autofocus_readback": raw,
            "autofocus_mode": autofocus_mode(raw, cap.getBackendName())}


def corner_difference(reference, current):
    """Compare absolute image positions; don't align away optical scale changes."""
    a, b = np.asarray(reference, float).reshape(-1, 2), np.asarray(current, float).reshape(-1, 2)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("角点维度或数值无效")
    # A rectangular chessboard can be returned with a 180-degree reversed order.
    squared = min((np.sum((a - candidate)**2, axis=1) for candidate in (b, b[::-1])), key=np.mean)
    center_shift = float(np.linalg.norm(a.mean(0) - b.mean(0)))
    spread_a = np.sqrt(np.mean(np.sum((a - a.mean(0))**2, axis=1)))
    spread_b = np.sqrt(np.mean(np.sum((b - b.mean(0))**2, axis=1)))
    if spread_a <= 0:
        raise ValueError("参考角点几何退化")
    return {"corner_rms_px": float(np.sqrt(squared.mean())),
            "center_shift_px": center_shift, "relative_spread_change_percent": float((spread_b / spread_a - 1) * 100)}


def capture(args):
    cfg = read_json(args.config)
    camera = cfg["camera_index"] if args.camera is None else args.camera
    backend = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "any": cv2.CAP_ANY}[args.backend]
    folder = args.output / datetime.now().strftime("session_%Y%m%d_%H%M%S_%f")
    folder.mkdir(parents=True)
    cap = cv2.VideoCapture(camera, backend)
    events, samples, history = [], [], []
    target = None
    return_focus = None
    last_command = time.monotonic()
    manual_requested = False

    def command(prop, value, name):
        nonlocal last_command
        before = focus_state(cap)
        accepted = bool(cap.set(prop, value))
        last_command = time.monotonic()
        history.clear()
        events.append({"time": datetime.now().astimezone().isoformat(), "property": name,
                       "requested": value, "set_return": accepted, "before": before,
                       "immediate_readback": focus_state(cap)})
        print(f"{name}={value:g}，set 返回 {accepted}；立即回读 {focus_state(cap)}", flush=True)
        write_json(folder / "events.json", events)
        return accepted

    try:
        if not cap.isOpened():
            raise ValueError("无法打开相机；先关闭定位程序，核对编号或更换 --backend")
        width, height = cfg["camera_resolution"]
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, cfg["camera_fps"])
        initial = focus_state(cap)
        write_json(folder / "session.json", {"camera_index": camera, "backend": cap.getBackendName(),
            "image_size": [width, height], "flip_camera_180": cfg["flip_camera_180"],
            "inner_corners": [args.cols, args.rows], "initial_readback": initial,
            "note": "Readback is a driver property, not verified physical or per-exposure metadata. Scene must stay fixed."})
        print(f"输出目录：{folder.resolve()}\n初始读数：{initial}\n"
              "A 自动对焦；M 请求手动；B 记住返回焦点；[ / ] 减/增焦点；R 返回记住的焦点；S 保存；Q 退出。\n"
              "相机和棋盘均须固定。读数为 0 或 set=True 都不能单独证明接口有效。", flush=True)
        cv2.namedWindow("Focus probe", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Focus probe", 960, 720)
        while True:
            before = focus_state(cap)
            ok, frame = cap.read()
            if not ok:
                raise ValueError("相机读帧失败")
            state = focus_state(cap)
            history.append((time.monotonic(), state["focus_readback"]))
            history[:] = [(stamp, value) for stamp, value in history if time.monotonic() - stamp <= 1.]
            if frame.shape[1::-1] != (width, height):
                raise ValueError(f"实际分辨率为 {frame.shape[1::-1]}，与配置不符")
            if cfg["flip_camera_180"]:
                frame = cv2.flip(frame, -1)
            corners = detect_corners(frame, (args.cols, args.rows))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if corners is not None:
                x, y, w, h = cv2.boundingRect(corners)
                roi = gray[max(y, 0):min(y+h, height), max(x, 0):min(x+w, width)]
            else:
                roi = gray
            sharpness = float(cv2.Laplacian(roi, cv2.CV_64F).var()) if roi.size else 0.
            elapsed = time.monotonic() - last_command
            preview = frame.copy()
            if corners is not None:
                cv2.drawChessboardCorners(preview, (args.cols, args.rows), corners, True)
            labels = [f"focus={state['focus_readback']}  AF={state['autofocus_mode']} (raw={state['autofocus_readback']})  requested={target}",
                f"board={'FOUND' if corners is not None else 'NOT FOUND'}  sharpness={sharpness:.0f}  saved={len(samples)}",
                "A:auto M:manual B:remember [ ]:focus R:return S:save Q:quit",
                f"return focus={return_focus} | wait after request: {max(0.,args.settle-elapsed):.1f}s"]
            for i, label in enumerate(labels):
                cv2.putText(preview, label, (8, 20+i*21), cv2.FONT_HERSHEY_SIMPLEX, .43, (0, 230, 255), 1, cv2.LINE_AA)
            cv2.imshow("Focus probe", preview)
            key = cv2.waitKey(1) & 0xff
            if key in (ord('q'), ord('Q'), 27) or cv2.getWindowProperty("Focus probe", cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord('a'), ord('A')):
                command(cv2.CAP_PROP_AUTOFOCUS, 1, "autofocus")
                target = None
                manual_requested = False
            elif key in (ord('m'), ord('M')):
                manual_requested = command(cv2.CAP_PROP_AUTOFOCUS, 0, "autofocus")
                target = None
            elif key in (ord('b'), ord('B')):
                return_focus = state["focus_readback"]
                print(f"已记住返回焦点 {return_focus}；这只是驱动读数，尚未证明物理位置准确。", flush=True)
            elif key in (ord('['), ord(']'), ord('r'), ord('R')):
                if not manual_requested or state["autofocus_mode"] != "MANUAL":
                    print("先按 M 请求关闭自动对焦，并核对 AF=MANUAL（DSHOW 通常 raw=2）；未知模式需用厂家工具确认。", flush=True)
                    continue
                if key in (ord('r'), ord('R')):
                    value = return_focus
                else:
                    current = target if target is not None else state["focus_readback"]
                    value = None if current is None else current + (args.step if key == ord(']') else -args.step)
                if value is None:
                    print("没有可用焦点读数；返回测试须先按 B。", flush=True)
                elif command(cv2.CAP_PROP_FOCUS, value, "focus"):
                    target = value
            elif key in (ord('s'), ord('S')):
                if elapsed < args.settle:
                    print("刚发出控制请求，请等倒计时结束并确认画面已稳定。", flush=True)
                    continue
                number = len(samples)+1
                name = f"sample_{number:03d}.png"
                values = [v for _, v in history if v is not None]
                sample = {"sample": number, "image": name, "time": datetime.now().astimezone().isoformat(),
                    **state, "requested_focus": target, "manual_set_request_accepted": manual_requested,
                    "return_focus": return_focus, "seconds_since_request": elapsed, "before_read": before,
                    "recent_focus_min": min(values) if values else None, "recent_focus_max": max(values) if values else None,
                    "sharpness": sharpness, "corners": None if corners is None else corners.reshape(-1, 2).tolist()}
                write_image(folder / name, frame)
                samples.append(sample)
                write_json(folder / "samples.json", samples)
                print(f"已保存 {name}，focus={state['focus_readback']}，AF={state['autofocus_readback']}，sharpness={sharpness:.0f}", flush=True)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        write_json(folder / "events.json", events)
        write_json(folder / "samples.json", samples)
        print("已关闭相机。对焦设置可能由驱动保留；下一次采集前请核对所需模式。", flush=True)
    analyze(folder)


def analyze(folder):
    folder = Path(folder)
    samples = json.loads((folder / "samples.json").read_text(encoding="utf-8"))
    events = json.loads((folder / "events.json").read_text(encoding="utf-8"))
    session = read_json(folder / "session.json") if (folder / "session.json").exists() else {}
    backend = session.get("backend", "")
    comparisons = []
    references = {}
    for sample in samples:
        # Deliberately do not round/merge nearby focus codes or mix AF-on samples.
        mode = autofocus_mode(sample["autofocus_readback"], backend)
        if mode != "MANUAL" or sample["corners"] is None or sample["focus_readback"] is None:
            continue
        q = sample["focus_readback"]
        if q in references:
            ref = references[q]
            comparisons.append({"focus_readback": q, "reference_sample": ref["sample"], "sample": sample["sample"],
                **corner_difference(ref["corners"], sample["corners"]),
                "reference_sharpness": ref["sharpness"], "sharpness": sample["sharpness"]})
        else:
            references[q] = sample
    fields = ["focus_readback", "reference_sample", "sample", "corner_rms_px", "center_shift_px",
              "relative_spread_change_percent", "reference_sharpness", "sharpness"]
    with (folder / "repeatability.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparisons)
    focus_values = sorted({s["focus_readback"] for s in samples if s["focus_readback"] is not None})
    summary = {"saved_images": len(samples), "saved_focus_readbacks": focus_values,
        "accepted_focus_requests": sum(e["property"] == "focus" and e["set_return"] for e in events),
        "backend": backend, "same_focus_comparisons": len(comparisons), "comparisons": comparisons,
        "interpretation": "Driver readback and fixed-scene repeatability evidence only; not a focus encoder or intrinsics calibration."}
    write_json(folder / "summary.json", summary)
    lines = ["# 对焦接口与回位重复性检查", "", f"保存图像 {len(samples)} 张；保存时焦点读数 {focus_values}。",
        f"后端：{backend or '未记录'}；同焦点且解码为手动模式、成功检测棋盘的比较共 {len(comparisons)} 次。", "",
        "| 焦点 | 参考图→比较图 | 角点 RMS px | 中心位移 px | 尺度变化 % |", "|---|---|---:|---:|---:|"]
    for r in comparisons:
        lines.append(f"| {r['focus_readback']} | {r['reference_sample']}→{r['sample']} | {r['corner_rms_px']:.3f} | {r['center_shift_px']:.3f} | {r['relative_spread_change_percent']:.3f} |")
    lines += ["", "只在相机、棋盘及照明稳定时解释这些结果。没有自动平移/缩放配准，以免消除要检查的成像变化。",
        "先在同一焦点连续保存几张，估计静止检测噪声，再比较离开后返回该焦点的结果。回位差异应与本机的静止噪声比较，不用一个通用像素阈值直接判定合格。",
        "DirectShow 的 AF raw=1 表示自动、raw=2 表示手动；布尔式后端一般为 1/0。未知模式不参加比较。",
        "set=True、手动模式回读、读数不变都不能单独证明硬件已执行。至少要确认多个设置产生不同回读和可见对焦变化；返回同一个读数后清晰度和几何应可重复。",
        "读数一直为 0 或不随操作改变，可能是接口未支持，也可能是合法值/步长问题；需换后端或厂家工具验证。返回值不是逐帧曝光元数据。",
        "没有同焦点比较时，本次数据不足以评价回位重复性；请按说明补采。仅有少量固定棋盘比较不能证明所有场景下参数稳定，也没有完成内参随焦点标定。"]
    (folder / "检查报告.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(f"检查报告：{(folder / '检查报告.md').resolve()}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture_parser = commands.add_parser("capture", help="交互式检查，启动时不自动修改对焦模式")
    capture_parser.add_argument("--config", type=Path, default=ROOT / "config/positioning_config.json")
    capture_parser.add_argument("--camera", type=int)
    capture_parser.add_argument("--backend", choices=["dshow", "msmf", "any"], default="dshow")
    capture_parser.add_argument("--step", type=float, default=5., help="一次手动设置增量，不代表设备实际步长或范围")
    capture_parser.add_argument("--settle", type=float, default=2., help="每次请求后至少等待秒数；仍需人工确认稳定")
    capture_parser.add_argument("--cols", type=int, default=9)
    capture_parser.add_argument("--rows", type=int, default=6)
    capture_parser.add_argument("--output", type=Path, default=ROOT / "calibration/focus_checks")
    analyze_parser = commands.add_parser("analyze", help="重新分析已保存的会话")
    analyze_parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "capture":
            if not np.isfinite(args.step) or args.step <= 0 or not np.isfinite(args.settle) or args.settle < 0:
                raise ValueError("step 须为有限正数，settle 须为有限非负数")
            if args.cols < 3 or args.rows < 3:
                raise ValueError("棋盘内角点行列数至少为 3")
            capture(args)
        else:
            analyze(args.folder)
    except (OSError, ValueError, cv2.error) as error:
        parser.exit(1, f"错误：{error}\n")


if __name__ == "__main__":
    main()
