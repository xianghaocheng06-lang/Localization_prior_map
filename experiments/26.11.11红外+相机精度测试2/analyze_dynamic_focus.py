"""Evaluate the 9 dynamic-focus captures and replay them with fixed intrinsics."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
sys.path[:0] = [str(PROJECT), str(ROOT.parent / "26.11.11红外+相机精度测试")]
from analyze_20261111 import analyze, csv_transform, make_3d, save_json, self_check_registration, summarize
from localization.positioning_core import VisionEstimator, load_camera_calibration, load_config

MARKER_FILE = "markerPos_test_2.mat"
EXCLUDED_MOCAP_GROUP = 10
POSITION, ATTITUDE = "camera_position_error_mm", "camera_rotation_error_deg"
METRICS = (POSITION, ATTITUDE)


def read_sidecars(rows):
    keys = {(str(row.group_id), row.snapshot_at) for row in rows.itertuples()}
    snapshots = ROOT / "camera_snapshots"
    if not snapshots.is_dir() and (PROJECT / "camera_snapshots").is_dir():
        snapshots = PROJECT / "camera_snapshots"
    sidecars = {}
    for date in {time[:10].replace("-", "") for _, time in keys}:
        for path in snapshots.glob(f"*_{date}_*.json"):
            info = json.loads(path.read_text(encoding="utf-8"))
            key = (str(info["group_id"]), info["snapshot_at"])
            if key in keys:
                if key in sidecars:
                    raise ValueError(f"组号和快照时间对应多个 JSON：{key}")
                sidecars[key] = path, info
    if sidecars.keys() != keys:
        raise ValueError(f"缺少快照 JSON：{sorted(keys - sidecars.keys())}")
    return sidecars


def pose_error(pose, reference):
    translation = np.linalg.norm(pose[:3, 3] - reference[:3, 3])
    attitude = Rotation.from_matrix(pose[:3, :3] @ reference[:3, :3].T).magnitude()
    return float(translation), float(np.degrees(attitude))


def focus_method(table, focus):
    if focus < table.focus[0] or focus > table.focus[-1]:
        return "extrapolation"
    if np.any(np.isclose(focus, table.focus, atol=1e-6, rtol=0)):
        validated = any(v["kind"] == "anchor" and v["focus"] == focus and v["passed"]
                        for v in table.data["validation"])
        return "validated_anchor" if validated else "unvalidated_anchor"
    return ("validated_interpolation" if any(a < focus < b for a, b in table.data["validated_intervals"])
            else "unvalidated_interpolation")


def compare_intrinsics(poses, rows, sidecars):
    config_path = PROJECT / "config/positioning_config.json"
    config = load_config(config_path)
    estimator = VisionEstimator(config)
    table = estimator.focus_map
    estimator.focus_map = None
    estimator.config = {**config, "flip_camera_180": False}
    fixed = load_camera_calibration(config["calibration_file"], config["camera_resolution"],
                                    config["flip_camera_180"])[:2]
    width, height = config["camera_resolution"]
    reference_by_group = {pose["group_id"]: pose["T_W_C_gt_mm"] for pose in poses}
    comparison, inputs = [], [config_path, table.path, config["calibration_file"], config["tag_layout_file"]]
    for row in rows.itertuples():
        gid = int(row.group_id)
        path, info = sidecars[(str(gid), row.snapshot_at)]
        inputs.append(path)
        focus = float(info["focus_position"])
        matrix, distortion = np.array(info["camera_matrix"]), np.array(info["dist_coeffs"])
        expected_matrix, expected_distortion = table.lookup(focus, require_validation=False)
        np.testing.assert_allclose(matrix, expected_matrix, atol=1e-9, rtol=0)
        np.testing.assert_allclose(distortion, expected_distortion, atol=1e-9, rtol=0)
        np.testing.assert_allclose([row.focus_position, row.camera_fx_px, row.camera_fy_px],
                                   [focus, matrix[0, 0], matrix[1, 1]], atol=1e-6, rtol=0)
        ids = json.loads(row.detected_tag_ids)
        corners = json.loads(row.tag_corners_px_json)
        centers = info["tag_centers_px"]
        assert ids == info["detected_tag_ids"] and len(ids) == len(corners) == len(centers)
        tags = [SimpleNamespace(tag_id=tag_id, corners=np.asarray(corner), center=np.asarray(center))
                for tag_id, corner, center in zip(ids, corners, centers)]
        estimator.detector = SimpleNamespace(detect=lambda gray: tags)
        estimates = []
        for intrinsics, coefficients in ((matrix, distortion), fixed):
            estimator.K, estimator.D = intrinsics, coefficients
            cv2.setRNGSeed(20261111 + gid)
            result = estimator.process(np.zeros((height, width, 3), dtype=np.uint8))
            if result.camera_pose is None:
                raise ValueError(f"第 {gid} 组重放失败：{result.status}")
            estimate = result.camera_pose.copy()
            estimate[:3, 3] *= 1000
            estimates.append((estimate, result.reprojection_rmse_px))
        dynamic, fixed_replay = estimates
        online = csv_transform(row._asdict(), "T_W_C")
        reproduced = pose_error(dynamic[0], online)
        assert reproduced[0] < 1e-5 and reproduced[1] < 1e-5
        reference = reference_by_group[gid]
        fixed_position, fixed_attitude = pose_error(fixed_replay[0], reference)
        comparison.append(dict(group_id=gid, focus_position=focus,
                               focus_method=focus_method(table, focus), camera_fx_px=row.camera_fx_px,
                               camera_fy_px=row.camera_fy_px, known_tag_count=len(json.loads(row.known_tag_ids)),
                               dynamic_reprojection_rmse_px=dynamic[1], fixed_reprojection_rmse_px=fixed_replay[1],
                               fixed_camera_position_error_mm=fixed_position,
                               fixed_camera_rotation_error_deg=fixed_attitude,
                               replay_position_difference_mm=reproduced[0],
                               replay_rotation_difference_deg=reproduced[1],
                               camera_metadata=path.name))
    return pd.DataFrame(comparison), inputs


def plot_errors(frame, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": .18})
    blue, orange, green = "#1b6585", "#b76220", "#2a7860"
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.6), layout="constrained")
    for ax, key, unit, name in zip(axes[0], METRICS, ("mm", "°"), ("相机位置误差", "相机姿态误差")):
        ax.plot(frame.group_id, frame[key], "o-", color=blue, label=f"动态 {frame[key].mean():.2f} {unit}")
        ax.plot(frame.group_id, frame[f"fixed_{key}"], "s--", color=orange, mfc="white",
                label=f"固定重放 {frame[f'fixed_{key}'].mean():.2f} {unit}")
        ax.set(title=name, ylabel=f"误差（{unit}）", ylim=(0, None))
        ax.legend()
    ax = axes[1, 0]
    for axis, color in zip("xyz", (blue, orange, green)):
        ax.plot(frame.group_id, frame[f"camera_error_{axis}_mm"], "o-", color=color, label=axis.upper())
    ax.axhline(0, color="#657383", lw=1)
    ax.set(title="动态内参的机翼系位置偏差", ylabel="视觉减动捕（mm）")
    ax.legend(ncol=3)
    ax = axes[1, 1]
    extrapolation = frame.focus_method == "extrapolation"
    ax.scatter(frame.loc[~extrapolation, "group_id"], frame.loc[~extrapolation, "focus_position"],
               color=blue, s=55, label="标定档位或插值")
    ax.scatter(frame.loc[extrapolation, "group_id"], frame.loc[extrapolation, "focus_position"],
               color=orange, marker="D", s=60, label="外推")
    ax.axhspan(272, 304, color="#dceee7", label="已验证区间", zorder=0)
    ax.set(title="采集时的对焦读数", ylabel="驱动读数", ylim=(263, 451))
    ax.legend(loc="upper left")
    for ax in axes.flat:
        ax.set_xticks(frame.group_id)
        ax.set_xlabel("采集组号")
    fig.suptitle("2026 年 11 月 11 日动态内参与固定内参同帧对照", fontsize=17, weight="bold")
    fig.savefig(output / "error_dashboard.png", dpi=180)
    plt.close(fig)


def report(frame, summary, output, scene):
    position_delta = frame[POSITION] - frame[f"fixed_{POSITION}"]
    attitude_delta = frame[ATTITUDE] - frame[f"fixed_{ATTITUDE}"]
    worst = frame.loc[frame[POSITION].idxmax()]
    lines = [
        "# 2026 年 11 月 11 日动态内参定位精度分析",
        "",
        "视觉 CSV 的第 1–9 组按组号对应动捕 MAT 第 1–9 组。按实验说明作废动捕第 10 组。机翼标记位置固定，采用本次第 7 组机翼标记建立统一动捕基准。每组机器人位姿使用该组自己的动捕点。",
        "",
        "| 解算方式 | 有效组数 | 平均位置误差（mm） | 位置 RMSE（mm） | 平均姿态误差（°） | 姿态 RMSE（°） |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, results in (("动态内参原始记录", summary), ("固定内参同帧重放", summary["fixed_replay"])):
        p, a = results[POSITION], results[ATTITUDE]
        lines.append(f"| {label} | {p['n']} | {p['mean']:.3f} | {p['rmse']:.3f} | {a['mean']:.3f} | {a['rmse']:.3f} |")
    lines += [
        "",
        f"使用动态内参后，{int((position_delta < 0).sum())}/9 组的位置误差低于固定内参重放，均值低 {abs(position_delta.mean()):.3f} mm。"
        f"姿态误差均值高 {attitude_delta.mean():.3f}°，{int((attitude_delta < 0).sum())}/9 组的姿态误差降低。",
        f"动态内参误差最大的为第 {int(worst.group_id)} 组，位置误差 {worst[POSITION]:.3f} mm，姿态误差 {worst[ATTITUDE]:.3f}°。",
        "",
        "三路红外在全部 9 组均显示 DISCONNECTED，有效红外平面为 0 组。",
        "",
        "![动态和固定内参定位误差及对焦读数](error_dashboard.png)",
        "",
        "## 内参与重放口径",
        "",
        "逐帧使用的内参与 CSV 和同名快照 JSON 一致。对焦读数 304 有 1 组，336 有 5 组，368 有 2 组，432 有 1 组。"
        "304 是已验证的标定点，336 使用尚未通过独立验证的区间插值，368 是尚未通过独立验证的标定点，432 使用线性外推。",
        "",
        "两种重放共用原始 CSV 角点、快照 JSON 的检测中心、同一标签地图及加权 PnP。动态重放先复现了 9 组在线位姿，再将 K 和畸变系数替换为此前使用的固定标定参数。"
        "位置取三维光心距离，姿态取相对旋转角。动捕 MAT 缺少逐组时间戳，评价依赖视觉与动捕按组采集时姿态一致。",
        "",
        "机翼的 3 点匹配遍历全部排列，机器人点去重后遍历全部匹配，选择满足基准平面在前方且残差最小的位姿。"
        "若仅有 3 个机器人标记，程序遍历 4 选 3 的所有子集和排列。本次纳入的各组均有 4 个唯一机器人标记。",
        "",
        "## 输出文件",
        "",
        "- [逐组误差、对焦读数和固定内参重放](per_group_errors.csv)",
        "- [汇总指标](summary_metrics.csv)和[动捕与视觉位姿](poses.json)",
        "- 在本目录运行 `python analyze_dynamic_focus.py` 可重算。",
    ]
    if scene:
        lines += ["", "![机翼模型与第 9 组位姿对照](scene_comparison.png)"]
    (output / "分析报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-3d", action="store_true")
    args = parser.parse_args()
    rows = pd.read_csv(ROOT / "positioning_records.csv")
    sidecars = read_sidecars(rows)
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    self_check_registration()
    frame, summary, poses, _ = analyze(output, experiment=ROOT, marker_filename=MARKER_FILE,
                                       excluded_mocap_groups=(EXCLUDED_MOCAP_GROUP,))
    comparison, inputs = compare_intrinsics(poses, rows, sidecars)
    frame = frame.merge(comparison, on="group_id", validate="one_to_one")
    summary.update(excluded_mocap_groups=[EXCLUDED_MOCAP_GROUP],
                   fixed_replay={key: summarize(frame[f"fixed_{key}"]) for key in METRICS})
    frame.to_csv(output / "per_group_errors.csv", index=False, encoding="utf-8-sig")
    metrics = [dict(method=method, metric=key, **results[key])
               for method, results in (("dynamic_recorded", summary), ("fixed_replay", summary["fixed_replay"]))
               for key in (*METRICS, "ir_distance_error_abs_mm", "ir_normal_error_deg") if key in results]
    pd.DataFrame(metrics).to_csv(output / "summary_metrics.csv", index=False, encoding="utf-8-sig")
    save_json(output / "summary.json", summary)
    manifest = json.loads((output / "input_manifest.json").read_text(encoding="utf-8"))
    manifest.update({p.relative_to(PROJECT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    save_json(output / "input_manifest.json", manifest)
    plot_errors(frame, output)
    if not args.skip_3d:
        make_3d(poses, int(frame.loc[frame[POSITION].idxmax(), "group_id"]), output)
    report(frame, summary, output, not args.skip_3d)
    print(pd.DataFrame(metrics).to_string(index=False))


if __name__ == "__main__":
    main()
