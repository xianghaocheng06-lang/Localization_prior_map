"""Replay the third test's saved observations with its single-tag layout."""

import argparse
import hashlib
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from analyze_error_directions import (
    ATTITUDE, CAMERA, PROJECT, ROOT, WORLD, analyze, csv_transform,
    diagnose_group_two, direction_stats, make_3d, per_capture_errors,
    plot_directions, save_json, self_check_registration, vector_text,
)
from analyze_20261111 import summarize
from localization.positioning_core import VisionEstimator, load_config, load_tag_layout, matrix_fields, robot_pose

ERROR_METRICS = ["position_error_mm", "rotation_error_deg"]
REPLAY_LAYOUT = ROOT / "replay_tag_layout.json"


def replay(rows, config):
    estimator = VisionEstimator(config)
    # The PNGs are already rotated. Use the K and D stored for each exposure.
    estimator.config = {**config, "flip_camera_180": False}
    estimator.focus_map = None
    detector = estimator.detector
    snapshots = ROOT / "camera_snapshots"
    metadata_paths = sorted(snapshots.glob("*.json"))
    if {path.with_suffix(".png") for path in metadata_paths} != set(snapshots.glob("*.png")):
        raise ValueError("快照 PNG 和 JSON 未一一对应")
    metadata = {}
    for path in metadata_paths:
        info = json.loads(path.read_text(encoding="utf-8"))
        key = (int(info["group_id"]), info["snapshot_at"])
        if key in metadata:
            raise ValueError(f"快照编号和时间重复：{key}")
        metadata[key] = (path.with_suffix(".png"), info)
    if set(metadata) != set(zip(rows.group_id, rows.snapshot_at)) or len(metadata) != len(rows):
        raise ValueError("快照与定位记录未按组号和 snapshot_at 一一匹配")

    updated = rows.copy()
    audit = []
    for index, row in rows.iterrows():
        path, info = metadata[row.group_id, row.snapshot_at]
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or tuple(image.shape[1::-1]) != tuple(config["camera_resolution"]):
            raise ValueError(f"快照不可读或分辨率不符：{path.name}")
        redetected = detector.detect(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        ids = json.loads(row.detected_tag_ids)
        corners = np.asarray(json.loads(row.tag_corners_px_json), dtype=float)
        centers = np.asarray(info["tag_centers_px"], dtype=float)
        if (ids != info["detected_tag_ids"] or corners.shape != (len(ids), 4, 2)
                or centers.shape != (len(ids), 2) or not np.isfinite(corners).all()
                or not np.isfinite(centers).all() or row.focus_position != info["focus_position"]
                or row.vision_status != info["vision_status"]):
            raise ValueError(f"快照与定位记录的相机数据不一致：{path.name}")
        estimator.K = np.asarray(info["camera_matrix"], dtype=float)
        estimator.D = np.asarray(info["dist_coeffs"], dtype=float)
        np.testing.assert_allclose(estimator.K[[0, 1], [0, 1]], [row.camera_fx_px, row.camera_fy_px])
        tags = [SimpleNamespace(tag_id=tag_id, corners=corner, center=center)
                for tag_id, corner, center in zip(ids, corners, centers)]
        estimator.detector = SimpleNamespace(detect=lambda _gray, tags=tags: tags)
        result = estimator.process(image)
        if result.camera_pose is None or result.status != "VALID":
            raise ValueError(f"快照 {path.name} 求解失败：{result.status}")
        robot = robot_pose(result.camera_pose, config["T_R_C"])
        for prefix, pose in (("T_W_C", result.camera_pose), ("T_W_R", robot)):
            updated.loc[index, matrix_fields(prefix)] = pose.ravel()
        updated.loc[index, ["vision_status", "known_tag_ids", "reprojection_rmse_px"]] = [
            result.status, json.dumps(result.known_ids), result.reprojection_rmse_px,
        ]
        old = csv_transform(row, "T_W_C")
        shift = result.camera_pose[:3, 3] * 1000 - old[:3, 3]
        rotation = Rotation.from_matrix(result.camera_pose[:3, :3] @ old[:3, :3].T)
        audit.append(dict(
            capture=row.capture, snapshot=path.name, png_redetected_ids=json.dumps([int(t.tag_id) for t in redetected]),
            center_inlier_ids=json.dumps(result.center_inlier_ids), original_reprojection_rmse_px=row.reprojection_rmse_px,
            layout_rotation_change_deg=np.degrees(rotation.magnitude()),
            **{f"layout_shift_w_{axis}_mm": value for axis, value in zip("xyz", shift)},
        ))
    return updated, pd.DataFrame(audit), [*metadata_paths, *(path.with_suffix(".png") for path in metadata_paths)]


def filter_outlier_groups(primary):
    errors = primary[ERROR_METRICS]
    q1, q3 = errors.quantile([.25, .75]).to_numpy()
    upper = pd.Series(q3 + 1.5 * (q3 - q1), index=ERROR_METRICS)
    exceeded = errors.gt(upper)
    retained = primary.loc[~exceeded.any(axis=1)]
    excluded = [
        dict(capture=primary.loc[index, "capture"], exceeded=flags.index[flags].tolist(),
             **errors.loc[index].to_dict())
        for index, flags in exceeded.loc[exceeded.any(axis=1)].iterrows()
    ]
    return retained, dict(
        upper_limits=upper.to_dict(), excluded=excluded, captures=retained.capture.tolist(),
        **{metric: summarize(retained[metric]) for metric in ERROR_METRICS},
        median_error_w_xyz_mm=retained[WORLD].median().to_numpy(),
    )


def plot_filtered_errors(retained, output):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), layout="constrained")
    for ax, metric, ylabel, color in zip(
        axes, ERROR_METRICS, ["位置误差（mm）", "姿态误差（°）"], ["#195f81", "#ae561c"],
    ):
        values = retained[metric]
        bars = ax.bar(retained.capture, values, color=color, width=.62)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
        ax.axhline(values.mean(), color="#687789", ls="--", lw=1, label=f"均值 {values.mean():.2f}")
        ax.set(xlabel="采集组号", ylabel=ylabel, ylim=(0, 1.28 * values.max()))
        ax.legend(loc="upper right")
        ax.set_axisbelow(True)
    fig.suptitle("剔除离群组后的相机误差", fontsize=16, weight="bold")
    fig.savefig(output / "filtered_errors.png", dpi=180)
    plt.close(fig)


def write_report(output, data, stats, candidates, config, layout, scene):
    shift_cols = [f"layout_shift_w_{axis}_mm" for axis in "xyz"]
    shifts = data[shift_cols].to_numpy()
    tag_position = layout["5"][:3, 3] * 1000
    largest = data.loc[data.position_error_mm.idxmax()]
    usable_png = sum(bool(set(json.loads(ids)) & set(map(int, layout))) for ids in data.png_redetected_ids)
    filtered = stats["filtered_primary"]
    position, attitude = (filtered[metric] for metric in ERROR_METRICS)
    limits = filtered["upper_limits"]
    lines = [
        "# 2026 年 11 月 11 日第三次定位测试 · 快照重算",
        "",
        "## 剔除离群组后的统计",
        "",
        f"以 8A 主方案的 10 组为样本，分别计算位置和姿态误差的上界 `Q3 + 1.5 × (Q3 − Q1)`。"
        f"位置上界为 {limits['position_error_mm']:.3f} mm，姿态上界为 {limits['rotation_error_deg']:.3f}°。"
        "任一项超过上界即剔除整组，上界只按原样本计算一次。",
        "",
        "| 剔除组 | 位置误差（mm） | 姿态误差（°） | 超过上界的指标 |",
        "|---|---:|---:|---|",
    ]
    labels = dict(zip(ERROR_METRICS, ["位置", "姿态"]))
    for row in filtered["excluded"]:
        reasons = "、".join(labels[metric] for metric in row["exceeded"])
        lines.append(f"| {row['capture']} | {row['position_error_mm']:.3f} | {row['rotation_error_deg']:.3f} | {reasons} |")
    lines += [
        "",
        f"保留 {'、'.join(filtered['captures'])} 共 {position['n']} 组。平均位置误差 **{position['mean']:.3f} mm**，"
        f"位置 RMSE **{position['rmse']:.3f} mm**，中位数 {position['median']:.3f} mm。"
        f"平均姿态误差 **{attitude['mean']:.3f}°**，姿态 RMSE **{attitude['rmse']:.3f}°**。",
        "",
        f"保留样本的机翼系 XYZ 误差分量中位数为 {vector_text(filtered['median_error_w_xyz_mm'])} mm。"
        "以上指标仅统计保留样本。8B 为第 8 组替代帧，参与下方全样本敏感性对照。",
        "",
        "[保留组误差明细](filtered_errors.csv) · [筛选规则与统计](direction_summary.json)",
        "",
        "![保留组的相机位置和姿态误差](filtered_errors.png)",
        "",
        "## 重算方法与全样本对照",
        "",
        f"本批回放布局 `{config['tag_layout_file'].name}` 中 5 号 AprilTag 的机翼系坐标为 "
        f"({tag_position[0]:.1f}, {tag_position[1]:.1f}, {tag_position[2]:.1f}) mm，"
        f"边长为 {config['tag_size_m'] * 1000:g} mm，本批 11 帧均使用 5 号标签。PNG 上叠加了标签框、文字和坐标轴，"
        f"直接检测有 {usable_png}/11 帧检出布局内的标签。位姿重算使用同帧 CSV 中标注前检测的角点、"
        "快照 JSON 中的检测中心及逐帧内参和畸变系数。按组号和 `snapshot_at` 匹配 PNG、JSON 与 CSV，"
        "以现有程序的四角点 PnP 和中心点加权优化重新求解。",
        "",
        "动捕机翼坐标基准沿用第 7 组，逐组重建机器人位姿，三个机器人标记的组遍历 4 选 3 的对应关系。"
        "位置误差为视觉光心与动捕光心的距离，姿态误差为相对旋转角。第 8 组先保存 8A（18:21:57，对焦 368），"
        "后保存 8B（18:22:20，对焦 496，使用外推内参）。主统计沿用 8A，并列给出 8B 替换结果。"
        "动捕文件缺少逐组时间戳，两帧与动捕的同步关系无法验证。",
        "",
        "| 第 8 组取帧 | 组数 | 平均位置误差旧 → 新（mm） | 位置 RMSE 旧 → 新（mm） | 位置中位数旧 → 新（mm） | 平均姿态误差旧 → 新（°） |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for capture, key in (("8A（主统计）", "first"), ("8B", "second")):
        old, new = stats[key]["original_position_W"], stats[key]["position_W"]
        old_r, new_r = stats[key]["original_attitude_W"], stats[key]["attitude_W"]
        lines.append(f"| {capture} | {new['n']} | {old['mean_error']:.3f} → {new['mean_error']:.3f} | "
                     f"{old['rmse']:.3f} → {new['rmse']:.3f} | {old['median_error']:.3f} → {new['median_error']:.3f} | "
                     f"{old_r['mean_error']:.3f} → {new_r['mean_error']:.3f} |")
    primary = stats["first"]["position_W"]
    secondary = stats["second"]["position_W"]
    lines += [
        "",
        f"相对原记录，视觉光心的逐帧坐标变化一致，平均为 {vector_text(shifts.mean(0))} mm（机翼系 XYZ）。"
        f"姿态变化最大为 {data.layout_rotation_change_deg.max():.2e}°，重投影 RMSE 最大变化为 "
        f"{(data.reprojection_rmse_px - data.original_reprojection_rmse_px).abs().max():.2e} px。",
        "",
        "## 方向与异常组",
        "",
        f"主统计的有符号误差分量中位数从 {vector_text(stats['first']['original_position_W']['median_xyz'])} mm "
        f"变为 {vector_text(primary['median_xyz'])} mm。机翼系 X、Y 分别有 {primary['negative'][0]}/10、"
        f"{primary['negative'][1]}/10 组为负。换用 8B 后，分量中位数为 {vector_text(secondary['median_xyz'])} mm。",
        "",
        f"第 {largest.group_id} 组的位置误差最大，为 {largest.position_error_mm:.3f} mm，"
        f"姿态误差为 {largest.rotation_error_deg:.3f}°，重投影 RMSE 为 {largest.reprojection_rmse_px:.3f} px。"
        "用该组角点和新版布局做 IPPE 双候选诊断，残差较低的解与动捕相差更大。两支候选仅供诊断，主统计沿用定位程序的求解结果。",
        "",
        "| 第 2 组 IPPE 候选 | 四角点重投影 RMSE（px） | 位置误差（mm） | 姿态误差（°） |",
        "|---|---:|---:|---:|",
    ]
    for row in candidates.itertuples():
        lines.append(f"| {row.candidate} | {row.corner_rmse_px:.3f} | {row.position_error_mm:.3f} | {row.attitude_error_deg:.3f} |")
    lines += [
        "",
        f"有效红外平面为 {len(stats['ir_valid_groups'])} 组。本批无法统计红外距离和法向误差。",
        "",
        "## 逐帧误差",
        "",
        "| 帧 | 位置误差旧 → 新（mm） | 新误差 ΔX / ΔY / ΔZ（mm） | 姿态误差（°） |",
        "|---|---:|---:|---:|",
    ]
    for row in data.to_dict("records"):
        lines.append(f"| {row['capture']} | {row['original_position_error_mm']:.3f} → {row['position_error_mm']:.3f} | "
                     f"{vector_text([row[column] for column in WORLD])} | {row['rotation_error_deg']:.3f} |")
    lines += [
        "",
        "[11 帧重算明细](directional_errors.csv) · [重算统计](direction_summary.json) · "
        "[主方案位姿](poses.json) · [第 2 组双候选](planar_pose_candidates.csv) · [输入校验清单](input_manifest.json)",
        "",
        "在本测试目录运行 `python replay_snapshots.py` 可重算。",
        "",
        "![机翼坐标系误差方向和符号统计](error_directions.png)",
    ]
    if scene:
        lines += ["", "![第 2 组动捕与视觉位姿的三维对照](scene_comparison.png)"]
    (output / "快照重算误差分析.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-3d", action="store_true")
    args = parser.parse_args()
    config_path = PROJECT / "config/positioning_config.json"
    config = load_config(config_path)
    config["tag_layout_file"] = REPLAY_LAYOUT
    rows = pd.read_csv(ROOT / "positioning_records.csv").sort_values(["group_id", "snapshot_at"])
    if rows.group_id.tolist() != [1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 10]:
        raise ValueError("定位记录应包含第 1–10 组，其中第 8 组有两帧")
    rows["capture"] = rows.group_id.astype(str)
    rows.loc[rows.group_id == 8, "capture"] = ["8A", "8B"]
    updated, audit, inputs = replay(rows, config)

    output = ROOT / "results" / "snapshot_replay"
    output.mkdir(parents=True, exist_ok=True)
    self_check_registration()
    base, summary, poses, _ = analyze(output, experiment=ROOT, marker_filename="markerPos_test_3.mat",
                                     positioning_rows=updated.loc[updated.capture != "8B"])
    old = per_capture_errors(rows, poses)
    data = per_capture_errors(updated, poses).merge(audit, on="capture", validate="one_to_one")
    metrics = ["position_error_mm", "rotation_error_deg", *WORLD, *ATTITUDE]
    data = data.merge(old[["capture", *metrics]].rename(columns={key: f"original_{key}" for key in metrics}),
                      on="capture", validate="one_to_one")
    primary = data.loc[data.capture != "8B"]
    np.testing.assert_allclose(primary.position_error_mm, base.camera_position_error_mm)
    np.testing.assert_allclose(primary.rotation_error_deg, base.camera_rotation_error_deg)
    stats = {}
    for name, skipped in (("first", "8B"), ("second", "8A")):
        frame, original = (source.loc[source.capture != skipped] for source in (data, old))
        stats[name] = dict(position_W=direction_stats(frame[WORLD]), position_C=direction_stats(frame[CAMERA]),
                           attitude_W=direction_stats(frame[ATTITUDE]), captures=frame.capture.tolist(),
                           original_position_W=direction_stats(original[WORLD]),
                           original_attitude_W=direction_stats(original[ATTITUDE]))
    stats.update(ir_valid_groups=summary["ir_groups"], wing_reference_group=summary["wing_reference_group"])
    retained, stats["filtered_primary"] = filter_outlier_groups(primary)
    candidates, _ = diagnose_group_two(
        updated.loc[updated.group_id == 2].iloc[0], poses[1]["T_W_C_gt_mm"], config,
    )
    data.to_csv(output / "directional_errors.csv", index=False, encoding="utf-8-sig")
    retained.to_csv(output / "filtered_errors.csv", index=False, encoding="utf-8-sig")
    candidates.to_csv(output / "planar_pose_candidates.csv", index=False, encoding="utf-8-sig")
    save_json(output / "direction_summary.json", stats)
    manifest_path = output / "input_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for path in [*inputs, config_path, config["tag_layout_file"], PROJECT / "localization/positioning_core.py",
                 ROOT / "replay_snapshots.py", ROOT / "analyze_error_directions.py"]:
        manifest[path.relative_to(PROJECT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    save_json(manifest_path, manifest)
    plot_directions(primary, data.loc[data.capture == "8B"], stats, output)
    plot_filtered_errors(retained, output)
    if not args.skip_3d:
        make_3d(poses, 2, output)
    layout = load_tag_layout(config["tag_layout_file"])
    write_report(output, data, stats, candidates, config, layout, not args.skip_3d)
    print(output / "快照重算误差分析.md")


if __name__ == "__main__":
    main()
