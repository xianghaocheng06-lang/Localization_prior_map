"""Analyze position and attitude error directions in the third 2026-11-11 test."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
sys.path[:0] = [str(PROJECT), str(ROOT.parent / "26.11.11红外+相机精度测试")]
from analyze_20261111 import analyze, csv_transform, inverse_rigid, make_3d, save_json, self_check_registration
from localization.positioning_core import VisionEstimator, load_config

AXES = "xyz"
WORLD = [f"error_w_{axis}_mm" for axis in AXES]
CAMERA = [f"error_c_{axis}_mm" for axis in AXES]
ATTITUDE = [f"rotation_vector_w_{axis}_deg" for axis in AXES]


def direction_stats(vectors):
    values = np.asarray(vectors)
    bias = values.mean(0)
    centered = values - bias
    norms = np.linalg.norm(values, axis=1)
    residual_norms = np.linalg.norm(centered, axis=1)
    mse = np.mean(norms ** 2)
    mean_unit = (values[norms > 0] / norms[norms > 0, None]).mean(0)
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, axes = np.linalg.eigh(covariance)
    dominant = axes[:, -1]
    if dominant[np.argmax(np.abs(dominant))] < 0:
        dominant = -dominant
    leave_one_out = values - (values.sum(0) - values) / (len(values) - 1)
    assert np.isclose(np.mean(residual_norms ** 2) + bias @ bias, mse)
    return dict(n=len(values), mean_xyz=bias, median_xyz=np.median(values, axis=0),
                negative=(values < 0).sum(0), positive=(values > 0).sum(0),
                minus_x_plus_y=int(((values[:, 0] < 0) & (values[:, 1] > 0)).sum()),
                mean_unit_vector=mean_unit, direction_concentration=np.linalg.norm(mean_unit),
                mean_error=norms.mean(), median_error=np.median(norms), rmse=np.sqrt(mse),
                centered_mean_error=residual_norms.mean(), centered_rmse=np.sqrt(np.mean(residual_norms ** 2)),
                leave_one_out_mean_error=np.linalg.norm(leave_one_out, axis=1).mean(),
                bias_fraction_of_mse=(bias @ bias) / mse, sample_std_xyz=values.std(0, ddof=1),
                pca_variance_fraction=eigenvalues[::-1] / eigenvalues.sum(), dominant_pca_axis=dominant)


def per_capture_errors(rows, poses):
    references = {pose["group_id"]: pose["T_W_C_gt_mm"] for pose in poses}
    records = []
    for row in rows.to_dict("records"):
        gid = row["group_id"]
        reference = references[gid]
        estimate = csv_transform(row, "T_W_C")
        assert row["vision_status"] == "VALID" and np.isfinite(estimate).all()
        world_error = estimate[:3, 3] - reference[:3, 3]
        camera_error = reference[:3, :3].T @ world_error
        rotation = Rotation.from_matrix(estimate[:3, :3] @ reference[:3, :3].T)
        rotation_vector = rotation.as_rotvec(degrees=True)
        records.append(dict(group_id=gid, capture=row["capture"], snapshot_at=row["snapshot_at"],
                            focus_position=row["focus_position"], known_tag_ids=row["known_tag_ids"],
                            reprojection_rmse_px=row["reprojection_rmse_px"],
                            position_error_mm=np.linalg.norm(world_error),
                            rotation_error_deg=np.degrees(rotation.magnitude()),
                            **dict(zip(WORLD, world_error)), **dict(zip(CAMERA, camera_error)),
                            **dict(zip(ATTITUDE, rotation_vector))))
    return pd.DataFrame(records)


def diagnose_group_two(row, reference, config=None):
    """Compare the two planar candidates without changing any measured pose."""
    metadata = []
    for path in (ROOT / "camera_snapshots").glob("2_*.json"):
        info = json.loads(path.read_text(encoding="utf-8"))
        if info["group_id"] == "2" and info["snapshot_at"] == row.snapshot_at:
            metadata.append((path, info))
    if len(metadata) != 1:
        raise ValueError("第 2 组相机快照与 JSON 配对失败")
    path, info = metadata[0]
    if config is None:
        config = load_config(PROJECT / "config/positioning_config.json")
    estimator = VisionEstimator(config)
    matrix, distortion = np.array(info["camera_matrix"]), np.array(info["dist_coeffs"])
    detected = json.loads(row.detected_tag_ids)
    tag_ids = json.loads(row.known_tag_ids)
    assert len(tag_ids) == 1 and detected == info["detected_tag_ids"]
    corners = np.array(json.loads(row.tag_corners_px_json)[detected.index(tag_ids[0])])
    ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(estimator.local, corners, matrix, distortion, flags=cv2.SOLVEPNP_IPPE)
    if ok != 2:
        raise ValueError("第 2 组平面位姿诊断未返回两个候选")
    tag_pose = estimator.layout[str(tag_ids[0])]
    records = []
    for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs), 1):
        transform = np.eye(4)
        transform[:3, :3] = cv2.Rodrigues(rvec)[0]
        transform[:3, 3] = tvec.ravel()
        camera_pose = inverse_rigid(transform @ inverse_rigid(tag_pose))
        camera_pose[:3, 3] *= 1000
        projection = cv2.projectPoints(estimator.local, rvec, tvec, matrix, distortion)[0].reshape(-1, 2)
        rotation = Rotation.from_matrix(camera_pose[:3, :3] @ reference[:3, :3].T)
        records.append(dict(candidate=i, corner_rmse_px=np.sqrt(np.mean(np.sum((projection - corners) ** 2, axis=1))),
                            position_error_mm=np.linalg.norm(camera_pose[:3, 3] - reference[:3, 3]),
                            attitude_error_deg=np.degrees(rotation.magnitude())))
    return pd.DataFrame(records), path


def plot_directions(primary, extra, stats, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": .18})
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.1), layout="constrained")
    colors = ("#195f81", "#ae561c", "#257d69")
    ax = axes[0, 0]
    for name, color, column in zip("XYZ", colors, WORLD):
        ax.plot(primary.group_id, primary[column], "o-", color=color, label=name)
        ax.scatter(extra.group_id, extra[column], color=color, marker="D", s=50, zorder=4)
    ax.axhline(0, color="#687789", lw=1)
    ax.set(title="逐组有符号位置误差（◆ 为 8B）", xlabel="采集组号", ylabel="视觉减动捕（mm）")
    ax.set_xticks(primary.group_id)
    ax.legend(ncol=3)

    for ax, fields, labels in ((axes[0, 1], WORLD[:2], "XY"), (axes[1, 0], (WORLD[0], WORLD[2]), "XZ")):
        ax.axhline(0, color="#798693", lw=1)
        ax.axvline(0, color="#798693", lw=1)
        ax.scatter(primary[fields[0]], primary[fields[1]], color=colors[0], s=46, label="首帧方案", zorder=3)
        ax.scatter(extra[fields[0]], extra[fields[1]], color=colors[1], marker="D", s=55, label="8B", zorder=4)
        for row in pd.concat([primary.loc[primary.group_id.isin([2, 8])], extra]).to_dict("records"):
            label = row["capture"] if row["group_id"] == 8 else "2"
            ax.annotate(label, (row[fields[0]], row[fields[1]]), xytext=(5, 4), textcoords="offset points")
        ax.set(title=f"机翼系 {labels} 误差投影", xlabel=f"{labels[0]}（mm）", ylabel=f"{labels[1]}（mm）")
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(loc="upper right" if labels == "XY" else "lower right", fontsize=9)

    ax = axes[1, 1]
    locations = np.arange(3)
    for i, (label, color, values) in enumerate((
        ("首帧方案", colors[0], stats["first"]["position_W"]),
        ("替换为 8B", colors[1], stats["second"]["position_W"]),
    )):
        offset = (-.5 + i) * .36
        ax.bar(locations + offset, values["positive"], color=color, width=.32, label=label)
        ax.bar(locations + offset, -values["negative"], color=color, width=.32)
    ax.axhline(0, color="#687789", lw=1)
    ax.set(title="误差方向计数", ylabel="组数（上为正向，下为负向）", ylim=(-10.5, 10.5))
    ax.set_xticks(locations, ["X", "Y", "Z"])
    ax.set_yticks([-10, -5, 0, 5, 10], ["10", "5", "0", "5", "10"])
    ax.legend(loc="upper left", ncol=2)
    fig.suptitle("2026 年 11 月 11 日第三次测试误差方向", fontsize=18, weight="bold")
    fig.savefig(output / "error_directions.png", dpi=180)
    plt.close(fig)


def vector_text(values):
    return "(" + ", ".join(f"{x:+.3f}" for x in values) + ")"


def write_report(data, stats, candidates, corner_shifts, output, scene):
    first, second = stats["first"], stats["second"]
    a, b = first["position_W"], second["position_W"]
    largest = data.loc[data.capture == "2"].iloc[0]
    lines = [
        "# 2026 年 11 月 11 日第三次定位测试",
        "",
        "动捕 MAT 有第 1–10 组。视觉 CSV 有 11 行，其中第 8 组保存了 8A（18:21:57，对焦 368）和 8B（18:22:20，对焦 496）。"
        "主统计使用采集顺序中的首帧 8A，另以 8B 替换重算。两个方案各含 10 个独立组号。"
        "第 7 组机翼标记用于建立统一坐标基准，每组机器人位姿使用对应组的动捕点。",
        "",
        "| 第 8 组取帧 | 平均位置误差（mm） | 位置 RMSE（mm） | 平均姿态误差（°） | 位置分量中位数 XYZ（mm） |",
        "|---|---:|---:|---:|---|",
        f"| 8A | {a['mean_error']:.3f} | {a['rmse']:.3f} | {first['attitude_W']['mean_error']:.3f} | {vector_text(a['median_xyz'])} |",
        f"| 8B | {b['mean_error']:.3f} | {b['rmse']:.3f} | {second['attitude_W']['mean_error']:.3f} | {vector_text(b['median_xyz'])} |",
        "",
        "## 方向一致性",
        "",
        f"机翼系位置误差按视觉减动捕计算。两种取帧方式下，Y 向误差均有 {a['positive'][1]}/10 组为正，"
        f"X 向误差分别有 {a['negative'][0]}/10、{b['negative'][0]}/10 组为负。"
        f"同时满足 −X、+Y 的组数分别是 {a['minus_x_plus_y']}/10 和 {b['minus_x_plus_y']}/10。"
        "两种方案的分量中位数都指向 −X、+Y、+Z，Z 的中位偏移约 1.14 mm。",
        "",
        f"按单位误差方向计算平均合向量的长度，8A 和 8B 方案分别为 {a['direction_concentration']:.3f} 和 {b['direction_concentration']:.3f}（范围 0–1）。"
        f"直接按大小求均值则得到 {vector_text(a['mean_xyz'])} mm 和 {vector_text(b['mean_xyz'])} mm。"
        "少数大误差显著拉动均值，导致平均向量与多数样本的方向不同。",
        "",
        f"主方案扣除本批样本的平均平移后，RMSE 从 {a['rmse']:.3f} mm 降至 {a['centered_rmse']:.3f} mm，"
        f"平均误差却从 {a['mean_error']:.3f} mm 升至 {a['centered_mean_error']:.3f} mm。"
        f"替换为 8B 后，平均误差也从 {b['mean_error']:.3f} mm 升至 {b['centered_mean_error']:.3f} mm。"
        "本批可见方向倾向，单一固定平移对整体误差的解释有限。",
        "",
        f"把各组位置误差转到其动捕相机坐标系后，主方案的 X 负向为 {first['position_C']['negative'][0]}/10，"
        f"Y 负向为 {first['position_C']['negative'][1]}/10，光轴 Z 负向为 {first['position_C']['negative'][2]}/10。"
        f"机翼系相对姿态的旋转向量 Y 分量在 8A 和 8B 方案中分别有 {first['attitude_W']['positive'][1]}/10 和 "
        f"{second['attitude_W']['positive'][1]}/10 组为正。旋转向量分量描述绕对应轴的旋转方向。",
        "",
        "![机翼坐标系误差方向和符号统计](error_directions.png)",
        "",
        "## 大误差记录",
        "",
        f"第 2 组的位置误差为 {largest.position_error_mm:.3f} mm，姿态误差为 {largest.rotation_error_deg:.3f}°，"
        f"而在线重投影 RMSE 为 {largest.reprojection_rmse_px:.3f} px。"
        "这次各帧用于定位的标签均只有机翼上的第 5 号 AprilTag。",
        "",
        "对第 2 组保留的四角点和逐帧内参做平面 PnP 双候选诊断，得到以下结果。"
        "角点 RMSE 使用四角点计算，在线 RMSE 还可能包含标签中心。诊断候选未用于主统计。",
        "",
        "| 候选 | 四角点重投影 RMSE（px） | 相对动捕位置误差（mm） | 姿态误差（°） |",
        "|---|---:|---:|---:|",
    ]
    for row in candidates.itertuples():
        lines.append(f"| {row.candidate} | {row.corner_rmse_px:.3f} | {row.position_error_mm:.3f} | {row.attitude_error_deg:.3f} |")
    lines += [
        "",
        "第 2 组存在两个能较好解释角点的姿态，重投影残差较小的一支与动捕的差距反而较大。"
        "8A 和 8B 对同一组动捕的位置误差分别为 4.028 mm 和 37.035 mm。"
        f"两帧对应角点相差 {corner_shifts.min():.1f}–{corner_shifts.max():.1f} px，8B 同时使用外推内参。"
        "MAT 文件没有逐组时间戳，无法确认哪帧与第 8 组动捕同步。",
        "",
        "三路红外在全部记录均显示断开，有效红外平面为 0 组。",
        "",
        "## 输出文件",
        "",
        "- [11 帧位置和姿态分量](directional_errors.csv)",
        "- [两种第 8 组取帧方案的统计](direction_summary.json)",
        "- [第 2 组平面候选](planar_pose_candidates.csv)",
        "- [误差方向一致性专项分析](方向一致性分析.md)（`python analyze_direction_consistency.py`）",
        "- 在本目录运行 `python analyze_error_directions.py` 可重算。",
    ]
    if scene:
        lines += ["", "![第 2 组动捕与视觉位姿的三维对照](scene_comparison.png)"]
    (output / "分析报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-3d", action="store_true")
    args = parser.parse_args()
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    rows = pd.read_csv(ROOT / "positioning_records.csv")
    assert rows.group_id.tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 10]
    rows["capture"] = rows.group_id.astype(str)
    rows.loc[rows.group_id == 8, "capture"] = ["8A", "8B"]
    self_check_registration()
    base, summary, poses, _ = analyze(output, experiment=ROOT, marker_filename="markerPos_test_3.mat",
                                     positioning_rows=rows.loc[rows.capture != "8B"])
    data = per_capture_errors(rows, poses)
    primary, secondary = (data.loc[data.capture != skip] for skip in ("8B", "8A"))
    np.testing.assert_allclose(primary.position_error_mm, base.camera_position_error_mm)
    np.testing.assert_allclose(primary.rotation_error_deg, base.camera_rotation_error_deg)
    stats = {name: dict(position_W=direction_stats(frame[WORLD]), position_C=direction_stats(frame[CAMERA]),
                        attitude_W=direction_stats(frame[ATTITUDE]), captures=frame.capture.tolist())
             for name, frame in (("first", primary), ("second", secondary))}
    stats["ir_valid_groups"] = summary["ir_groups"]
    stats["wing_reference_group"] = summary["wing_reference_group"]
    candidates, metadata = diagnose_group_two(rows.loc[rows.group_id == 2].iloc[0], poses[1]["T_W_C_gt_mm"])
    corners_8a, corners_8b = (np.array(json.loads(corners)[0]) for corners in rows.loc[rows.group_id == 8, "tag_corners_px_json"])
    corner_shifts = np.linalg.norm(corners_8a - corners_8b, axis=1)
    data.to_csv(output / "directional_errors.csv", index=False, encoding="utf-8-sig")
    candidates.to_csv(output / "planar_pose_candidates.csv", index=False, encoding="utf-8-sig")
    save_json(output / "direction_summary.json", stats)
    manifest = json.loads((output / "input_manifest.json").read_text(encoding="utf-8"))
    manifest.update({p.relative_to(PROJECT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in (metadata, PROJECT / "config/positioning_config.json", PROJECT / "config/tag_layout.json")})
    save_json(output / "input_manifest.json", manifest)
    plot_directions(primary, data.loc[data.capture == "8B"], stats, output)
    if not args.skip_3d:
        make_3d(poses, 2, output)
    write_report(data, stats, candidates, corner_shifts, output, not args.skip_3d)
    print((output / "分析报告.md").resolve())


if __name__ == "__main__":
    main()
