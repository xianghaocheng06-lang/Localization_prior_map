"""Process the 2026-11-11 localization experiment and render its error plots."""

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
PRIOR_ANALYSIS = ROOT.parent / "26.11.1红外+相机精度测试"
sys.path.insert(0, str(PRIOR_ANALYSIS))
from analyze_localization import csv_transform, save_json, self_check_registration
from marker_registration import (
    PLANE_Z_MM, ROBOT_MARKERS_MM, T_R_C_MM, WING_MARKERS_MM,
    enumerate_fits, faces_reference_plane, inverse_rigid, load_marker_groups, transform_points,
)

WING_REFERENCE_GROUP = 7
METRICS = {
    "camera_position_error_mm": ("相机位置误差", "mm"),
    "camera_rotation_error_deg": ("相机姿态误差", "°"),
    "ir_distance_error_abs_mm": ("红外距离绝对误差", "mm"),
    "ir_normal_error_deg": ("红外法向夹角误差", "°"),
}


def summarize(series):
    values = series.dropna().to_numpy(float)
    result = dict(n=len(values), mean=None, median=None, rmse=None, p95=None, maximum=None)
    if len(values):
        result.update(mean=values.mean(), median=np.median(values),
                      rmse=np.sqrt(np.mean(values ** 2)), maximum=values.max())
    if len(values) > 1:
        result["p95"] = np.percentile(values, 95)
    return result


def infrared_comparison(row, robot_gt):
    """Compare the measured plane with z=-75 mm in the robot frame."""
    normal = row[["plane_nx_R", "plane_ny_R", "plane_nz_R"]].to_numpy(float)
    norm = np.linalg.norm(normal)
    assert np.isfinite(norm) and norm > 0
    distance = float(row.plane_distance_mm)
    offset = row.plane_d_R_m * 1000 / norm
    normal /= norm
    gt_normal = robot_gt[2, :3]
    alignment = float(normal @ gt_normal)
    angle = np.degrees(np.arccos(np.clip(abs(alignment), 0, 1)))
    if alignment < 0:
        normal, offset = -normal, -offset
    points = np.array([[row[f"s{i}_point_{axis}_m"] * 1000 for axis in "xyz"] for i in (1, 2, 3)])
    assert np.allclose(points @ normal + offset, 0, atol=1e-5)
    assert np.isclose(abs(offset), distance, atol=1e-5, rtol=0)
    gt_origin = robot_gt[:3, 3]
    gt_distance = abs(gt_origin[2] - PLANE_Z_MM)
    normal_w = robot_gt[:3, :3] @ normal
    return dict(ir_distance_gt_mm=gt_distance, ir_distance_measured_mm=distance,
                ir_distance_error_signed_mm=distance - gt_distance,
                ir_distance_error_abs_mm=abs(distance - gt_distance), ir_normal_error_deg=angle), dict(
        ir_points_W_mm=transform_points(robot_gt, points),
        ir_normal_W=normal_w, ir_projection_W_mm=gt_origin - offset * normal_w,
        ir_true_projection_W_mm=np.array([*gt_origin[:2], PLANE_Z_MM]))


def analyze(output, *, experiment=ROOT, marker_filename="markerPos_test_1.mat",
            excluded_mocap_groups=(), positioning_rows=None):
    rows = pd.read_csv(experiment / "positioning_records.csv") if positioning_rows is None else positioning_rows
    marker_path = experiment / marker_filename
    all_groups = load_marker_groups(marker_path)
    missing = set(excluded_mocap_groups).difference(group["mat_row_1based"] for group in all_groups)
    if missing:
        raise ValueError(f"需作废的动捕组未找到：{sorted(missing)}")
    groups = [group for group in all_groups if group["mat_row_1based"] not in excluded_mocap_groups]
    if rows.group_id.tolist() != [group["mat_row_1based"] for group in groups]:
        raise ValueError("CSV 组号与动捕 MAT 行号不匹配")
    reference_group = next(group for group in groups if group["mat_row_1based"] == WING_REFERENCE_GROUP)
    wing_reference = enumerate_fits(WING_MARKERS_MM, reference_group["wing"])[0]
    wing_from_mocap = inverse_rigid(wing_reference["T"])

    records, poses, candidates, edges = [], [], [], []
    for group, (_, row) in zip(groups, rows.iterrows()):
        gid = int(row.group_id)
        wing = (wing_reference if gid == WING_REFERENCE_GROUP else
                enumerate_fits(WING_MARKERS_MM, group["wing"])[0])
        wing_ordered = np.empty_like(group["wing"])
        wing_ordered[list(wing["mapping"])] = group["wing"]
        for a, b in ((0, 1), (0, 2), (1, 2)):
            nominal = np.linalg.norm(WING_MARKERS_MM[a] - WING_MARKERS_MM[b])
            measured = np.linalg.norm(wing_ordered[a] - wing_ordered[b])
            edges.append(dict(group_id=gid, used_for_reference=gid == WING_REFERENCE_GROUP,
                              edge=f"W{a + 1}-W{b + 1}", nominal_mm=nominal,
                              measured_mm=measured, difference_mm=measured - nominal))

        camera = csv_transform(row, "T_W_C")
        camera_recorded = row.vision_status == "VALID" and np.isfinite(camera).all()
        plane_recorded = bool(row.plane_valid)
        record = dict(
            group_id=gid, mat_row_1based=group["mat_row_1based"], snapshot_at=row.snapshot_at,
            reference_valid=True, wing_reference_group=WING_REFERENCE_GROUP,
            vision_status=row.vision_status, camera_recorded=camera_recorded, camera_evaluated=False,
            ir_plane_recorded=plane_recorded, ir_evaluated=False,
            **{f"s{i}_{field}": row[f"s{i}_{field}"] for i in (1, 2, 3) for field in ("status", "age_s")},
            wing_fit_rms_mm=wing_reference["rms_mm"],
            wing_marker_ids=" ".join(str(i + 1) for i in wing_reference["mapping"]),
            observed_wing_fit_rms_mm=wing["rms_mm"],
            robot_marker_count=len(group["robot"]),
            duplicate_marker_indices=" ".join(map(str, group["duplicate_marker_indices"])),
            **dict.fromkeys(METRICS, np.nan),
        )
        records.append(record)
        robot_fits = enumerate_fits(ROBOT_MARKERS_MM, group["robot"])
        assert len(robot_fits) == 24
        for rank, fit in enumerate(robot_fits, 1):
            fit["robot_gt"] = wing_from_mocap @ fit["T"]
            fit["front"] = faces_reference_plane(fit["robot_gt"])
            candidates.append(dict(group_id=gid, rank=rank, rms_mm=fit["rms_mm"],
                                   robot_marker_ids=" ".join(str(i + 1) for i in fit["mapping"]),
                                   faces_reference_plane=fit["front"]))
        best, runner = [fit for fit in robot_fits if fit["front"]][:2]
        robot_gt = best["robot_gt"]
        camera_gt = robot_gt @ T_R_C_MM
        rotation = camera_gt[:3, :3]
        assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)
        assert np.isclose(np.linalg.det(rotation), 1)
        record.update(robot_marker_ids=" ".join(str(i + 1) for i in best["mapping"]),
                      robot_fit_rms_mm=best["rms_mm"], robot_front_margin_mm=runner["rms_mm"] - best["rms_mm"],
                      ir_distance_gt_mm=abs(robot_gt[2, 3] - PLANE_Z_MM))
        pose = dict(group_id=gid, wing_reference_group=WING_REFERENCE_GROUP, T_W_M_mm=wing_from_mocap,
                    T_W_C_gt_mm=camera_gt, T_W_R_gt_mm=robot_gt,
                    T_W_C_vision_mm=None, T_W_R_vision_mm=None)
        if camera_recorded:
            robot = csv_transform(row, "T_W_R")
            assert np.allclose(csv_transform(row, "T_R_C"), T_R_C_MM, atol=1e-8, rtol=0)
            assert np.allclose(robot @ T_R_C_MM, camera, atol=1e-5, rtol=0)
            difference = camera[:3, 3] - camera_gt[:3, 3]
            delta = camera[:3, :3] @ rotation.T
            angle = np.degrees(Rotation.from_matrix(delta).magnitude())
            record.update(camera_evaluated=True, camera_position_error_mm=np.linalg.norm(difference),
                          camera_rotation_error_deg=angle)
            for i, axis in enumerate("xyz"):
                record[f"camera_gt_{axis}_mm"] = camera_gt[i, 3]
                record[f"camera_vision_{axis}_mm"] = camera[i, 3]
                record[f"camera_error_{axis}_mm"] = difference[i]
            pose.update(T_W_C_vision_mm=camera, T_W_R_vision_mm=robot)
        if plane_recorded and all(row[f"s{i}_status"] == "VALID" for i in (1, 2, 3)):
            metrics, geometry = infrared_comparison(row, robot_gt)
            record.update(ir_evaluated=True, **metrics)
            pose.update(geometry)
        poses.append(pose)

    frame = pd.DataFrame(records)
    edge_frame = pd.DataFrame(edges)
    summary = dict(total=len(frame), wing_reference_group=WING_REFERENCE_GROUP,
                   wing_reference_rms_mm=wing_reference["rms_mm"],
                   excluded_groups=frame.loc[~frame.reference_valid, "group_id"].tolist(),
                   camera_groups=frame.loc[frame.camera_evaluated, "group_id"].tolist(),
                   ir_groups=frame.loc[frame.ir_evaluated, "group_id"].tolist(),
                   **{key: summarize(frame[key]) for key in METRICS})
    frame.to_csv(output / "per_group_errors.csv", index=False, encoding="utf-8-sig")
    edge_frame.to_csv(output / "wing_marker_edge_diagnostics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(candidates).to_csv(output / "robot_correspondence_candidates.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([dict(metric=key, **summary[key]) for key in METRICS]).to_csv(
        output / "summary_metrics.csv", index=False, encoding="utf-8-sig")
    save_json(output / "summary.json", summary)
    save_json(output / "poses.json", poses)
    inputs = [experiment / "README.md", experiment / "positioning_records.csv", marker_path,
              PROJECT / "models/wing_general_simplified.STL", PROJECT / "models/robot_general_simplified.stl"]
    save_json(output / "input_manifest.json", {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                               for path in inputs})
    return frame, summary, poses, edge_frame


def make_error_plot(frame, summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": .18})
    blue, red = "#1b6585", "#ba523e"
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.8), layout="constrained")
    for ax, key in zip(axes[0], list(METRICS)[:2]):
        title, unit = METRICS[key]
        evaluated = frame.loc[frame.camera_evaluated]
        ax.plot(evaluated.group_id, evaluated[key], "o-", color=blue, lw=1.2, ms=5)
        ax.axhline(summary[key]["mean"], ls="--", lw=1.4, color=red,
                   label=f"均值 {summary[key]['mean']:.2f} {unit}")
        ax.set(title=f"{title}（{summary[key]['n']} 组）", xlabel="采集组号",
               ylabel=f"误差（{unit}）", xlim=(.5, len(frame) + .5), ylim=(0, None))
        ax.set_xticks([1, 5, 10, 15, 20, 25])
        ax.legend(loc="upper left", fontsize=9)
    infrared = frame.loc[frame.ir_evaluated].iloc[0]
    group = int(infrared.group_id)
    ax = axes[1, 0]
    bars = ax.bar(["动捕基准", "红外测量"], [infrared.ir_distance_gt_mm, infrared.ir_distance_measured_mm],
                  width=.55, color=[blue, red])
    ax.bar_label(bars, fmt="%.2f", padding=4)
    ax.set(title=f"第 {group} 组红外平面距离", ylabel="垂直距离（mm）",
           ylim=(0, 1.18 * max(infrared.ir_distance_gt_mm, infrared.ir_distance_measured_mm)))
    ax = axes[1, 1]
    bars = ax.bar([f"第 {group} 组"], [infrared.ir_normal_error_deg], width=.38, color=red)
    ax.bar_label(bars, fmt="%.2f°", padding=4)
    ax.set(title="红外平面法向夹角误差", ylabel="夹角误差（°）",
           xlim=(-.65, .65), ylim=(0, 1.22 * infrared.ir_normal_error_deg))
    fig.suptitle("2026 年 11 月 11 日定位误差（机翼基准取第 7 组）", fontsize=19, weight="bold")
    fig.savefig(output / "error_dashboard.png", dpi=180)
    plt.close(fig)


def make_3d(poses, group_id, output):
    import pyvista as pv
    from visualization_3d import load_display_meshes, plane_patch

    wing, robot = load_display_meshes(PROJECT / "models")
    pose = next(item for item in poses if item["group_id"] == group_id)
    gt = np.array([item["T_W_C_gt_mm"][:3, 3] for item in poses])
    vision = np.array([item["T_W_C_vision_mm"][:3, 3] for item in poses])
    blue, orange, red = "#14748e", "#d67d2e", "#bd4057"
    plotter = pv.Plotter(shape=(1, 2), off_screen=True, window_size=(1920, 900))
    try:
        plotter.set_background("#f5f7fa", all_renderers=True)
        plotter.subplot(0, 0)
        plotter.add_mesh(wing, color="#8ea3b7", opacity=.25, smooth_shading=True)
        plotter.add_points(gt, color=blue, point_size=10, render_points_as_spheres=True, label="Mocap camera")
        plotter.add_points(vision, color=orange, point_size=10, render_points_as_spheres=True, label="Vision camera")
        plotter.add_lines(np.stack([gt, vision], axis=1).reshape(-1, 3), color="#536678", width=2)
        plotter.add_points(WING_MARKERS_MM, color="#253a4b", point_size=12, render_points_as_spheres=True,
                           label="Wing markers")
        plotter.add_text(f"Wing and camera positions | n = {len(poses)}", position="upper_left", font_size=15)
        plotter.camera_position = [(1420, -1400, 950), (725, 100, -10), (0, 0, 1)]
        plotter.enable_parallel_projection()
        plotter.camera.parallel_scale = 660
        legend = plotter.add_legend(bcolor="#f5f7fa", face="rectangle", size=(.36, .14))
        legend.SetPosition(.59, .045)
        legend.SetPosition2(.36, .14)
        plotter.add_axes()

        plotter.subplot(0, 1)
        center = pose["T_W_R_gt_mm"][:3, 3]
        x, y, _ = center
        local_wing = wing.clip_box((x - 120, x + 120, y - 110, y + 110, -140, 150), invert=False)
        plotter.add_mesh(local_wing, color="#8ea3b7", opacity=.10)
        reference = pv.Plane(center=(x, y, PLANE_Z_MM), direction=(0, 0, 1), i_size=200, j_size=170)
        plotter.add_mesh(reference, color="#73a9c7", opacity=.26, label="Reference plane")
        plotter.add_mesh(robot.transform(pose["T_W_R_gt_mm"], inplace=False), color=blue, opacity=.65,
                         label="Mocap robot")
        plotter.add_mesh(robot.transform(pose["T_W_R_vision_mm"], inplace=False), color=orange, opacity=.35,
                         label="Vision robot")
        if "ir_normal_W" in pose:
            projection = pose["ir_projection_W_mm"]
            patch = plane_patch(projection, pose["ir_normal_W"])
            plotter.add_mesh(pv.PolyData(patch, np.array([4, 0, 1, 2, 3])), color=red, opacity=.56,
                             label="IR plane")
            plotter.add_points(pose["ir_points_W_mm"], color=red, point_size=11, render_points_as_spheres=True)
            for point, normal, color in ((projection, pose["ir_normal_W"], red),
                                         (pose["ir_true_projection_W_mm"], np.array([0, 0, 1]), blue)):
                plotter.add_lines(np.array([center, point]), color=color, width=2)
                plotter.add_arrows(np.array([point]), np.array([normal]), mag=34, color=color)
            title = f"Group {group_id} | Reference z = -75 mm and IR plane"
        else:
            points = np.array([pose["T_W_C_gt_mm"][:3, 3], pose["T_W_C_vision_mm"][:3, 3]])
            plotter.add_lines(points, color="#253a4b", width=3)
            for point, color in zip(points, (blue, orange)):
                plotter.add_points(np.array([point]), color=color, point_size=13, render_points_as_spheres=True)
            title = f"Group {group_id} | Mocap and vision positions"
        plotter.add_text(title, position="upper_left", font_size=15)
        plotter.camera_position = [(x + 280, y - 380, 240), (x, y, -22), (0, 0, 1)]
        plotter.enable_parallel_projection()
        plotter.camera.parallel_scale = 180
        legend = plotter.add_legend(bcolor="#f5f7fa", face="rectangle", size=(.36, .17))
        legend.SetPosition(.59, .045)
        legend.SetPosition2(.36, .17)
        plotter.add_axes()
        plotter.show(screenshot=str(output / "scene_comparison.png"), auto_close=False)
    finally:
        plotter.close()


def write_report(frame, summary, edges, output, scene_generated):
    def value(number):
        return "—" if number is None else f"{number:.3f}"

    lines = [
        "# 2026 年 11 月 11 日相机与红外定位精度分析",
        "",
        "CSV 与动捕 MAT 各含 25 组，按 CSV 组号和 MAT 行号对应。实验期间机翼标记位置固定，采用第 7 组的三个机翼标记建立统一动捕基准。"
        "第 1–25 组均可评估视觉位姿，红外仅第 11 组有有效平面。",
        "",
        "| 指标 | 有效组数 | 均值 | RMSE | 中位数 | P95 | 最大值 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, (title, unit) in METRICS.items():
        metrics = summary[key]
        values = " | ".join(value(metrics[name]) for name in ("mean", "rmse", "median", "p95", "maximum"))
        lines.append(f"| {title}（{unit}） | {metrics['n']} | {values} |")
    infrared = frame.loc[frame.ir_evaluated].iloc[0]
    position_peak = frame.loc[frame.camera_position_error_mm.idxmax()]
    attitude_peak = frame.loc[frame.camera_rotation_error_deg.idxmax()]
    bias = frame[[f"camera_error_{axis}_mm" for axis in "xyz"]].mean()
    max_edge = edges.loc[edges.used_for_reference, "difference_mm"].abs().max()
    lines += [
        "",
        f"第 {int(infrared.group_id)} 组的红外平面距离为 {infrared.ir_distance_measured_mm:.3f} mm，"
        f"动捕参考距离为 {infrared.ir_distance_gt_mm:.3f} mm，距离偏差为 {infrared.ir_distance_error_signed_mm:+.3f} mm。"
        f"法向夹角为 {infrared.ir_normal_error_deg:.3f}°。红外统计只反映这一组。",
        "",
        f"视觉位置误差最大的是第 {int(position_peak.group_id)} 组（{position_peak.camera_position_error_mm:.3f} mm），"
        f"姿态误差最大的是第 {int(attitude_peak.group_id)} 组（{attitude_peak.camera_rotation_error_deg:.3f}°）。"
        f"视觉相对动捕参考的 XYZ 平均偏差为 ({bias.iloc[0]:+.3f}, {bias.iloc[1]:+.3f}, {bias.iloc[2]:+.3f}) mm。",
        "",
        "![逐组视觉误差及第 11 组红外误差](error_dashboard.png)",
        "",
        "## 配准与误差口径",
        "",
        "对第 7 组的 3 个机翼标记点遍历 6 种对应，按 Kabsch 刚体拟合残差选择对应并计算动捕系到机翼系的变换。该变换应用于每组各自的机器人标记。"
        "机器人点先去重，再遍历所有对应。只有 3 个唯一点时遍历 4 选 3 的 4 个子集及各自的 6 种排列。"
        "对机器人候选位姿，要求前向射线与机翼系 z = −75 mm 平面的交点位于前方，再选择残差最小的候选。对应关系仅由标记点及该几何约束确定。"
        "本批用于评价的各组去重后均有 4 个机器人标记点，第 23 组删除了一个重复点。",
        "",
        "动捕机器人位姿经相机安装外参换算为相机参考位姿，光心偏移为 (0, −29.5, 4) mm。位置误差取机翼系内两相机光心间的欧氏距离，姿态误差取相对旋转角。"
        "红外距离比较机器人前端盖原点到两个平面的垂直距离，法向夹角按无向平面法向计算。",
        "",
        f"第 7 组机翼标记的配准点残差 RMS 为 {summary['wing_reference_rms_mm']:.3f} mm，"
        f"名义与实测边长的最大绝对差为 {max_edge:.3f} mm。各组原始机翼点的边长另存用于诊断。",
        "MAT 没有逐组时间戳，按组序匹配依赖采集时样机姿态保持一致。"
        "视觉位姿直接取原始 CSV，其中 25 组均使用同一组固定内参，逐帧对焦读数为空。",
        "",
        "## 输出文件",
        "",
        "- [逐组误差与状态](per_group_errors.csv)保留全部组号，未评价项留空。",
        "- [汇总指标](summary_metrics.csv)、[动捕位姿与红外几何](poses.json)、[机器人标记候选](robot_correspondence_candidates.csv)和[机翼标记边长诊断](wing_marker_edge_diagnostics.csv)用于复核。",
        "- 在实验目录运行 `python analyze_20261111.py` 可重算。",
    ]
    if scene_generated:
        lines += ["", "![机翼模型与第 11 组红外平面对照](scene_comparison.png)"]
    (output / "分析报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-3d", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    self_check_registration()
    frame, summary, poses, edges = analyze(output)
    make_error_plot(frame, summary, output)
    if not args.skip_3d:
        make_3d(poses, summary["ir_groups"][0], output)
    write_report(frame, summary, edges, output, not args.skip_3d)
    print(frame.loc[frame.camera_evaluated, list(METRICS)].mean().to_string())


if __name__ == "__main__":
    main()
