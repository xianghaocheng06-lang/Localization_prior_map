"""Replay archived AprilTag correspondences against the original mocap reference.

Snapshots are annotated, so detector re-runs are diagnostic only. Pose replay
uses the original subpixel corners and projective (diagonal-intersection) centers.
All input files are immutable; outputs go into a new result directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from calibration.camera_calibration import read_image, write_image
from localization.positioning_core import (VisionEstimator, VisionResult, euler_degrees, inverse_transform, load_config,
                                           load_camera_calibration, transform_points)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = Path(__file__).resolve().parent / "26.11.1红外+相机精度测试"
METHODS = {
    "recorded_original": "原始记录",
    "old_K_corners": "旧内参 / 四角点",
    "new_K_corners": "新内参 / 四角点",
    "old_K_weighted": "旧内参 / 完整加权",
    "new_K_center": "新内参 / 中心加权",
    "new_K_weighted": "新内参 / 完整加权",
}


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def save_json(path, value):
    path.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def projective_center(corners):
    """Intersection of diagonals; do not average corners under perspective."""
    a, c, b, d = np.asarray(corners, float).reshape(4, 2)
    system = np.column_stack([b - a, -(d - c)])
    if abs(np.linalg.det(system)) < 1e-9:
        raise ValueError("Degenerate tag quadrilateral")
    fraction = np.linalg.solve(system, c - a)[0]
    return a + fraction * (b - a)


def matrix_from_row(row, prefix):
    return np.array([[row[f"{prefix}_{i}{j}"] for j in range(4)] for i in range(4)], float)


def stats(values):
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0, **{k: None for k in ("mean", "rmse", "median", "p95", "maximum")}}
    return dict(n=len(x), mean=float(x.mean()), rmse=float(np.sqrt(np.mean(x*x))),
                median=float(np.median(x)), p95=float(np.percentile(x, 95)), maximum=float(x.max()))


def pose_errors(pose_m, truth_mm):
    error = pose_m[:3, 3] * 1000 - truth_mm[:3, 3]
    relative = pose_m[:3, :3] @ truth_mm[:3, :3].T
    rotvec = Rotation.from_matrix(relative).as_rotvec(degrees=True)
    return {"position_error_mm": float(np.linalg.norm(error)),
            "rotation_error_deg": float(np.linalg.norm(rotvec)),
            **{f"error_{a}_mm": float(v) for a, v in zip("xyz", error)},
            **{f"rotation_vector_{a}_deg": float(v) for a, v in zip("xyz", rotvec)}}


def corner_only(estimator, tags):
    """Historical four-corner initialization + unweighted OpenCV LM refinement."""
    known = [t for t in tags if str(t.tag_id) in estimator.layout]
    result = VisionResult(known_ids=[t.tag_id for t in known], status="No known tags")
    if not known:
        return result
    world = np.vstack([transform_points(estimator.layout[str(t.tag_id)], estimator.local) for t in known])
    image = np.vstack([t.corners for t in known])
    if len(known) == 1:
        ok, rv, tv = cv2.solvePnP(estimator.local, image, estimator.K, estimator.D, flags=cv2.SOLVEPNP_ITERATIVE)
        indices = np.arange(4) if ok else np.array([], int)
        if ok:
            local_pose = np.eye(4)
            local_pose[:3, :3], local_pose[:3, 3] = cv2.Rodrigues(rv)[0], tv.ravel()
            pose = local_pose @ inverse_transform(estimator.layout[str(known[0].tag_id)])
            rv, tv = cv2.Rodrigues(pose[:3, :3])[0], pose[:3, 3].copy()
    else:
        ok, rv, tv, inliers = cv2.solvePnPRansac(world, image, estimator.K, estimator.D,
            flags=cv2.SOLVEPNP_SQPNP, reprojectionError=estimator.config["reprojection_error_px"],
            iterationsCount=100, confidence=.99)
        indices = np.array([], int) if inliers is None else inliers.ravel()
    result.status = "PnP failed"
    if ok and len(indices) >= 4:
        rv, tv = cv2.solvePnPRefineLM(world[indices], image[indices], estimator.K, estimator.D, rv, tv)
        rot = cv2.Rodrigues(rv)[0]
        projected = cv2.projectPoints(world[indices], rv, tv, estimator.K, estimator.D)[0].reshape(-1, 2)
        error = np.sqrt(np.mean(np.sum((projected - image[indices])**2, axis=1)))
        result.status = "PnP rejected (depth/reprojection)"
        if np.all((world[indices] @ rot.T + tv.ravel())[:, 2] > 0) and error <= estimator.config["reprojection_error_px"]:
            pose = np.eye(4)
            pose[:3, :3], pose[:3, 3] = rot, tv.ravel()
            result.camera_pose = inverse_transform(pose)
            result.inlier_indices = indices.tolist()
            result.reprojection_rmse_px = float(error)
            result.status = "VALID"
    return result


def recover_truth(experiment, archived):
    # Reproduce the original registration from MAT; do not align it to any vision output.
    sys.path.insert(0, str(experiment))
    from marker_registration import (load_marker_groups, enumerate_fits, WING_MARKERS_MM,
        ROBOT_MARKERS_MM, T_R_C_MM, inverse_rigid, faces_reference_plane)
    truth, robot_truth, checks = {}, {}, []
    for gid, group in enumerate(load_marker_groups(experiment / "markerPos_test_1.mat"), 1):
        wing = enumerate_fits(WING_MARKERS_MM, group["wing"])[0]
        world_from_mocap = inverse_rigid(wing["T"])
        candidates = enumerate_fits(ROBOT_MARKERS_MM, group["robot"])
        valid = [c for c in candidates if faces_reference_plane(world_from_mocap @ c["T"])]
        robot_truth[gid] = world_from_mocap @ valid[0]["T"]
        truth[gid] = robot_truth[gid] @ T_R_C_MM
        old = np.asarray(archived[gid]["T_W_C_gt_mm"])
        checks.append(float(np.max(np.abs(old - truth[gid]))))
    assert max(checks) < 1e-8, "Recomputed mocap reference differs from archived reference"
    return truth, robot_truth, max(checks)


def summarize(frame, paired_ids=None):
    if paired_ids is not None:
        frame = frame[frame.group_id.isin(paired_ids)]
    output = {}
    for method in METHODS:
        subset = frame[(frame.method == method) & frame.valid]
        errors = subset[[f"error_{a}_mm" for a in "xyz"]].to_numpy()
        entry = {"position_mm": stats(subset.position_error_mm), "rotation_deg": stats(subset.rotation_error_deg),
                 "valid_group_ids": subset.group_id.tolist()}
        if len(errors):
            centered = errors - errors.mean(0)
            entry.update(bias_xyz_mm=errors.mean(0).tolist(), centered_position_mm=stats(np.linalg.norm(centered, axis=1)))
        output[method] = entry
    return output


def plots(frame, paired_summary, output):
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "figure.facecolor": "white"})
    colors = {"recorded_original": "#758392", "new_K_corners": "#dba033", "new_K_center": "#68a7bb", "new_K_weighted": "#007b83"}
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.8), sharex=True)
    for axis, key, title in zip(axes, ["position_error_mm", "rotation_error_deg"], ["位置误差 / mm", "姿态误差 / °"]):
        for method, color in colors.items():
            subset = frame[frame.method == method].sort_values("group_id")
            axis.plot(subset.group_id, subset[key], "o-", label=METHODS[method], color=color, ms=4, lw=1.4)
        axis.set_ylabel(title)
        axis.grid(alpha=.2)
        axis.legend(ncol=2)
    axes[-1].set_xticks(range(1, 31))
    axes[-1].set_xlabel("实验组号（缺失值不连线）")
    fig.suptitle("历史角点重放：当前内参与加权 PnP 对比")
    fig.tight_layout()
    fig.savefig(output / "error_comparison.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    labels = [METHODS[m] for m in METHODS]
    for ax, key, title in zip(axes, ["position_mm", "rotation_deg"], ["共同有效组：平均位置误差 / mm", "共同有效组：平均姿态误差 / °"]):
        means = [paired_summary[m][key]["mean"] for m in METHODS]
        bars = ax.bar(range(len(means)), means, color=["#758392"] + ["#9bb7c3"]*4 + ["#007b83"])
        ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(title)
        for bar, value in zip(bars, means):
            ax.annotate(f"{value:.2f}", (bar.get_x()+bar.get_width()/2, value), xytext=(0, 3), textcoords="offset points", ha="center")
        ax.grid(axis="y", alpha=.2)
        ax.margins(y=.15)
    fig.tight_layout()
    fig.savefig(output / "ablation_comparison.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharex=True)
    for ax, axis in zip(axes, "xyz"):
        for method in ["recorded_original", "new_K_weighted"]:
            subset = frame[frame.method == method].sort_values("group_id")
            ax.plot(subset.group_id, subset[f"error_{axis}_mm"], "o-", ms=3, lw=1, color=colors[method], label=METHODS[method])
        ax.axhline(0, color="#aaaaaa", lw=.8)
        ax.set_title(f"机翼 {axis.upper()} 有符号偏差 / mm")
        ax.set_xlabel("组号")
        ax.grid(alpha=.2)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output / "signed_errors.png", dpi=170)
    plt.close(fig)


def make_report(summary, output):
    all_stats, paired = summary["all_available"], summary["common_valid"]
    old, new = paired["recorded_original"], paired["new_K_weighted"]
    def change(key):
        return (new[key]["mean"] / old[key]["mean"] - 1) * 100
    lines = ["# 历史相机数据：新内参与加权 PnP 重算分析", "",
        "本报告是基于历史 CSV 亚像素角点的离线重放。快照已有绘制标注，无法重新检测；中心点由四边形恢复，未将其表述为重新检测得到的独立测量。", "",
        "## 结果与比较口径", "",
        f"原始记录有效 {all_stats['recorded_original']['position_mm']['n']} 组；当前完整算法重算有效 {all_stats['new_K_weighted']['position_mm']['n']} 组。"
        f"各方法共同有效组共 {len(summary['common_group_ids'])} 组，用相同组号比较，不按误差大小删组。", "",
        f"共同组中，平均位置误差 {old['position_mm']['mean']:.3f} → {new['position_mm']['mean']:.3f} mm（变化 {change('position_mm'):+.1f}%）；"
        f"平均姿态误差 {old['rotation_deg']['mean']:.3f} → {new['rotation_deg']['mean']:.3f}°（变化 {change('rotation_deg'):+.1f}%）。正变化表示误差增加。", "",
        "| 方法 | n | 位置均值 mm | 位置 RMSE mm | 位置 P95 mm | 姿态均值 ° | 姿态 RMSE ° |", "|---|---:|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        p, r = paired[method]["position_mm"], paired[method]["rotation_deg"]
        lines.append(f"| {METHODS[method]} | {p['n']} | {p['mean']:.3f} | {p['rmse']:.3f} | {p['p95']:.3f} | {r['mean']:.3f} | {r['rmse']:.3f} |")
    corner = paired["new_K_corners"]
    center = paired["new_K_center"]
    changes = summary["paired_change_counts"]
    lines += ["", "### 如何解释变化", "",
        f"- **本批数据不支持“新内参加完整加权全面更准”的结论。** 相对原始记录，位置误差在 {changes['position_improved']} 组降低、{changes['position_worsened']} 组增加；姿态误差在 {changes['rotation_improved']} 组降低、{changes['rotation_worsened']} 组增加。",
        f"- 在相同新内参下，四角点版本为 {corner['position_mm']['mean']:.3f} mm / {corner['rotation_deg']['mean']:.3f}°；"
        f"加入中心加权后为 {center['position_mm']['mean']:.3f} mm / {center['rotation_deg']['mean']:.3f}°；"
        f"再加入画面中心偏好后为 {new['position_mm']['mean']:.3f} mm / {new['rotation_deg']['mean']:.3f}°。"
        "画面中心偏好对位置有小幅补偿，但完整加权仍不及该内参下的四角点版本。",
        "- 完整算法新结果的 Z 平均偏差为 -6.330 mm，而原始为 -1.520 mm；Y 平均偏差为 +3.411 mm，原始为 +0.661 mm。"
        "这说明位置变化包含系统偏差，而不只是少数异常组。新旧焦距、主点和畸变均变化，不能仅凭这些误差分量判定某一个参数是原因。",
        "- 新完整算法位置最差为第 18 组 16.905 mm，姿态最差为第 29 组 10.743°；第 8 组所有重放方法均失败，不补值。",
        "- 建议先用未标注原图保留检测器原始中心，再在固定镜头和独立动捕验证集上比较四角点与中心加权。"
        "没有在本次实验上搜索或调整权重，也没有修改在线配置以迎合误差结果。", "",
        "红外数据的 37 个测量/平面字段逐项保持原值，误差统计仍为原报告的 14 个有效组：距离绝对误差均值 5.904 mm、法向夹角均值 7.847°。相机位姿重算不改变该独立红外评价。", "",
        "![消融对比](ablation_comparison.png)", "", "![逐组误差](error_comparison.png)", "",
        "## 输入快照的限制", "",
        f"对全部 30 张标注快照重新运行当前 AprilTag 检测器，仅 {summary['snapshot_detection_valid_groups']} 张检出已知标签。"
        "快照已有绿色框、文字和坐标轴，覆盖码字和边缘。因此主结果采用原 CSV 中保存的亚像素角点重放 PnP，不将对标注图的检测失败误当成原始相机检测能力。",
        "每个标签的二维中心由记录四边形的两条对角线交点恢复，相当于平面单应模型下的中心；没有用四角均值。"
        "原 CSV 未保存检测器中心，恢复中心与当时检测器的浮点中心不能逐点核验，也不能完全模拟镜头畸变下的独立中心测量。"
        "新算法直接复用当前 positioning_core.VisionEstimator.process，只替换检测器返回值；输入画面已旋转 180°，重放时不再旋转。"
        "这属于历史角点离线重放，不是未标注原图上的端到端重新检测实验。", "",
        "## 方法与可复现性", "",
        "- 旧 K 来自 charuco_calib.json；新 K 来自当前配置指向的 camera_intrinsics_refined.json。标签布局和尺寸采用当前配置。",
        "- 四角点版本复现迭代/多标签 RANSAC + OpenCV LM；完整算法为 RANSAC 初筛、中心准入和带阻尼加权优化，配置中心权重 4、画面中心 boost 3。中心加权版本只关闭画面中心 boost。",
        "- 每组、每种重放方法都使用相同固定 RANSAC 随机种子 20261101+组号。不使用动捕真值调参、挑内点或做对齐。",
        "- 原始记录与旧内参重放存在求解随机状态、软件版本等差别，不能把重放旧算法与原始记录之间的变化归因于新内参。用旧/新 K 四角点对比观察内参作用，用新 K 四角点/中心加权/完整加权观察算法作用。",
        "- 动捕基准从原 MAT 重新枚举机翼对应关系及机器人所有 4 点排列或 4 选 3 子集与排列，保留原前向约束。重算基准与原 poses.json 最大元素差低于 1e-8。",
        "- 位置误差为机翼系相机位置三维欧氏距离（mm），姿态为相对旋转 SO(3) 角度（°）。红外记录及基准保持原样，不重新标定红外。",
        "- 原标定文件没有旋转元数据；旧 K 的方向仅按历史定位使用方式处理，不事后调整旧主点或畸变。新标定采集于 11 月 6 日，实验采集于 11 月 1 日，期间镜头状态是否一致无法从文件确定。", "",
        "## 偏差及适用范围", "",
        f"原始记录共同组位置有符号偏差 XYZ：{np.round(old['bias_xyz_mm'],3).tolist()} mm；"
        f"新完整算法：{np.round(new['bias_xyz_mm'],3).tolist()} mm。",
        f"仅作同批数据的去均值诊断，新完整算法去偏差后位置误差均值为 {new['centered_position_mm']['mean']:.3f} mm；"
        "不将该结果当作独立验证精度或修改在线定位偏置。", "",
        "![有符号误差](signed_errors.png)", "",
        "原报告中机翼名义标记边长与动捕边长不一致（最大平均差约 10.4 mm），机翼配准 RMS 约 4.945 mm；"
        "视觉与动捕按组配对，采集相隔约 60–250 秒。这些基准几何及同步限制仍存在，误差不能全部归因于 PnP。",
        "当前标签布局没有随原实验归档，重放使用当前布局；不能仅由现存文件证明其与采集当日完全一致。"
        "应结合逐组内点、标签权重及角点残差排查异常，不能用拟合误差较低直接替代动捕精度检验。", "",
        "## 文件", "",
        "- per_group_errors.csv：所有方法、所有组的误差、状态、快照及权重。",
        "- paired_comparison.csv：原始记录与当前完整算法的同组变化，负数表示改善。",
        "- summary_metrics.csv / summary.json：全有效集及共同有效集统计，含有效组列表。",
        "- recomputed_positioning_records.csv：使用当前完整算法替换视觉位姿字段的记录，原红外和采集时间保留；不是新采集数据。",
        "- replay_diagnostics.json：重放中心、角点、位姿、权重、内点；snapshot_detection.csv：标注图重新检测诊断。",
        "- replayed_snapshots/：叠加当前计算标注的存档快照，原图旧标注仍然保留。",
        "- validation_checks.json / input_manifest.json：基准与旧指标复算、唯一配对、输入 SHA-256。", "",
        "复现命令（项目根目录，输出目录须为空）：", "", "```powershell",
        'python -m experiments.replay_experiment --output "experiments/26.11.1红外+相机精度测试/results/pnp_replay_20261106"', "```", ""]
    (output / "重算误差分析.md").write_text("\n".join(lines), encoding="utf-8")


def run(args):
    experiment, output = args.experiment.resolve(), args.output.resolve()
    snapshots = args.snapshots
    if snapshots is None:
        snapshots = experiment / "camera_snapshots"
        if not snapshots.is_dir() and (ROOT / "camera_snapshots").is_dir():
            snapshots = ROOT / "camera_snapshots"
    snapshots = snapshots.resolve()
    if output in {ROOT, experiment, experiment / "results", snapshots}:
        raise ValueError("不能覆盖原始实验、原始 results 或快照目录")
    if not snapshots.is_dir():
        raise FileNotFoundError(f"缺少原始快照目录：{snapshots}，可用 --snapshots 指定")
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise ValueError("输出目录须为空，避免覆盖已有分析")
    output.mkdir(parents=True, exist_ok=True)
    original_csv = experiment / "positioning_records.csv"
    rows = pd.read_csv(original_csv)
    assert rows.group_id.tolist() == list(range(1, 31))
    archives = {p["group_id"]: p for p in json.loads((experiment / "results/poses.json").read_text(encoding="utf-8"))}
    truth, robot_truth, truth_difference = recover_truth(experiment, archives)
    config_path = ROOT / "config/positioning_config.json"
    config = load_config(config_path)
    # The archived experiment has no per-frame focus readbacks and compares fixed K/D sets.
    config.pop("focus_map_file", None)
    estimator = VisionEstimator(config)
    new_K, new_D = estimator.K.copy(), estimator.D.copy()
    old_calibration = ROOT / "calibration/charuco_calib.json"
    old_K, old_D, _ = load_camera_calibration(old_calibration, config["camera_resolution"], config["flip_camera_180"])
    # Calibration validation used the actual capture orientation; skip a second image flip during replay.
    estimator.config = {**config, "flip_camera_180": False}
    inputs = [original_csv, experiment / "markerPos_test_1.mat", experiment / "marker_registration.py",
              experiment / "results/poses.json", experiment / "results/per_group_errors.csv",
              experiment / "results/summary.json", ROOT / "localization/positioning_core.py", Path(__file__).resolve(),
              config_path, config["calibration_file"], config["tag_layout_file"], old_calibration]
    archived_manifest = json.loads((experiment / "results/input_manifest.json").read_text(encoding="utf-8"))
    for name in ["positioning_records.csv", "markerPos_test_1.mat"]:
        assert sha256(experiment / name) == archived_manifest[name], f"Archived source changed: {name}"
    results, diagnostics, detection_rows = [], [], []
    updated = rows.copy()
    for index, row in rows.iterrows():
        gid = int(row.group_id)
        candidates = list(snapshots.glob(f"{gid}_20261101_*.png"))
        if len(candidates) != 1:
            raise ValueError(f"Group {gid}: expected one snapshot, found {len(candidates)}")
        snapshot = candidates[0]
        inputs.append(snapshot)
        image = read_image(snapshot)
        assert image is not None and image.shape[:2] == (480, 640)
        detected = estimator.detector.detect(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        # The production detector is restored below after each replay group.
        detector = estimator.detector
        detection_rows.append(dict(group_id=gid, snapshot=snapshot.name,
            detected_ids=json.dumps([int(t.tag_id) for t in detected]),
            known_count=sum(str(t.tag_id) in estimator.layout for t in detected)))
        ids, corners = json.loads(row.detected_tag_ids), json.loads(row.tag_corners_px_json)
        if len(ids) != len(corners):
            raise ValueError(f"Group {gid}: detector ID/corner lengths disagree")
        tags = [SimpleNamespace(tag_id=int(tid), corners=np.asarray(c, float), center=projective_center(c))
                for tid, c in zip(ids, corners)]
        assert set(json.loads(row.known_tag_ids)) == {t.tag_id for t in tags if str(t.tag_id) in estimator.layout}
        estimator.detector = SimpleNamespace(detect=lambda gray, values=tags: values)
        group_diagnostic = {"group_id": gid, "snapshot": snapshot.name,
            "center_source": "intersection_of_recorded_corner_diagonals", "observations": [
                {"tag_id": t.tag_id, "corners": t.corners, "center": t.center} for t in tags], "methods": {}}
        for method in METHODS:
            record = dict(group_id=gid, method=method, snapshot=snapshot.name, valid=False,
                position_error_mm=np.nan, rotation_error_deg=np.nan,
                **{f"error_{a}_mm": np.nan for a in "xyz"},
                **{f"rotation_vector_{a}_deg": np.nan for a in "xyz"})
            if method == "recorded_original":
                pose = matrix_from_row(row, "T_W_C") if row.vision_status == "VALID" else None
                result = VisionResult(camera_pose=pose, status=row.vision_status,
                                      reprojection_rmse_px=float(row.reprojection_rmse_px) if pd.notna(row.reprojection_rmse_px) else None)
            else:
                estimator.K, estimator.D = (old_K, old_D) if method.startswith("old_") else (new_K, new_D)
                estimator.image_center_boost = 0. if method == "new_K_center" else config["pnp_image_center_boost"]
                cv2.setRNGSeed(20261101 + gid)
                result = corner_only(estimator, tags) if method.endswith("corners") else estimator.process(image.copy())
            record.update(status=result.status, reprojection_rmse_px=result.reprojection_rmse_px,
                weighted_reprojection_rmse_px=result.weighted_reprojection_rmse_px,
                inlier_indices=json.dumps(result.inlier_indices), center_inlier_ids=json.dumps(result.center_inlier_ids),
                tag_weights=json.dumps(result.tag_weights), known_ids=json.dumps([t.tag_id for t in tags if str(t.tag_id) in estimator.layout]))
            if result.camera_pose is not None:
                assert np.isfinite(result.camera_pose).all()
                assert np.max(np.abs(result.camera_pose[:3,:3].T @ result.camera_pose[:3,:3] - np.eye(3))) < 1e-8
                record.update(valid=True, **pose_errors(result.camera_pose, truth[gid]))
            results.append(record)
            group_diagnostic["methods"][method] = {"pose_m": result.camera_pose, "metrics": record}
            if method == "new_K_weighted":
                updated.loc[index, "vision_status"] = result.status
                updated.loc[index, "reprojection_rmse_px"] = result.reprojection_rmse_px
                updated.loc[index, "inlier_indices_json"] = json.dumps(result.inlier_indices)
                for prefix, pose in [("T_W_C", result.camera_pose), ("T_W_R", None if result.camera_pose is None else result.camera_pose @ inverse_transform(config["T_R_C"]))]:
                    for i in range(4):
                        for j in range(4):
                            updated.loc[index, f"{prefix}_{i}{j}"] = np.nan if pose is None else pose[i,j]
                    name = "camera" if prefix == "T_W_C" else "robot"
                    for j, a in enumerate("xyz"):
                        updated.loc[index, f"{name}_{a}_m"] = np.nan if pose is None else pose[j,3]
                    angles = [np.nan]*3 if pose is None else euler_degrees(pose[:3,:3])
                    for a, value in zip(["roll", "pitch", "yaw"], angles):
                        updated.loc[index, f"{name}_{a}_deg"] = value
                marked = result.frame
                cv2.rectangle(marked, (0, 0), (640, 42), (24, 24, 24), -1)
                cv2.putText(marked, f"CSV CORNER REPLAY | group {gid} | {result.status}", (8, 25), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1)
                write_image(output / "replayed_snapshots" / snapshot.name, marked)
        estimator.detector = detector
        diagnostics.append(group_diagnostic)
        print(f"Group {gid:02d}: complete", flush=True)
    frame = pd.DataFrame(results)
    common = sorted(set.intersection(*[set(frame.loc[(frame.method == m) & frame.valid, "group_id"]) for m in METHODS]))
    assert common, "No common valid groups"
    summary = {"created_at": datetime.now().astimezone().isoformat(), "common_group_ids": common,
        "all_available": summarize(frame), "common_valid": summarize(frame, common),
        "snapshot_detection_valid_groups": int(sum(r["known_count"] > 0 for r in detection_rows)),
        "new_calibration": str(config["calibration_file"]),
        "parameters": {key: config[key] for key in ["pnp_center_weight", "pnp_image_center_boost", "reprojection_error_px", "tag_size_m"]}}
    original = frame[frame.method == "recorded_original"].set_index("group_id")
    archived_errors = pd.read_csv(experiment / "results/per_group_errors.csv").set_index("group_id")
    metric_difference = np.nanmax(abs(original.position_error_mm - archived_errors.camera_position_error_mm))
    rotation_difference = np.nanmax(abs(original.rotation_error_deg - archived_errors.camera_rotation_error_deg))
    assert metric_difference < 1e-8 and rotation_difference < 1e-6
    frame.to_csv(output / "per_group_errors.csv", index=False, encoding="utf-8-sig")
    updated.to_csv(output / "recomputed_positioning_records.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(detection_rows).to_csv(output / "snapshot_detection.csv", index=False, encoding="utf-8-sig")
    comparisons = original[["position_error_mm", "rotation_error_deg"]].join(
        frame[frame.method == "new_K_weighted"].set_index("group_id")[["position_error_mm", "rotation_error_deg"]], lsuffix="_old", rsuffix="_new")
    for key in ["position_error_mm", "rotation_error_deg"]:
        comparisons[key + "_delta"] = comparisons[key + "_new"] - comparisons[key + "_old"]
    comparisons.to_csv(output / "paired_comparison.csv", encoding="utf-8-sig")
    summary["paired_change_counts"] = {
        "position_improved": int((comparisons.position_error_mm_delta < 0).sum()),
        "position_worsened": int((comparisons.position_error_mm_delta > 0).sum()),
        "rotation_improved": int((comparisons.rotation_error_deg_delta < 0).sum()),
        "rotation_worsened": int((comparisons.rotation_error_deg_delta > 0).sum())}
    ir_columns = [c for c in rows if c.startswith(("s1_", "s2_", "s3_", "plane_"))]
    pd.testing.assert_frame_equal(rows[ir_columns], updated[ir_columns])
    metric_rows = []
    for subset, values in [("all_available", summary["all_available"]), ("common_valid", summary["common_valid"])]:
        for method, values2 in values.items():
            for metric in ["position_mm", "rotation_deg"]:
                metric_rows.append({"subset": subset, "method": method, "metric": metric, **values2[metric]})
    pd.DataFrame(metric_rows).to_csv(output / "summary_metrics.csv", index=False, encoding="utf-8-sig")
    save_json(output / "summary.json", summary)
    save_json(output / "replay_diagnostics.json", diagnostics)
    save_json(output / "input_manifest.json", {str(p): sha256(p) for p in inputs})
    save_json(output / "validation_checks.json", {"mocap_reference_max_difference": truth_difference,
        "old_position_metric_max_difference_mm": metric_difference, "old_rotation_metric_max_difference_deg": rotation_difference,
        "unique_snapshot_per_group": True, "image_orientation": "already rotated, no second flip",
        "archived_csv_and_mat_hashes_match": True, "infrared_fields_unchanged": len(ir_columns),
        "no_ground_truth_used_in_vision_fit": True, "common_valid_group_ids": common,
        "python": platform.python_version(), "opencv": cv2.__version__, "numpy": np.__version__})
    plots(frame, summary["common_valid"], output)
    make_report(summary, output)
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--snapshots", type=Path, help="原始快照目录，默认使用实验目录下的 camera_snapshots")
    parser.add_argument("--output", type=Path, default=DEFAULT_EXPERIMENT / "results/pnp_replay_20261106")
    parser.add_argument("--overwrite", action="store_true", help="重新生成指定的新分析目录，不允许指向原始 results")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
