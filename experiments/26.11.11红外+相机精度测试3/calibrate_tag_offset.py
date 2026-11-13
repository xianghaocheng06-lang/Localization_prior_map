"""Fit and validate a fixed translation of AprilTag 5 using mocap poses."""

import hashlib
import itertools
import json
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from analyze_error_directions import PROJECT, ROOT, WORLD, save_json, vector_text
from localization.positioning_core import inverse_transform, load_config, load_tag_layout, transform_points
from replay_snapshots import REPLAY_LAYOUT

REPLAY = ROOT / "results" / "snapshot_replay"
OUTPUT = REPLAY / "tag_offset_calibration"
METHODS = {"reprojection": "固定动捕位姿拟合像点", "position": "拟合相机位置偏差"}
SCENARIOS = {"all": (), "without_2": (2,), "without_2_4_10": (2, 4, 10)}
SCENARIO_NAMES = {"all": "全部组", "without_2": "剔除 2", "without_2_4_10": "剔除 2、4、10"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class Sample:
    capture: str
    group_id: int
    error_mm: np.ndarray
    attitude_error_deg: float
    tag_points_camera_mm: np.ndarray
    rotation_camera_world: np.ndarray
    pixels: np.ndarray
    matrix: np.ndarray
    distortion: np.ndarray
    weights: np.ndarray

    def pixel_residuals(self, offset_mm):
        translation = self.rotation_camera_world @ offset_mm
        pixels, _ = cv2.projectPoints(self.tag_points_camera_mm, np.zeros(3), translation,
                                      self.matrix, self.distortion)
        return pixels.reshape(-1, 2) - self.pixels


def load_samples(config):
    layout_path = config["tag_layout_file"]
    manifest = read_json(REPLAY / "input_manifest.json")
    if hashlib.sha256(layout_path.read_bytes()).hexdigest() != manifest.get(layout_path.relative_to(PROJECT).as_posix()):
        raise ValueError("请先运行 replay_snapshots.py，生成与 replay_tag_layout.json 对应的回放基线")
    layout = load_tag_layout(layout_path)["5"].copy()
    layout[:3, 3] *= 1000
    half = config["tag_size_m"] * 500
    local = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0], [0, 0, 0]])
    points = transform_points(layout, local)
    poses = {pose["group_id"]: pose for pose in read_json(REPLAY / "poses.json")}
    recorded = pd.read_csv(ROOT / "positioning_records.csv").set_index(["group_id", "snapshot_at"])
    replay = pd.read_csv(REPLAY / "directional_errors.csv", dtype={"capture": str})
    primary = replay.loc[replay.capture != "8B"].sort_values("group_id")
    samples = []
    for row in primary.to_dict("records"):
        gid = row["group_id"]
        pose = poses[gid]
        truth = np.asarray(pose["T_W_C_gt_mm"])
        vision = np.asarray(pose["T_W_C_vision_mm"])
        error = vision[:3, 3] - truth[:3, 3]
        np.testing.assert_allclose(error, [row[key] for key in WORLD], atol=1e-9)
        metadata = read_json((ROOT / "camera_snapshots" / row["snapshot"]).with_suffix(".json"))
        original = recorded.loc[gid, row["snapshot_at"]]
        ids = json.loads(original.detected_tag_ids)
        if (ids != metadata["detected_tag_ids"] or int(metadata["group_id"]) != gid
                or metadata["snapshot_at"] != row["snapshot_at"] or json.loads(row["known_tag_ids"]) != [5]):
            raise ValueError(f"第 {gid} 组快照与基线记录不匹配，或参与求解的标签并非只有 5 号")
        index = ids.index(5)
        pixels = [*json.loads(original.tag_corners_px_json)[index], metadata["tag_centers_px"][index]]
        camera_from_world = inverse_transform(truth)
        samples.append(Sample(
            row["capture"], gid, error, row["rotation_error_deg"], transform_points(camera_from_world, points),
            camera_from_world[:3, :3], np.asarray(pixels), np.asarray(metadata["camera_matrix"]),
            np.asarray(metadata["dist_coeffs"]), np.array([1, 1, 1, 1, config["pnp_center_weight"]]),
        ))
    return samples, layout[:3, 3]


def fit_offset(samples, method):
    if method == "position":
        return -np.mean([sample.error_mm for sample in samples], axis=0)

    def residuals(offset):
        return np.concatenate([
            (sample.pixel_residuals(offset) * np.sqrt(sample.weights[:, None])).ravel()
            for sample in samples
        ])

    result = least_squares(residuals, np.zeros(3))
    if not result.success or np.linalg.matrix_rank(result.jac) != 3:
        raise ValueError(f"像点拟合未得到有效三维偏移：{result.message}")
    return result.x


def evaluate(samples, offset):
    # With a single tag, translating it in W translates the visual camera pose by the same vector.
    errors = np.asarray([sample.error_mm for sample in samples]) + offset
    distances = np.linalg.norm(errors, axis=1)
    squared_pixels = []
    for sample in samples:
        depth = sample.tag_points_camera_mm[:, 2] + (sample.rotation_camera_world @ offset)[2]
        if np.any(depth <= 0):
            raise ValueError(f"第 {sample.capture} 帧标记点位于动捕相机后方")
        residuals = sample.pixel_residuals(offset)
        squared_pixels.append(np.average(np.sum(residuals ** 2, axis=1), weights=sample.weights))
    return dict(position_mean_mm=distances.mean(), position_rmse_mm=np.sqrt(np.mean(distances ** 2)),
                mocap_projection_rmse_px=np.sqrt(np.mean(squared_pixels)), distances_mm=distances,
                error_xyz_mm=errors)


def run_split(scenario, train, validation, split):
    before = evaluate(validation, np.zeros(3))
    for method in METHODS:
        offset = fit_offset(train, method)
        after = evaluate(validation, offset)
        record = dict(scenario=scenario, split=split, method=method, calibration=[s.capture for s in train],
                      validation=[s.capture for s in validation], offset_xyz_mm=offset,
                      validation_attitude_mean_deg=np.mean([s.attitude_error_deg for s in validation]))
        for key in ("position_mean_mm", "position_rmse_mm", "mocap_projection_rmse_px"):
            record[f"before_{key}"] = before[key]
            record[f"after_{key}"] = after[key]
        yield record, before, after


def plot_validation(frame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained", sharey=True)
    for ax, scenario in zip(axes, ("without_2", "without_2_4_10")):
        panel = frame.loc[(frame.scenario == scenario) & (frame.split == "early_calibration")]
        reprojection = panel.loc[panel.method == "reprojection"]
        position = panel.loc[panel.method == "position"]
        x = np.arange(len(position))
        ax.bar(x - .24, position.before_position_mm, width=.22, label="修正前", color="#7d8995")
        ax.bar(x, reprojection.after_position_mm, width=.22, label="像点拟合", color="#195f81")
        ax.bar(x + .24, position.after_position_mm, width=.22, label="位置拟合", color="#ae561c")
        ax.set(title=SCENARIO_NAMES[scenario], xlabel="验证组", xticks=x, xticklabels=position.capture)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("相机位置误差（mm）")
    axes[0].legend()
    fig.suptitle("用前半组校准后，验证组的逐组位置误差", fontsize=16)
    fig.savefig(OUTPUT / "validation_errors.png", dpi=180)
    plt.close(fig)


def write_report(center, named, pooled, exhaustive):
    lines = [
        "# 11.11 第三次测试 · 5 号标签三维平移校准",
        "",
        f"基线为新版布局，5 号标签中心的机翼系 XYZ 为 {vector_text(center)} mm。第 8 组用 8A，"
        "动捕机翼系沿用第 7 组机翼标记建立的坐标基准，相机真值由动捕机器人位姿与相机安装外参相乘得到。",
        "",
        "像点拟合固定校准组的动捕相机位置和姿态，以快照中 5 号标签的四角、检测中心、逐帧内参和畸变系数"
        "拟合机翼系 XYZ 平移。每角权重为 1，中心沿用定位程序配置。另以相机光心三维误差平方和最小为目标做对照，"
        "其偏移等于校准组视觉减动捕的平均位置误差取反。",
        "",
        "每个方案按采集顺序拆为前、后两半，先用前半拟合并验证后半，再交换角色。每次拟合只用校准组。"
        "单标签布局的平移会给所有视觉光心加上同一个向量，视觉姿态误差和视觉 PnP 自身的重投影残差保持不变。"
        "表中像点 RMSE 以动捕相机位姿投影标签点后与观测比较，用每点二维残差计算加权 RMSE。",
        "",
        "合并两次互换的验证结果，每组只计入一次，使用由另一半数据拟合的偏移。",
        "",
        "| 样本 | 拟合目标 | 验证平均位置误差（mm） | 验证位置 RMSE（mm） |",
        "|---|---|---:|---:|",
    ]
    for row in pooled:
        lines.append(f"| {SCENARIO_NAMES[row['scenario']]} | {METHODS[row['method']]} | "
                     f"{row['before_mean_mm']:.3f} → {row['after_mean_mm']:.3f} | "
                     f"{row['before_rmse_mm']:.3f} → {row['after_rmse_mm']:.3f} |")
    lines += [
        "",
        "各次划分的拟合偏移与验证结果如下。",
        "",
        "| 样本 | 校准 → 验证 | 拟合目标 | 偏移 ΔX / ΔY / ΔZ（mm） | 验证平均位置误差（mm） | 验证位置 RMSE（mm） | 验证像点 RMSE（px） |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in named:
        lines.append(f"| {SCENARIO_NAMES[row['scenario']]} | {','.join(row['calibration'])} → {','.join(row['validation'])} | "
                     f"{METHODS[row['method']]} | {vector_text(row['offset_xyz_mm'])} | "
                     f"{row['before_position_mean_mm']:.3f} → {row['after_position_mean_mm']:.3f} | "
                     f"{row['before_position_rmse_mm']:.3f} → {row['after_position_rmse_mm']:.3f} | "
                     f"{row['before_mocap_projection_rmse_px']:.3f} → {row['after_mocap_projection_rmse_px']:.3f} |")
    lines += [
        "",
        "枚举近半划分及其交换方向，检查改善是否依赖某一次分组。下表的 RMSE 变化为修正后减修正前，负数表示下降。"
        "不同划分复用了同一批数据，结果用于检查分组敏感性。剔除名单来自此前对本批全样本误差的检查，"
        "剔除后的验证结果反映保留样本的表现。全部组结果可对照筛选影响。",
        "",
        "| 样本 | 拟合目标 | 位置 RMSE 下降的划分 | 位置 RMSE 变化中位数（mm） | 像点 RMSE 下降的划分 |",
        "|---|---|---:|---:|---:|",
    ]
    for row in exhaustive:
        lines.append(f"| {SCENARIO_NAMES[row['scenario']]} | {METHODS[row['method']]} | "
                     f"{row['position_rmse_improved']}/{row['n']} | {row['median_position_rmse_change_mm']:+.3f} | "
                     f"{row['projection_rmse_improved']}/{row['n']} |")
    lines += ["", "每种偏移都由当前批次的一部分样本拟合。结果反映固定平移在本批数据上的校正效果，"
              "动捕配准、安装外参和相机内参的偏差也可能进入拟合出的平移。",
              "", "[逐组验证误差](validation_errors.csv) · [统计与所有划分](summary.json)",
              "", "![验证组修正前后位置误差](validation_errors.png)", ""]
    (OUTPUT / "标签偏移校准验证.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    config = load_config(PROJECT / "config/positioning_config.json")
    config["tag_layout_file"] = REPLAY_LAYOUT
    samples, center = load_samples(config)
    named, details, exhaustive, all_splits = [], [], [], []
    for scenario, excluded in SCENARIOS.items():
        eligible = [s for s in samples if s.group_id not in excluded]
        cut = (len(eligible) + 1) // 2
        for split, train, validation in (
            ("early_calibration", eligible[:cut], eligible[cut:]),
            ("late_calibration", eligible[cut:], eligible[:cut]),
        ):
            for row, before, after in run_split(scenario, train, validation, split):
                named.append(row)
                for i, sample in enumerate(validation):
                    details.append(dict(scenario=scenario, split=split, method=row["method"], capture=sample.capture,
                                        before_position_mm=before["distances_mm"][i],
                                        after_position_mm=after["distances_mm"][i],
                                        attitude_error_deg=sample.attitude_error_deg,
                                        **dict(zip(WORLD, after["error_xyz_mm"][i]))))
        for count in sorted({cut, len(eligible) - cut}):
            for train_indices in itertools.combinations(range(len(eligible)), count):
                train = [eligible[i] for i in train_indices]
                validation = [s for i, s in enumerate(eligible) if i not in train_indices]
                all_splits.extend(row for row, _, _ in run_split(scenario, train, validation, "balanced"))
        for method in METHODS:
            selected = [row for row in all_splits if row["scenario"] == scenario and row["method"] == method]
            changes = np.array([row["after_position_rmse_mm"] - row["before_position_rmse_mm"] for row in selected])
            improved_pixels = sum(row["after_mocap_projection_rmse_px"] < row["before_mocap_projection_rmse_px"]
                                  for row in selected)
            exhaustive.append(dict(scenario=scenario, method=method, n=len(selected),
                                   position_rmse_improved=int((changes < 0).sum()),
                                   median_position_rmse_change_mm=np.median(changes),
                                   projection_rmse_improved=int(improved_pixels)))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(details)
    pooled = []
    for (scenario, method), held_out in frame.groupby(["scenario", "method"], sort=False):
        before, after = held_out.before_position_mm, held_out.after_position_mm
        pooled.append(dict(scenario=scenario, method=method, n=len(held_out), before_mean_mm=before.mean(),
                           after_mean_mm=after.mean(), before_rmse_mm=np.sqrt(np.mean(before ** 2)),
                           after_rmse_mm=np.sqrt(np.mean(after ** 2))))
    frame.to_csv(OUTPUT / "validation_errors.csv", encoding="utf-8-sig", index=False)
    save_json(OUTPUT / "summary.json", dict(baseline_tag_xyz_mm=center, chronological=named,
                                           pooled_validation=pooled, balanced_summary=exhaustive,
                                           balanced_splits=all_splits))
    plot_validation(frame)
    write_report(center, named, pooled, exhaustive)


if __name__ == "__main__":
    main()
