"""Quantify whether the third 2026-11-11 test has a repeatable error direction.

Input is ``results/directional_errors.csv`` written by ``analyze_error_directions.py``.
Nothing here re-estimates a pose: the only transform applied to a measured error is
the rigid change of basis between the wing frame and each capture's reference
camera frame.  Statistics use exact binomial sign tests, a 3-D / 2-D Rayleigh test
on unit error directions, a bootstrap interval for the bias vector, and a
leave-one-out robust bias correction whose bias for group *i* comes from the other
groups only, so a corrected error is never fitted to its own point.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2, linregress

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
WORLD = [f"error_w_{axis}_mm" for axis in "xyz"]
CAMERA = [f"error_c_{axis}_mm" for axis in "xyz"]
ROTATION = [f"rotation_vector_w_{axis}_deg" for axis in "xyz"]
BRANCH_FAILURE = "2"          # single-tag planar flip diagnosed in 分析报告.md
TAG5_CENTER_MM = np.array([334.5, 66.0, -75.0])
RNG_SEED = 20261111
BOOTSTRAP = 20000


def unit(vectors):
    norms = np.linalg.norm(vectors, axis=1)
    keep = norms > 1e-12
    return vectors[keep] / norms[keep, None]


def bootstrap_ci(values, statistic, rng, n=BOOTSTRAP):
    idx = rng.integers(0, len(values), size=(n, len(values)))
    draws = np.array([statistic(values[i]) for i in idx])
    return np.percentile(draws, [2.5, 97.5], axis=0)


def sign_block(values):
    n = len(values)
    positive = (values > 0).sum(0)
    negative = (values < 0).sum(0)
    return dict(positive=positive, negative=negative,
                p=[binomtest(int(k), n, 0.5).pvalue for k in positive])


def quadrant_count(values):
    """Count of groups in the -X/+Y quadrant and the exact one-sided p."""
    hits = int(((values[:, 0] < 0) & (values[:, 1] > 0)).sum())
    return hits, float(binomtest(hits, len(values), 0.25, alternative="greater").pvalue)


def rayleigh(directions):
    mean_unit = directions.mean(0)
    resultant = float(np.linalg.norm(mean_unit))
    p3 = float(chi2.sf(3 * len(directions) * resultant ** 2, 3))
    horizontal = directions[:, :2] / np.linalg.norm(directions[:, :2], axis=1, keepdims=True)
    r2 = float(np.linalg.norm(horizontal.mean(0)))
    return dict(mean_unit=mean_unit, resultant=resultant, p3=p3,
                resultant_xy=r2, p2=float(np.exp(-len(directions) * r2 ** 2)))


def angles_to_reference(vectors, reference):
    reference = reference / np.linalg.norm(reference)
    cosine = vectors @ reference / np.linalg.norm(vectors, axis=1)
    return np.degrees(np.arccos(np.clip(cosine, -1, 1)))


def corrected_metrics(values):
    """Remove a leave-one-out median bias and report before/after magnitudes."""
    corrected = np.array([row - np.median(np.delete(values, i, axis=0), axis=0)
                          for i, row in enumerate(values)])
    before, after = np.linalg.norm(values, axis=1), np.linalg.norm(corrected, axis=1)

    def block(norms):
        return dict(mean=float(norms.mean()), median=float(np.median(norms)),
                    rmse=float(np.sqrt(np.mean(norms ** 2))), p90=float(np.percentile(norms, 90)))
    return dict(before=block(before), after=block(after), corrected=corrected)


def describe(values, label, rng):
    norms = np.linalg.norm(values, axis=1)
    lo, hi = bootstrap_ci(values, lambda v: np.median(v, axis=0), rng)
    mean_lo, mean_hi = bootstrap_ci(values, lambda v: np.mean(v, axis=0), rng)
    quad_hits, quad_p = quadrant_count(values)
    return dict(
        label=label, n=len(values), mean=values.mean(0), median=np.median(values, axis=0),
        median_ci=np.stack([lo, hi]), mean_ci=np.stack([mean_lo, mean_hi]),
        std=values.std(0, ddof=1) if len(values) > 1 else np.zeros(3),
        signs=sign_block(values), quadrant_hits=quad_hits, quadrant_p=quad_p,
        mean_norm=float(norms.mean()), median_norm=float(np.median(norms)),
        rmse=float(np.sqrt(np.mean(norms ** 2))), p90=float(np.percentile(norms, 90)),
        ray=rayleigh(unit(values)),
    )


def plot(primary, stats, correction, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": .18})
    blue, orange, teal, grey = "#1b6585", "#c1663a", "#2a7f6f", "#7d8b99"
    fig = plt.figure(figsize=(13.4, 9.4), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=(1.08, 1), height_ratios=(1.05, 1))

    # (a) horizontal error vectors
    ax = fig.add_subplot(grid[0, 0])
    x, y = primary[WORLD[0]].to_numpy(), primary[WORLD[1]].to_numpy()
    ax.axhline(0, color=grey, lw=1)
    ax.axvline(0, color=grey, lw=1)
    for radius in (5, 10, 20):
        ax.add_patch(plt.Circle((0, 0), radius, fill=False, ls=":", lw=.9, color=grey))
    ax.scatter(x, y, s=44, color=blue, zorder=3)
    for gid, xi, yi in zip(primary.capture, x, y):
        if gid in {"4", "10"}:
            ax.annotate(gid, (xi, yi), xytext=(6, 4), textcoords="offset points", fontsize=9)
    bx, by = stats["median"][:2]
    mx, my = stats["mean"][:2]
    ax.annotate("", xy=(bx, by), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=orange, lw=2.6))
    ax.annotate("", xy=(mx, my), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=grey, lw=1.6, ls="--"))
    ax.set(xlabel="机翼系 X 误差（mm）", ylabel="机翼系 Y 误差（mm）",
           title="(a) 水平面误差向量（视觉 − 动捕）")
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(.14)
    ax.annotate("组 2：单标签平面解翻转\n（+44.1, −15.7）", xy=(x[1], y[1]), xytext=(.48, .08),
                textcoords="axes fraction", fontsize=9, color=orange,
                arrowprops=dict(arrowstyle="->", color=orange, lw=1))
    ax.text(.02, .97, "实线箭头 = 中位偏差\n虚线箭头 = 平均偏差", transform=ax.transAxes,
            fontsize=8.5, color="#40505f", va="top")

    zoom = ax.inset_axes([.60, .52, .38, .44])
    keep = (np.abs(x) < 12) & (np.abs(y) < 12)
    zoom.axhline(0, color=grey, lw=.8)
    zoom.axvline(0, color=grey, lw=.8)
    zoom.scatter(x[keep], y[keep], s=28, color=blue)
    for gid, xi, yi in zip(primary.capture[keep], x[keep], y[keep]):
        zoom.annotate(gid, (xi, yi), xytext=(4, 4), textcoords="offset points", fontsize=7.5)
    zoom.annotate("", xy=(bx, by), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=orange, lw=2))
    zoom.set(xlim=(-9, 4), ylim=(-8, 12), title="放大（不含组 2）")
    zoom.set_title("放大（不含组 2）", fontsize=9)
    zoom.tick_params(labelsize=7)

    # (b) per-axis signed values
    ax = fig.add_subplot(grid[0, 1])
    for i, (column, color) in enumerate(zip(WORLD, (blue, orange, teal))):
        values = primary[column].to_numpy()
        jitter = np.linspace(-.13, .13, len(values))
        positive = values > 0
        ax.scatter(np.full(positive.sum(), i) + jitter[positive], values[positive], s=42, color=color, zorder=3)
        ax.scatter(np.full((~positive).sum(), i) + jitter[~positive], values[~positive], s=42,
                   facecolor="white", edgecolor=color, lw=1.4, zorder=3)
        q1, q2 = np.percentile(values, [25, 75])
        ax.plot([i - .26, i + .26], [np.median(values)] * 2, color="#22313f", lw=2.4)
        ax.plot([i, i], [q1, q2], color="#22313f", lw=1, alpha=.55)
        ax.text(i, 49, f"{positive.sum()}+ / {(~positive).sum()}−", ha="center", fontsize=9, color=color)
    ax.axhline(0, color=grey, lw=1)
    ax.set_xticks(range(3), ["X", "Y", "Z"])
    ax.set_ylabel("有符号误差（mm）")
    ax.set_ylim(-22, 55)
    ax.set_title("(b) 各轴符号分布（实心 = 正，空心 = 负）")
    ax.text(.02, .04, "横线 = 中位数，竖条 = 四分位距", transform=ax.transAxes, fontsize=8.5, color="#40505f")

    # (c) horizontal direction rose
    ax = fig.add_subplot(grid[1, 0], projection="polar")
    angles = np.arctan2(y, x)
    counts, edges = np.histogram(angles, bins=np.linspace(-np.pi, np.pi, 19))
    ax.bar((edges[:-1] + edges[1:]) / 2, counts, width=np.diff(edges), color=blue,
           alpha=.72, edgecolor="white")
    mean_angle = np.arctan2(by, bx)
    ax.annotate("", xy=(mean_angle, 4.2), xytext=(mean_angle, 0),
                arrowprops=dict(arrowstyle="-|>", color=orange, lw=2.4))
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_title(f"(c) 水平误差方向（0° = +X，逆时针，每格 20°）\n"
                 f"R = {stats['ray']['resultant']:.2f}，3D Rayleigh p = {stats['ray']['p3']:.3f}", pad=16)
    ax.set_ylim(0, 4.6)
    ax.set_yticks([])
    ax.tick_params(labelsize=8)

    # (d) leave-one-out median bias correction
    ax = fig.add_subplot(grid[1, 1])
    labels = ["平均", "中位", "RMSE", "P90"]
    keys = ["mean", "median", "rmse", "p90"]
    before = [correction["before"][k] for k in keys]
    after = [correction["after"][k] for k in keys]
    position = np.arange(len(keys))
    ax.bar(position - .19, before, .36, color=grey, label="原始误差")
    ax.bar(position + .19, after, .36, color=teal, label="扣除中位偏差（留一法）")
    for i, (b, a) in enumerate(zip(before, after)):
        ax.text(i - .19, b + .2, f"{b:.1f}", ha="center", fontsize=8.5)
        ax.text(i + .19, a + .2, f"{a:.1f}", ha="center", fontsize=8.5, color=teal)
    ax.set_xticks(position, labels)
    ax.set_ylabel("误差（mm）")
    ax.set_ylim(0, max(before) * 1.36)
    ax.set_title("(d) 固定方向偏差能解释多少误差（9 组）")
    ax.legend(loc="upper right", fontsize=8.5)

    fig.suptitle("2026 年 11 月 11 日第三次测试：误差方向一致性", fontsize=18, weight="bold")
    fig.savefig(output / "direction_consistency.png", dpi=180)
    plt.close(fig)


def vector_text(values, digits=2):
    return "(" + ", ".join(f"{x:+.{digits}f}" for x in values) + ")"


def write_report(context, output):
    s, s2 = context["stats"], context["stats_no2"]
    signs, ray = s["signs"], s["ray"]
    att, corr = context["attitude"], context["correction"]
    corr2, rows = context["correction_no2"], context["rows"]
    slope, intercept = context["slope"], context["intercept"]
    quad_hits, quad_p = s["quadrant_hits"], s["quadrant_p"]
    swapped = context["stats_swapped"]
    d_median = 100 * (1 - corr2["after"]["median"] / corr2["before"]["median"])
    d_rmse = 100 * (1 - corr2["after"]["rmse"] / corr2["before"]["rmse"])

    def ci_text(ci):
        return "，".join(f"{axis} [{ci[0, i]:+.2f}, {ci[1, i]:+.2f}]" for i, axis in enumerate("XYZ"))

    lines = [
        "# 第三次测试（2026-11-11）：误差方向一致性",
        "",
        "数据为 `26.11.11红外+相机精度测试3`：动捕 MAT 第 1–10 组，视觉 CSV 共 11 行，"
        "其中第 8 组保存了 8A（对焦 368）和 8B（对焦 496）两帧，主统计用 8A。"
        "误差定义为**视觉估计 − 动捕参考**：位置误差取相机光心之差，姿态误差取相对旋转向量，"
        "全部在机翼动捕系（第 7 组三个机翼标记建立）中表达；+Z 指向机翼上方（标签平面在 z = −75 mm），"
        "+Y 指向机翼标记 y = 3 mm 到 241 mm 的一侧。",
        "",
        "## 结论",
        "",
        f"**有方向一致性，但要从“方向”这一层看。** 10 个独立组里 X 为负 {signs['negative'][0]}/10、Y 为正 {signs['positive'][1]}/10、"
        f"Z 为正 {signs['positive'][2]}/10，同时落在 −X/+Y 象限的有 {quad_hits}/10 组；"
        f"误差方向的合成向量长度 R = {ray['resultant']:.2f}（0 为各向随机，1 为完全同向），"
        f"三维 Rayleigh 检验 p = {ray['p3']:.4f}，水平面内 p = {ray['p2']:.4f}——方向分布明显不是各向同性。"
        f"分量中位偏差 {vector_text(s['median'])} mm，模长 {np.linalg.norm(s['median']):.2f} mm。",
        "",
        f"**单轴偏移量的证据弱于方向证据。** 单个轴的符号检验 8/10 只给出 p = {signs['p'][0]:.2f}；"
        f"中位偏差的 bootstrap 95% 区间 X [{s['median_ci'][0, 0]:+.2f}, {s['median_ci'][1, 0]:+.2f}]、"
        f"Z [{s['median_ci'][0, 2]:+.2f}, {s['median_ci'][1, 2]:+.2f}] 不含 0，"
        f"Y [{s['median_ci'][0, 1]:+.2f}, {s['median_ci'][1, 1]:+.2f}] 跨 0（重采样里抽到两个负样本就会把中位数拉下去）。"
        f"剔除选支失败的第 2 组后，9 组的 X [{s2['median_ci'][0, 0]:+.2f}, {s2['median_ci'][1, 0]:+.2f}]、"
        f"Y [{s2['median_ci'][0, 1]:+.2f}, {s2['median_ci'][1, 1]:+.2f}] 都不含 0，R 升到 {s2['ray']['resultant']:.2f}"
        f"（p = {s2['ray']['p3']:.4f}）。",
        "",
        f"**一致的是方向，不是幅度。** 单组合模长 2.54–11.03 mm（第 2 组的 46.8 mm 见后文），"
        f"用其余组的中位偏差做留一法补偿后，9 组的中位误差从 {corr2['before']['median']:.2f} mm 降到 "
        f"{corr2['after']['median']:.2f} mm（−{d_median:.0f}%），RMSE 从 {corr2['before']['rmse']:.2f} 降到 "
        f"{corr2['after']['rmse']:.2f} mm（−{d_rmse:.0f}%）。也就是说，典型误差里有约四成可以用一个固定方向抵消，"
        "剩下的仍是逐组散布，不能当成刚体平移一次性标定掉。",
        "",
        f"**方向一致主要来自姿态。** 姿态误差的旋转向量几乎只沿机翼系 Y 轴：Y 分量 {att['positive']}/10 为正"
        f"（二项 p = {att['p']:.4f}，中位 {att['median'][1]:+.2f}°）。"
        f"而位置 X 误差与绕 Y 的转角在 9 组（剔除第 2 组）内高度线性相关：r = {context['r']:.2f}，"
        f"p = {context['p']:.4f}，斜率 {slope:.2f} mm/°，截距 {intercept:+.2f} mm；"
        f"相机到 5 号标签中心的距离为 79–125 mm，对应力臂 1.4–2.2 mm/°，量级与实测斜率一致。"
        "这条关系与“姿态转了一点、相机光心被力臂带着走”一致；"
        f"但截距 {intercept:+.2f} mm 说明 X 上还叠着一个与姿态无关的固定偏移，转角只解释其中随组变化的部分。",
        "",
        "## 方向一致性的量化",
        "",
        "| 检验 | 结果 | 解释 |",
        "|---|---:|---|",
        f"| X 符号（负/正） | {signs['negative'][0]}/{signs['positive'][0]}，二项 p = {signs['p'][0]:.3f} | 单轴证据中等，未过 0.05 |",
        f"| Y 符号（负/正） | {signs['negative'][1]}/{signs['positive'][1]}，二项 p = {signs['p'][1]:.3f} | 同上；Y 是偏移量最大的一轴 |",
        f"| Z 符号（负/正） | {signs['negative'][2]}/{signs['positive'][2]}，二项 p = {signs['p'][2]:.3f} | 量级小但符号很齐 |",
        f"| −X 且 +Y 象限 | {quad_hits}/10，p = {quad_p:.4f} | 象限是看完数据后选的，p 值偏乐观，作参考 |",
        f"| 三维 Rayleigh | R = {ray['resultant']:.2f}，p = {ray['p3']:.4f} | 拒绝“方向各向同性” |",
        f"| 水平 Rayleigh | R_xy = {ray['resultant_xy']:.2f}，p = {ray['p2']:.4f} | 水平面内聚集更明显 |",
        f"| 中位偏差 95% CI（10 组） | {ci_text(s['median_ci'])} | X、Z 不含 0，Y 跨 0 |",
        f"| 中位偏差 95% CI（9 组） | {ci_text(s2['median_ci'])} | 剔除第 2 组后 X、Y 均不含 0 |",
        "",
        f"用平均向量判断会得到相反印象：均值 {vector_text(s['mean'])} mm，95% CI 为 {ci_text(s['mean_ci'])}，"
        "X、Y 的区间都跨 0，只有 Z 不含 0。原因是第 2 组的 +44 mm 把 X 均值拉成正的。"
        "**这批数据不能用平均误差描述偏差**，中位数才代表多数样本。",
        "",
        "## 逐组数据",
        "",
        "| 组 | X | Y | Z | 合模长 | 与中位偏差夹角 | 姿态误差 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row, angle in zip(rows.itertuples(), context["angles"]):
        lines.append(f"| {row.capture} | {row.error_w_x_mm:+.2f} | {row.error_w_y_mm:+.2f} | {row.error_w_z_mm:+.2f} "
                     f"| {row.position_error_mm:.2f} | {angle:.0f}° | {row.rotation_error_deg:.2f}° |")
    lines += [
        "",
        f"10 组误差向量与中位偏差方向的夹角中位数为 {np.median(context['angles']):.0f}°，"
        f"{int((context['angles'] < 45).sum())}/10 组在 45° 以内；例外是第 1 组 52°、第 10 组 105°、第 2 组 132°。"
        "方向确实集中，但同一个方向的模长能从 2.5 mm 变到 11 mm。",
        "",
        "## 第 2 组：方向反转是算法选支，不是反例",
        "",
        "第 2 组误差 (+44.06, −15.67, +2.15) mm、姿态 24.1°，方向与其余组正好相反，把均值和 RMSE 全部带偏。"
        "`分析报告.md` 中的平面 PnP 双候选诊断显示：同一组角点存在两支解，"
        "低四角点 RMSE 的那支（0.220 px）与动捕差 49.5 mm、姿态差 25.4°，另一支（0.367 px）只差 6.98 mm、姿态差 1.28°。"
        "在线重投影 RMSE 只有 0.223 px，残差小并不代表位姿对。这类样本必须单独判支，不能混进方向统计当作噪声。",
        "",
        "剔除第 2 组后的复核（9 组）：X 负 8/9、Y 正 8/9、−X/+Y 象限 7/9，方向合成 R 从 "
        f"{ray['resultant']:.2f} 升到 {s2['ray']['resultant']:.2f}，3D Rayleigh p = {s2['ray']['p3']:.4f}；"
        f"中位偏差 {vector_text(s2['median'])} mm，中位误差 {s2['median_norm']:.2f} mm。",
        "",
        "## 补偿效果：能减四成，不能全减",
        "",
        "| 方案 | 平均误差 | 中位误差 | RMSE | P90 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, block in (("原始（9 组）", corr2["before"]), ("扣除中位偏差（9 组）", corr2["after"]),
                        ("原始（全部 10 组）", corr["before"]), ("扣除中位偏差（10 组）", corr["after"])):
        lines.append(f"| {name} | {block['mean']:.2f} | {block['median']:.2f} | {block['rmse']:.2f} | {block['p90']:.2f} |")
    lines += [
        "",
        "9 组时补偿让中位数和 RMSE 都明显下降；把第 2 组放回去后 RMSE 反而从 "
        f"{corr['before']['rmse']:.2f} 升到 {corr['after']['rmse']:.2f} mm——因为选支失败那一组的误差被补偿推得更远。"
        "这正说明固定方向偏差只能解释“多数正常样本”，对少数结构性失败无效。",
        "",
        "## 为什么这批数据会这么一致",
        "",
        "1. 10 组全部只用机翼上的 5 号标签求解（`known_tag_ids` 恒为 `[5]`），没有多标签冗余。",
        "2. 机器人只在一个约 8 cm × 3 cm × 7 cm 的小范围内重复取位（相机光心 X 0.281–0.357 m、"
        "Y 0.056–0.081 m、Z −0.026–0.046 m），视角变化很小。",
        "3. 同一标签、近似同一视角下，内参、安装外参、标签位姿的任何系统误差都投影成几乎相同的世界系向量，"
        "于是“方向一致、幅度抖动”；单标签平面求解的倾斜/深度弱约束又让转角和位移耦合（见上文的 1.2 mm/° 关系）。",
        "",
        "**因此这批数据无法区分“固定视角下的系统偏差”和“与视角相关的模型误差”。** 要定位误差来源（内参、外参、"
        "标签位姿还是深度弱约束），必须换视角重复采集，例如改变距离、俯仰和偏航各若干组，再看误差向量是否随视角旋转。",
        "",
        "## 敏感性",
        "",
        f"若把第 8 组改用 8B 帧：X 负 {swapped['signs']['negative'][0]}/10、Y 正 {swapped['signs']['positive'][1]}/10、"
        f"−X/+Y 象限 {swapped['quadrant_hits']}/10、R = {swapped['ray']['resultant']:.2f}，"
        "方向结论不变，只是 8B 的 −35.6 mm Z 误差让 Z 的离散度变大。8B 与 8A 的角点相差 46–124 px，"
        "且动捕 MAT 没有逐组时间戳，无法确认 8B 对应同一时刻，故不计入主统计。",
        "",
        "## 红外",
        "",
        "三路红外在全部 11 条记录中都是 `DISCONNECTED`，`plane_valid` 全为 False，"
        "本次没有任何红外距离或法向数据，方向一致性只覆盖视觉。",
        "",
        "## 输出",
        "",
        "- [方向一致性图](direction_consistency.png)",
        "- [检验结果与逐组数据](direction_consistency.json)",
        "- [逐组夹角](direction_angles.csv)",
        "- 重算：先 `python analyze_error_directions.py`，再 `python analyze_direction_consistency.py`",
        "",
        "![误差方向一致性](direction_consistency.png)",
    ]
    (output / "方向一致性分析.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    rng = np.random.default_rng(RNG_SEED)
    rows = pd.read_csv(RESULTS / "directional_errors.csv", dtype={"capture": str})
    primary = rows.loc[rows.capture != "8B"].reset_index(drop=True)
    assert len(primary) == 10 and primary.group_id.is_unique

    values = primary[WORLD].to_numpy(float)
    stats = describe(values, "8A 全部 10 组", rng)
    clean = primary.loc[primary.capture != BRANCH_FAILURE].reset_index(drop=True)
    stats2 = describe(clean[WORLD].to_numpy(float), "剔除第 2 组", rng)

    angles = angles_to_reference(values, stats["median"])
    correction = corrected_metrics(values)
    correction2 = corrected_metrics(clean[WORLD].to_numpy(float))

    fit = linregress(clean.rotation_vector_w_y_deg, clean.error_w_x_mm)

    rot = primary[ROTATION].to_numpy(float)
    attitude = dict(mean=rot.mean(0), median=np.median(rot, axis=0),
                    positive=int((rot[:, 1] > 0).sum()),
                    positive_x=int((rot[:, 0] > 0).sum()), positive_z=int((rot[:, 2] > 0).sum()),
                    p=float(binomtest(int((rot[:, 1] > 0).sum()), len(rot), 0.5).pvalue),
                    y_abs_median=float(np.median(np.abs(rot[:, 1]))))

    swapped = pd.concat([primary.loc[primary.capture != "8A"], rows.loc[rows.capture == "8B"]]
                        ).reset_index(drop=True)
    stats_swapped = describe(swapped[WORLD].to_numpy(float), "8B 替换 8A", rng)

    context = dict(stats=stats, stats_no2=stats2, stats_swapped=stats_swapped, rows=primary,
                   angles=angles, correction=correction, correction_no2=correction2,
                   attitude=attitude, slope=float(fit.slope), intercept=float(fit.intercept),
                   r=float(fit.rvalue), p=float(fit.pvalue))

    payload = dict(primary=stats, without_branch_failure=stats2, swap_8b=stats_swapped,
                   correction_all=correction, correction_without_group2=correction2,
                   angles_deg=angles.tolist(),
                   attitude_rotation_vector_deg=attitude,
                   position_x_vs_rotation_y=dict(slope=float(fit.slope), intercept=float(fit.intercept),
                                                 r=float(fit.rvalue), p=float(fit.pvalue),
                                                 n=int(len(clean))),
                   group_errors=primary[["capture", *WORLD, *CAMERA, *ROTATION,
                                         "position_error_mm", "rotation_error_deg"]].to_dict("records"))

    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        raise TypeError(type(obj))

    (RESULTS / "direction_consistency.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=convert), encoding="utf-8")
    pd.DataFrame(dict(capture=primary.capture, angle_to_median_bias_deg=np.round(angles, 2))).to_csv(
        RESULTS / "direction_angles.csv", index=False, encoding="utf-8-sig")
    plot(primary, stats, correction2, RESULTS)
    write_report(context, RESULTS)

    print("all  n=%d median=%s |median|=%.2f R=%.3f p3=%.4f p2=%.4f signs+=%s p=%s quad=%d p=%.4f" % (
        stats["n"], np.round(stats["median"], 2), np.linalg.norm(stats["median"]), stats["ray"]["resultant"],
        stats["ray"]["p3"], stats["ray"]["p2"], stats["signs"]["positive"].tolist(),
        np.round(stats["signs"]["p"], 3).tolist(), stats["quadrant_hits"], stats["quadrant_p"]))
    print("no2  n=%d median=%s |median|=%.2f R=%.3f p3=%.5f quad=%d mean=%.2f median_d=%.2f rmse=%.2f" % (
        stats2["n"], np.round(stats2["median"], 2), np.linalg.norm(stats2["median"]), stats2["ray"]["resultant"],
        stats2["ray"]["p3"], stats2["quadrant_hits"], stats2["mean_norm"], stats2["median_norm"], stats2["rmse"]))
    print("corr9 before=%s" % {k: round(v, 2) for k, v in correction2["before"].items()},
          "after=%s" % {k: round(v, 2) for k, v in correction2["after"].items()})
    print("corr10 after=%s" % {k: round(v, 2) for k, v in correction["after"].items()})
    print("attitude Y+ %d/10 p=%.4f median=%s |Y|median=%.2f" % (
        attitude["positive"], attitude["p"], np.round(attitude["median"], 2), attitude["y_abs_median"]))
    print("fit slope=%.2f intercept=%.2f r=%.2f p=%.4f" % (fit.slope, fit.intercept, fit.rvalue, fit.pvalue))
    print("median angle to bias=%.0f deg; within45=%d/10" % (np.median(angles), (angles < 45).sum()))


if __name__ == "__main__":
    main()
