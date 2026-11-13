"""Camera error direction and centering analysis. Run from any directory.
Position errors use e_W = t_vision - t_mocap in mm.
Attitude bias is removed multiplicatively on SO(3), not by Euler subtraction.
"""
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from scipy.stats import binomtest, f
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "camera_centering"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20261101
BOOTSTRAPS = 20000
BLUE, ORANGE, RED = "#087ea4", "#e88b26", "#bf4156"
plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False, "font.size": 10.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": .17,
    "figure.facecolor": "#f6f8fb", "axes.facecolor": "white",
    "savefig.facecolor": "#f6f8fb",
})


def serial(value):
    if isinstance(value, np.ndarray):
        return serial(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    return value


def save_json(name, value):
    (OUT / name).write_text(json.dumps(serial(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def describe(x):
    x = np.asarray(x)
    n = len(x)
    bias = x.mean(0)
    centered = x - bias
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    for j in range(3):
        if eigenvectors[np.argmax(np.abs(eigenvectors[:, j])), j] < 0:
            eigenvectors[:, j] *= -1
    lengths, residual_lengths = np.linalg.norm(x, axis=1), np.linalg.norm(centered, axis=1)
    norm = np.linalg.norm(bias)
    unit = x / np.maximum(lengths[:, None], 1e-15)
    t2 = n * bias @ np.linalg.inv(covariance) @ bias
    pvalue = f.sf((n - 3) / (3 * (n - 1)) * t2, 3, n - 3)
    return dict(
        n=n, bias_xyz=bias, bias_norm=norm, bias_direction=bias / norm,
        raw_mean_length=lengths.mean(), raw_rmse=np.sqrt(np.mean(lengths**2)),
        centered_mean_length=residual_lengths.mean(),
        centered_rmse=np.sqrt(np.mean(residual_lengths**2)),
        centered_median_length=np.median(residual_lengths),
        centered_p95_length=np.percentile(residual_lengths, 95),
        mean_length_reduction_pct=100 * (1 - residual_lengths.mean() / lengths.mean()),
        rmse_reduction_pct=100 * (1 - np.sqrt(np.mean(residual_lengths**2) / np.mean(lengths**2))),
        bias_fraction_of_mse=norm**2 / np.mean(lengths**2),
        sample_std_xyz=np.std(centered, axis=0, ddof=1),
        covariance=covariance, correlation=np.corrcoef(x, rowvar=False),
        pca_eigenvalues=eigenvalues, pca_axes_columns=eigenvectors,
        pca_variance_fraction=eigenvalues / eigenvalues.sum(),
        pca_axis_std=np.sqrt(eigenvalues),
        pca_long_to_short_std_ratio=np.sqrt(eigenvalues[0] / eigenvalues[-1]),
        pca_first_to_second_eigenvalue_ratio=eigenvalues[0] / eigenvalues[1],
        positive_counts=(x > 0).sum(0), negative_counts=(x < 0).sum(0),
        unit_direction_resultant_length=np.linalg.norm(unit.mean(0)),
        unit_direction_mean=unit.mean(0),
        sign_test_two_sided_p=[binomtest(int(k), n, .5).pvalue for k in (x > 0).sum(0)],
        hotelling_t2=t2, hotelling_p=pvalue,
    )


def mean_bootstrap(x, indices):
    values = x[indices].mean(1)
    return np.percentile(values, [2.5, 97.5], axis=0).T


def scatter_projection(ax, values, axis_a, axis_b, three, centered=False):
    for flag, color, label, marker in [
        (False, BLUE, "4个动捕点", "o"),
        (True, ORANGE, "3个动捕点", "s"),
    ]:
        mask = three == flag
        ax.scatter(values[mask, axis_a], values[mask, axis_b],
                   c=color, marker=marker, s=39, alpha=.85, label=label, zorder=3)
    for point in values:
        ax.plot([0, point[axis_a]], [0, point[axis_b]], c="#94a5b5", alpha=.14, lw=.7)
    mean = values.mean(0)[[axis_a, axis_b]]
    cov = np.cov(values[:, [axis_a, axis_b]], rowvar=False)
    w, q = np.linalg.eigh(cov)
    idx = np.argsort(w)[::-1]
    w, q = w[idx], q[:, idx]
    angle = np.degrees(np.arctan2(q[1, 0], q[0, 0]))
    ellipse = Ellipse(mean, 2 * np.sqrt(w[0]), 2 * np.sqrt(w[1]), angle=angle,
                      edgecolor=BLUE, facecolor=BLUE, alpha=.10, lw=1.8)
    ax.add_patch(ellipse)
    ax.axhline(0, c="#7b8795", lw=.9)
    ax.axvline(0, c="#7b8795", lw=.9)
    ax.scatter(*mean, marker="*", s=150, c=RED, zorder=5)
    if not centered:
        ax.annotate("", xy=mean, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=RED, lw=2))
    for k in (13, 14, 15, 30):
        row = frame.index[frame.group_id == k][0]
        point = values[row]
        ax.annotate(str(k), point[[axis_a, axis_b]], xytext=(4, 4), textcoords="offset points", fontsize=8, color="#596575")
    letters = "XYZ"
    ax.set(xlabel=f"{letters[axis_a]}误差 / mm", ylabel=f"{letters[axis_b]}误差 / mm",
           xlim=(-17, 17), ylim=(-17, 17), aspect="equal",
           title=f"{'去中心化后' if centered else '原始误差'} · {letters[axis_a]}{letters[axis_b]}")
    ax.set_xticks([-15, -10, -5, 0, 5, 10, 15])
    ax.set_yticks([-15, -10, -5, 0, 5, 10, 15])


frame = pd.read_csv(ROOT / "results" / "per_group_errors.csv")
frame = frame.loc[frame.vision_valid].reset_index(drop=True)
poses = json.loads((ROOT / "results" / "poses.json").read_text(encoding="utf-8"))
n = len(frame)
assert n == 29 and 8 not in frame.group_id.values
E = frame[[f"camera_error_{a}_mm" for a in "xyz"]].to_numpy()
RV = frame[[f"camera_rotation_vector_{a}_deg" for a in "xyz"]].to_numpy()
T = np.array([poses[int(g) - 1]["T_W_C_gt_mm"] for g in frame.group_id])
EC = np.einsum("nji,nj->ni", T[:, :3, :3], E)
THREE = frame.unique_robot_marker_count.to_numpy() == 3
bias = E.mean(0)
Z = E - bias
position = describe(E)
position_camera = describe(EC)
four = describe(E[~THREE])
three = describe(E[THREE])
rotvec = describe(RV)
rng = np.random.default_rng(SEED)
resample = rng.integers(0, n, size=(BOOTSTRAPS, n))
ci_position = mean_bootstrap(E, resample)
ci_rotvec = mean_bootstrap(RV, resample)
position["bias_pointwise_bootstrap95_ci_xyz"] = ci_position
rotvec["bias_pointwise_bootstrap95_ci_xyz"] = ci_rotvec
# Leave-one-out constant translation offset: each observation is corrected
# using the other 28 observations, never its own error in the estimated bias.
loo_bias = (E.sum(0) - E) / (n - 1)
loo_residual = E - loo_bias
position["leave_one_out_mean_length"] = np.linalg.norm(loo_residual, axis=1).mean()
position["leave_one_out_rmse"] = np.sqrt(np.mean(np.sum(loo_residual**2, axis=1)))
D = Rotation.from_rotvec(RV, degrees=True)
B = D.mean()  # SciPy chordal SO(3) mean; appropriate for these small angles.
centered_D = D * B.inv()
centered_RV = centered_D.as_rotvec(degrees=True)
angles = np.degrees(D.magnitude())
centered_angles = np.degrees(centered_D.magnitude())
attitude = dict(
    method="Chordal SO(3) mean B of D_i=R_vision R_gt^T; D_centered_i=D_i B^T",
    bias_rotation_matrix=B.as_matrix(),
    bias_rotation_vector_W_deg=B.as_rotvec(degrees=True),
    bias_angle_deg=np.degrees(B.magnitude()),
    raw_mean_angle_deg=angles.mean(), raw_rmse_deg=np.sqrt(np.mean(angles**2)),
    centered_mean_angle_deg=centered_angles.mean(),
    centered_rmse_deg=np.sqrt(np.mean(centered_angles**2)),
    raw_rotation_vector_statistics=rotvec,
    centered_rotation_vector_statistics=describe(centered_RV),
)
# A coordinate correction consistent with this convention is
# R_vision_corrected = B^T R_vision. Its relative rotation is B^T D_i,
# conjugate to D_i B^T, so the residual angles are identical.
corrected_alt = B.inv() * D
assert np.allclose(corrected_alt.magnitude(), centered_D.magnitude(), atol=1e-12)
validation = dict(
    n_valid=n, excluded_vision_groups=[8],
    max_abs_centered_position_mean_mm=np.abs(Z.mean(0)).max(),
    covariance_invariance_max_abs_mm2=np.abs(np.cov(E, rowvar=False)-np.cov(Z, rowvar=False)).max(),
    mse_decomposition_abs_error_mm2=abs(
        np.mean(np.sum(E**2, axis=1)) - bias @ bias - np.mean(np.sum(Z**2, axis=1))),
    raw_norm_vs_previous_analysis_max_abs_mm=np.max(np.abs(np.linalg.norm(E, axis=1)-frame.camera_position_error_mm)),
    rotation_norm_vs_previous_analysis_max_abs_deg=np.max(np.abs(angles-frame.camera_rotation_error_deg)),
    fixed_world_offset_vs_camera_offset_raw_norm_difference_mm=np.max(np.abs(np.linalg.norm(E, axis=1)-np.linalg.norm(EC, axis=1))),
    so3_multiplicative_correction_verified=True,
)
assert validation["max_abs_centered_position_mean_mm"] < 1e-12
assert validation["covariance_invariance_max_abs_mm2"] < 1e-10
assert validation["mse_decomposition_abs_error_mm2"] < 1e-10
assert validation["raw_norm_vs_previous_analysis_max_abs_mm"] < 1e-8
validation["passed"] = True

for j, a in enumerate("xyz"):
    frame[f"position_centered_W_{a}_mm"] = Z[:, j]
    frame[f"position_bias_W_{a}_mm"] = bias[j]
    frame[f"position_error_C_{a}_mm"] = EC[:, j]
    frame[f"position_centered_C_{a}_mm"] = EC[:, j] - EC.mean(0)[j]
    frame[f"rotation_centered_so3_W_{a}_deg"] = centered_RV[:, j]
frame["position_centered_norm_mm"] = np.linalg.norm(Z, axis=1)
frame["position_leave_one_out_norm_mm"] = np.linalg.norm(loo_residual, axis=1)
frame["rotation_centered_so3_angle_deg"] = centered_angles
frame.to_csv(OUT / "centered_per_group.csv", index=False, encoding="utf-8-sig")
summary = dict(position_W=position, position_C=position_camera,
               position_W_four_markers=four, position_W_three_markers=three, attitude=attitude,
               bootstrap_seed=SEED, bootstrap_resamples=BOOTSTRAPS)
save_json("centering_summary.json", summary)
save_json("validation_checks.json", validation)
save_json("input_manifest.json", {
    name: hashlib.sha256((ROOT / "results" / name).read_bytes()).hexdigest()
    for name in ["per_group_errors.csv", "poses.json"]
})
save_json("software_versions.json", dict(python=platform.python_version(), numpy=np.__version__,
                                         pandas=pd.__version__, matplotlib=matplotlib.__version__))

fig, axes = plt.subplots(2, 3, figsize=(15, 10.5), layout="constrained")
for col, pair in enumerate([(0, 1), (0, 2), (1, 2)]):
    scatter_projection(axes[0, col], E, *pair, THREE)
    scatter_projection(axes[1, col], Z, *pair, THREE, centered=True)
axes[0, 0].legend(loc="upper left", fontsize=9)
fig.suptitle("相机位置误差：固定方向偏置与去中心化后的散布", fontsize=20, fontweight="bold")
fig.supxlabel("误差 = 视觉 − 动捕；均在机翼坐标系 W 中。红星为均值；淡色椭圆表示协方差的1σ主轴尺度，不是均值置信区间。\n"
              "去中心化只平移点云，不改变协方差、主轴形状和各向异性。数字为部分组号。", fontsize=11, color="#596575")
fig.savefig(OUT / "position_centering.png", dpi=180)
fig.savefig(OUT / "position_centering.svg")
plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(15, 9.5), layout="constrained")
ax = axes[0, 0]
means = position["bias_xyz"]
ax.bar(range(3), means, color=[BLUE, ORANGE, "#8a65b5"], width=.55, alpha=.9)
ax.errorbar(range(3), means, yerr=np.array([means-ci_position[:, 0], ci_position[:, 1]-means]),
            fmt="none", c="#263c51", capsize=6, lw=1.7)
ax.axhline(0, c="#7b8795", lw=1)
for j in range(3):
    ax.annotate(f"{means[j]:+.3f}", (j, means[j]), xytext=(0, 12 if means[j] > 0 else -17),
                textcoords="offset points", ha="center", color="#263c51")
ax.set(xticks=range(3), xticklabels=list("XYZ"), ylabel="有符号误差 / mm",
       title="平均平移偏置与逐轴 bootstrap 95% 区间")
ax = axes[0, 1]
g = frame.group_id.to_numpy()
raw_lengths = np.linalg.norm(E, axis=1)
for i in range(n):
    ax.plot([g[i], g[i]], [raw_lengths[i], frame.position_centered_norm_mm.iloc[i]], color="#a9b6c3", lw=1)
ax.scatter(g, raw_lengths, c=ORANGE, s=35, label=f"原始均值 {raw_lengths.mean():.3f} mm")
ax.scatter(g, frame.position_centered_norm_mm, c=BLUE, s=35,
           label=f"去中心后均值 {frame.position_centered_norm_mm.mean():.3f} mm")
ax.scatter(g[THREE], raw_lengths[THREE], facecolors="none", edgecolors=RED, s=100, marker="s", label="3个动捕点")
ax.set(xlabel="采集组号", ylabel="三维误差模长 / mm", title="去中心化前后逐组误差", ylim=(0, 18))
ax.legend(fontsize=9)
ax = axes[1, 0]
ratios = position["pca_variance_fraction"] * 100
bars = ax.bar(["PC1", "PC2", "PC3"], ratios, color=[BLUE, "#51a7bd", "#b1d3df"], width=.6)
ax.axhline(100/3, c="#8996a5", ls="--", lw=1, label="各向同性参照 33.3%")
ax.bar_label(bars, labels=[f"{v:.1f}%" for v in ratios], padding=5)
ax.set(ylabel="残差方差贡献 / %", title="去中心化残差 PCA：前两轴接近，无单一主导轴", ylim=(0, 60))
ax.legend(fontsize=9)
ax = axes[1, 1]
for offset, sample, name, color in [(-.18, four, "4点组 n=24", BLUE), (.18, three, "3点组 n=5", ORANGE)]:
    ax.bar(np.arange(3)+offset, sample["bias_xyz"], width=.35, label=name, color=color)
ax.axhline(0, c="#7b8795", lw=1)
ax.set(xticks=range(3), xticklabels=list("XYZ"), ylabel="平均有符号误差 / mm", title="按机器人动捕点数量分组：方向偏置不同")
ax.legend(fontsize=9)
fig.suptitle("偏置解释了多少误差？", fontsize=20, fontweight="bold")
fig.supxlabel(f"固定偏置模长 {np.linalg.norm(bias):.3f} mm，占原始均方误差的 {100*position['bias_fraction_of_mse']:.1f}%；"
              f"去中心后仍有 RMSE {position['centered_rmse']:.3f} mm。", fontsize=12, color="#596575")
fig.savefig(OUT / "position_bias_diagnostics.png", dpi=180)
plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(15, 9.5), layout="constrained")
for ax, values, title in [(axes[0, 0], RV, "原始姿态误差旋转向量 · XZ"),
                          (axes[0, 1], centered_RV, "去旋转偏置后 · XZ")]:
    for flag, color, label in [(False, BLUE, "4个动捕点"), (True, ORANGE, "3个动捕点")]:
        mask=THREE==flag
        ax.scatter(values[mask, 0], values[mask, 2], c=color, s=40, label=label)
    ax.axhline(0, c="#7b8795", lw=1); ax.axvline(0, c="#7b8795", lw=1)
    ax.scatter(*values.mean(0)[[0,2]], c=RED, marker="*", s=160)
    ax.set(xlabel="旋转向量 X / °", ylabel="旋转向量 Z / °", title=title,
           xlim=(-12, 8), ylim=(-12, 8), aspect="equal")
    ax.legend(fontsize=9)
ax=axes[1,0]
ax.plot(g, angles, "o-", c=ORANGE, ms=4, lw=1, label=f"原始 {angles.mean():.3f}°")
ax.plot(g, centered_angles, "o-", c=BLUE, ms=4, lw=1, label=f"去偏置后 {centered_angles.mean():.3f}°")
ax.set(xlabel="采集组号", ylabel="SO(3)旋转角误差 / °", title="乘法移除平均旋转后的角度误差")
ax.legend()
ax=axes[1,1]
rmean=RV.mean(0)
ax.bar(range(3), rmean, color=[BLUE, ORANGE, "#8a65b5"], width=.55)
ax.errorbar(range(3), rmean, yerr=[rmean-ci_rotvec[:,0], ci_rotvec[:,1]-rmean],
            fmt="none", color="#263c51", capsize=6)
ax.axhline(0,c="#7b8795",lw=1)
ax.set(xticks=range(3),xticklabels=list("XYZ"),ylabel="旋转向量分量 / °",
       title="原始旋转向量均值与逐轴 bootstrap 95% 区间")
fig.suptitle("姿态误差也存在方向偏置",fontsize=20,fontweight="bold")
fig.supxlabel(f"平均旋转偏置角 {attitude['bias_angle_deg']:.3f}°；使用SO(3)平均与旋转复合移除偏置，未直接相减欧拉角。",fontsize=11,color="#596575")
fig.savefig(OUT/"attitude_centering.png",dpi=180)
plt.close(fig)

# Self-contained scientific 3D error-cloud visualization.
import plotly.graph_objects as go
from plotly.subplots import make_subplots
viewer=make_subplots(rows=1,cols=2,specs=[[{"type":"scene"},{"type":"scene"}]],
                    subplot_titles=["原始位置误差 · 视觉 − 动捕","去中心化残差 · 减去平均误差向量"])
u,v=np.mgrid[0:2*np.pi:36j,0:np.pi:22j]
sphere=np.stack([np.cos(u)*np.sin(v),np.sin(u)*np.sin(v),np.cos(v)],axis=-1)
ellipsoid=(sphere*np.sqrt(position["pca_eigenvalues"]))@position["pca_axes_columns"].T
for col,values,mean in [(1,E,bias),(2,Z,np.zeros(3))]:
    for flag,color,label in [(False,BLUE,"4个动捕点"),(True,ORANGE,"3个动捕点")]:
        mask=THREE==flag
        viewer.add_trace(go.Scatter3d(x=values[mask,0],y=values[mask,1],z=values[mask,2],
            mode="markers",marker=dict(size=5,color=color,symbol="square" if flag else "circle"),
            name=label,legendgroup=label,showlegend=col==1,
            text=[f"组{int(k)}" for k in g[mask]],
            hovertemplate="%{text}<br>X=%{x:.3f} mm<br>Y=%{y:.3f} mm<br>Z=%{z:.3f} mm<extra></extra>"),row=1,col=col)
    surface=ellipsoid+mean
    viewer.add_trace(go.Surface(x=surface[:,:,0],y=surface[:,:,1],z=surface[:,:,2],
        opacity=.13,colorscale=[[0,BLUE],[1,BLUE]],showscale=False,hoverinfo="skip",
        name="1σ主轴尺度椭球",showlegend=False),row=1,col=col)
    viewer.add_trace(go.Scatter3d(x=[0,mean[0]],y=[0,mean[1]],z=[0,mean[2]],
        mode="lines+markers",line=dict(color=RED,width=7),marker=dict(color=RED,size=4),
        name="平均误差向量",showlegend=col==1,hoverinfo="skip"),row=1,col=col)
    for j in range(3):
        direction=position["pca_axes_columns"][:,j]*position["pca_axis_std"][j]*2
        segment=np.array([mean-direction,mean+direction])
        viewer.add_trace(go.Scatter3d(x=segment[:,0],y=segment[:,1],z=segment[:,2],
            mode="lines",line=dict(color="#586f84",width=3),showlegend=False,hoverinfo="skip"),row=1,col=col)
scene=dict(xaxis=dict(title="X误差 / mm",range=[-17,17]),yaxis=dict(title="Y误差 / mm",range=[-17,17]),
           zaxis=dict(title="Z误差 / mm",range=[-17,17]),aspectmode="cube",
           camera=dict(eye=dict(x=1.4,y=-1.6,z=1.0)))
viewer.update_layout(scene=scene,scene2=scene,title=dict(text="相机误差方向与去中心化 · 29组有效数据",x=.5),
    margin=dict(l=12,r=12,b=30,t=95),height=800,paper_bgcolor="#f6f8fb",
    font=dict(family="Microsoft YaHei, sans-serif"),legend=dict(orientation="h",y=1.04),
    annotations=list(viewer.layout.annotations)+[
        dict(text="同一W坐标系、相同比例；去中心化只改变均值，椭球形状不变。椭球为1σ主轴尺度示意。",
             x=.5,y=-.05,xref="paper",yref="paper",showarrow=False,font=dict(size=13))])
viewer.write_html(OUT/"camera_error_cloud_3d.html",include_plotlyjs=True,full_html=True,
                  config=dict(displaylogo=False,responsive=True))

def fmtvec(x):
    return "(" + ", ".join(f"{v:+.3f}" for v in x) + ")"

def table(sample,name):
    return f"| {name} | {sample['n']} | {fmtvec(sample['bias_xyz'])} | {sample['raw_mean_length']:.3f} | {sample['centered_mean_length']:.3f} |"

lines=[
"# 相机误差的方向一致性与去中心化分析","",
"实验数据日期：2026-11-01。基于前次动捕配准所得29组有效视觉记录；第8组PnP失败排除，所有其余组均保留。","",
"## 结论","",
"位置误差存在明确的 −X 方向偏置，但不能用一个固定方向的平移偏差解释全部误差。去中心化后，误差在Y、Z方向的散布更大，且前两个主成分很接近，没有单一主导方向。姿态误差也有以 −X、−Z 为主的旋转偏置。","",
"## 1. 位置误差的方向偏置","",
"定义 e_i=t_vision,i−t_mocap,i，均在机翼坐标系W中，单位mm。平均偏置 b=(1/n)Σe_i，去中心化残差 r_i=e_i−b。这里减去的是**三维有符号误差向量的均值**，不是从每组误差模长中减去7.840 mm。","",
f"- 平均误差向量 b={fmtvec(bias)} mm，模长 {position['bias_norm']:.3f} mm。",
f"- 平均向量的单位方向为 {fmtvec(position['bias_direction'])}；主要朝 −X，同时有较弱 −Z 分量。",
f"- X误差为负的记录有 {int(position['negative_counts'][0])}/{n} 组（{100*position['negative_counts'][0]/n:.1f}%）；Y为正19组，Z为负19组。",
f"- 单位误差方向的平均合向量长度为 {position['unit_direction_resultant_length']:.3f}（1为完全同向；0为方向抵消）。这是描述统计，没有把它当作独立的显著性检验。","",
"| 分量 | 平均偏置 / mm | bootstrap逐轴95%区间 / mm | 去中心化后样本标准差 / mm |",
"|---|---:|---:|---:|"]
for j,a in enumerate("XYZ"):
    lines.append(f"| {a} | {bias[j]:+.3f} | [{ci_position[j,0]:+.3f}, {ci_position[j,1]:+.3f}] | {position['sample_std_xyz'][j]:.3f} |")
lines += [
"",
f"X轴正负号的双侧二项检验 p={position['sign_test_two_sided_p'][0]:.5f}，即便对XYZ三轴作Bonferroni校正约为 {min(1,3*position['sign_test_two_sided_p'][0]):.5f}。三维均值的Hotelling T²检验 p={position['hotelling_p']:.6f}。这些推断依赖样本独立等假设；本实验组序和姿态可能相关，所以应以本批数据的方向偏置证据解释，而不是宣称普适误差模型。",
"",
"## 2. 去中心化后误差降低多少","",
"| 指标 | 原始 | 去中心化后 |","|---|---:|---:|",
f"| 三维误差模长均值 / mm | {position['raw_mean_length']:.3f} | {position['centered_mean_length']:.3f} |",
f"| 三维RMSE / mm | {position['raw_rmse']:.3f} | {position['centered_rmse']:.3f} |",
f"| 平均误差向量模长 / mm | {position['bias_norm']:.3f} | 0（按定义） |","",
f"平均误差模长下降 {position['mean_length_reduction_pct']:.1f}%，RMSE下降 {position['rmse_reduction_pct']:.1f}%。平方误差有严格分解：MSE_original=||b||²+MSE_centered；固定偏置只占原始均方误差的 {100*position['bias_fraction_of_mse']:.1f}%，剩余 {100*(1-position['bias_fraction_of_mse']):.1f}% 不会被一个统一平移量消除。",
"",
f"上述是使用同批数据估计均值后的描述性结果。留一法中，每组只用其余28组估计偏置，平均误差为 {position['leave_one_out_mean_length']:.3f} mm，RMSE为 {position['leave_one_out_rmse']:.3f} mm。这也不是独立实验验证，不应把同批去中心结果直接视为可部署标定精度。",
"",
f"若要按这批数据做固定W系平移补偿，表达式为 t_corrected=t_vision−b，即给视觉位置加 {fmtvec(-bias)} mm。实际使用前应在独立数据上验证。",
"",
"![位置去中心化](position_centering.png)","",
"## 3. 去中心化后仍有方向性吗","",
"去中心化强制平均误差为零，因此不能仅凭均值为零断言误差没有方向结构。减均值不会改变协方差矩阵、逐轴标准差或PCA主轴形状，只会把点云中心搬到原点。",
"",
f"去中心化后的标准差 σ_X、σ_Y、σ_Z = {fmtvec(position['sample_std_xyz'])} mm。Y、Z明显比X更分散，Y和Z合计占坐标轴总方差约 {100*sum(np.diag(position['covariance'])[1:])/np.trace(position['covariance']):.1f}%。",
"",
"| 主轴 | 方差贡献 | 该轴样本标准差 / mm |","|---|---:|---:|"]
for j in range(3):
    lines.append(f"| PC{j+1} | {100*position['pca_variance_fraction'][j]:.2f}% | {position['pca_axis_std'][j]:.3f} |")
lines += [
"",
f"最长/最短主轴的标准差比为 {position['pca_long_to_short_std_ratio']:.3f}。PC1与PC2特征值之比仅 {position['pca_first_to_second_eigenvalue_ratio']:.3f}，两轴接近，单个主轴的精确朝向容易随抽样变化，不适合宣称存在稳定的一条误差方向。",
"",
f"PC1单位向量为 {fmtvec(position['pca_axes_columns'][:,0])}（约沿Y）；PC2为 {fmtvec(position['pca_axes_columns'][:,1])}（约沿Z）。主轴正负号没有物理意义，表示同一条轴。",
"",
"## 4. 三点动捕组是否影响结论","",
"| 动捕机器人标记点 | 组数 | 平均误差向量 / mm | 原始平均模长 / mm | 各子组分别去中心后 / mm |",
"|---|---:|---|---:|---:|",table(four,"4点"),table(three,"3点"),"",
"三点组为13、14、15、26、29，五组X误差全部为负。它们的平均偏置明显不同于四点组；但剔开三点组单看四点组，仍有18/24组X误差为负，说明 −X 偏向不完全由缺失点造成。分组偏置也提示不能默认所有姿态与动捕质量都适用同一个校正量。三点组仅5个样本，不宜据此建立单独标定模型。","",
"![偏置诊断](position_bias_diagnostics.png)","",
"## 5. 世界系固定偏置还是相机系固定偏置","",
"另用动捕相机姿态计算 e_C=R_gtᵀe_W，将每组误差分解到真实相机局部轴后，再求相机系统一均值。这与先在W系去中心化再旋转是两个不同的模型，不能混为一谈。",
"",
f"相机系平均误差为 {fmtvec(position_camera['bias_xyz'])} mm，模长 {position_camera['bias_norm']:.3f} mm。固定C系偏置只解释原始均方误差的 {100*position_camera['bias_fraction_of_mse']:.1f}%，去中心后平均误差 {position_camera['centered_mean_length']:.3f} mm、RMSE {position_camera['centered_rmse']:.3f} mm。W系固定偏置解释度为 {100*position['bias_fraction_of_mse']:.1f}%，在本批数据中更高；这提供坐标系层面的诊断线索，但不能唯一确定误差来自世界系标定、安装误差或视觉算法。",
"",
"## 6. 姿态误差去中心化","",
"姿态误差使用 D_i=R_vision,i R_gt,iᵀ。由于旋转不满足普通向量加法，正式去偏置采用SciPy的SO(3)弦均值B，令 D_centered,i=D_i Bᵀ，随后计算其旋转角；没有直接相减欧拉角。对应的视觉姿态补偿可写为 R_corrected=BᵀR_vision，其误差角与上述残差相同。",
"",
f"- 平均旋转偏置B的轴角向量（W系）为 {fmtvec(attitude['bias_rotation_vector_W_deg'])}°，偏置角 {attitude['bias_angle_deg']:.3f}°。向量分量不是三个欧拉角。",
f"- 原始误差旋转向量的X分量26/29为负，Z分量23/29为负，存在较明显的 −X、−Z 旋转取向。",
f"- 平均姿态误差从 {attitude['raw_mean_angle_deg']:.3f}° 降到 {attitude['centered_mean_angle_deg']:.3f}°；姿态RMSE从 {attitude['raw_rmse_deg']:.3f}° 降到 {attitude['centered_rmse_deg']:.3f}°。",
"- 旋转向量减去普通算术均值的结果也保存在JSON中作为小角度诊断；CSV和上述正式角度结果使用SO(3)乘法去偏置。SO(3)去偏置不要求其对数向量算术均值恰好为零。","",
"![姿态去中心化](attitude_centering.png)","",
"## 7. 限制、文件与复现","",
"- 继续沿用前次动捕对应关系、去重和前向约束。机翼标记配准残差约4.945 mm，且视觉与动捕采集非同步；方向偏置可能同时包含参考几何、安装和组间运动等因素，不能直接归因为相机自身。",
"- 未使用三维误差模长平均值对位置向量作平移。未删除原分析保留的任何有效视觉样本；第17组的红外异常不构成本次视觉剔除依据。",
"- Bootstrap按29组有放回抽样，20000次，种子20261101，区间为逐轴百分位区间；没有同时置信保证，也不包含动捕参考坐标的系统性不确定度。",
"- 所有去中心化、PCA和区间均为这批实验数据的统计描述；残差不能直接叫作纯随机噪声，可能仍包含随姿态变化的系统误差。",
"- [camera_error_cloud_3d.html](camera_error_cloud_3d.html)：离线双视窗三维误差点云，坐标表示误差分量，而不是机器人在机翼中的位置。",
"- centered_per_group.csv：逐组W系去中心位置残差、C系误差、留一法误差、SO(3)去偏置姿态结果。原始输入文件未修改。",
"- centering_summary.json：均值、协方差、PCA、分组与置信区间完整数值；validation_checks.json记录零均值、协方差不变、均方误差分解等核验。",
"- 复现：在原项目目录运行 python analyze_camera_centering.py。",""]
(OUT/"相机误差去中心化分析.md").write_text("\n".join(lines),encoding="utf-8")
print(json.dumps(serial(dict(position_W=position, attitude=attitude,
                            four_marker_bias=four["bias_xyz"], three_marker_bias=three["bias_xyz"],
                            validation=validation)),ensure_ascii=False,indent=2))
print("Output:", str(OUT))
