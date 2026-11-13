"""Reproduce the README experiment: python analyze_localization.py.

Inputs are immutable; all generated artifacts go to results/ by default.
Distances use mm internally, rotations use SO(3), samples pair by group order.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from marker_registration import (
    WING_MARKERS_MM, ROBOT_MARKERS_MM, T_R_C_MM, PLANE_Z_MM,
    enumerate_fits, inverse_rigid, load_marker_groups, transform_points,
    faces_reference_plane,
)


def native(value):
    if isinstance(value, np.ndarray):
        return native(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {k: native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    return value


def save_json(path, value):
    path.write_text(json.dumps(native(value), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def csv_transform(row, prefix):
    transform = np.array([[row[f'{prefix}_{i}{j}'] for j in range(4)] for i in range(4)], float)
    transform[:3, 3] *= 1000
    return transform


def statistics(values):
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    return dict(n=len(x), mean=np.mean(x), median=np.median(x),
                rmse=np.sqrt(np.mean(x**2)), std_sample=np.std(x, ddof=1),
                p95=np.percentile(x, 95), maximum=np.max(x), minimum=np.min(x))


def self_check_registration():
    """Exact recovery across all full permutations and all missing-marker cases."""
    from itertools import combinations, permutations
    expected = np.eye(4)
    expected[:3, :3] = Rotation.from_euler('xyz', [171, 21, -8], degrees=True).as_matrix()
    expected[:3, 3] = [310, 70, 35]
    count = 0
    for size in (3, 4):
        for subset in combinations(range(4), size):
            for order in permutations(subset):
                observed = transform_points(expected, ROBOT_MARKERS_MM[list(order)])
                best = enumerate_fits(ROBOT_MARKERS_MM, observed)[0]
                assert best['mapping'] == order
                assert np.allclose(best['T'], expected, atol=1e-8)
                count += 1
    return dict(exact_synthetic_correspondence_recoveries=count, passed=True)


def analyse(root, output):
    vision = pd.read_csv(root / 'positioning_records.csv')
    rigid = pd.read_csv(root / 'rigid_body_test_1.csv')
    groups = load_marker_groups(root / 'markerPos_test_1.mat')
    assert len(groups) == len(vision) == len(rigid) == 30
    assert vision.group_id.tolist() == list(range(1, 31))
    row_results, poses, candidate_rows, wing_edges = [], [], [], []
    validation = self_check_registration()
    checks = dict(max_csv_extrinsic_chain_error_mm=0., max_csv_robot_chain_error_mm=0.,
                  max_so3_orthogonality_error=0., max_ir_plane_point_residual_mm=0.,
                  max_ir_distance_coefficient_difference_mm=0.,
                  max_ir_normal_norm_difference=0., max_inverse_roundtrip_error_mm=0.,
                  all_ir_hits_positive_robot_z=True)
    lags = (pd.to_datetime(rigid.TimeStamp).dt.tz_localize('Asia/Shanghai') -
            pd.to_datetime(vision.snapshot_at)).dt.total_seconds()

    for index, (group, (_, row)) in enumerate(zip(groups, vision.iterrows())):
        gid = int(row.group_id)
        wing_fits = enumerate_fits(WING_MARKERS_MM, group['wing'])
        wing = wing_fits[0]
        T_W_M = inverse_rigid(wing['T'])
        robot_fits = enumerate_fits(ROBOT_MARKERS_MM, group['robot'])
        for rank, candidate in enumerate(robot_fits, start=1):
            candidate['T_W_R'] = T_W_M @ candidate['T']
            candidate['front_facing'] = faces_reference_plane(candidate['T_W_R'])
            candidate_rows.append(dict(group_id=gid, residual_rank=rank,
                observed_to_robot_marker_ids_1based=' '.join(str(j + 1) for j in candidate['mapping']),
                rms_mm=candidate['rms_mm'], front_facing=candidate['front_facing']))
        front_fits = [c for c in robot_fits if c['front_facing']]
        if not front_fits:
            raise ValueError(f'Group {gid}: no pose faces the reference plane')
        best, runner = front_fits[:2]
        T_W_R = best['T_W_R']
        T_W_C = T_W_R @ T_R_C_MM
        for transform in (wing['T'], T_W_R, T_W_C):
            rot = transform[:3, :3]
            assert abs(np.linalg.det(rot) - 1) < 1e-8
            checks['max_so3_orthogonality_error'] = max(checks['max_so3_orthogonality_error'],
                np.max(np.abs(rot.T @ rot - np.eye(3))))
        camera_gt, robot_gt = T_W_C[:3, 3], T_W_R[:3, 3]
        record = dict(group_id=gid, mat_row_1based=group['mat_row_1based'],
            snapshot_at=row.snapshot_at, rigid_body_timestamp=rigid.iloc[index].TimeStamp,
            rigid_body_lag_s=lags.iloc[index], raw_robot_marker_count=len(group['raw'])-3,
            unique_robot_marker_count=len(group['robot']),
            duplicate_marker_indices_1based=' '.join(map(str, group['duplicate_marker_indices'])),
            wing_observed_to_model_ids_1based=' '.join(str(j+1) for j in wing['mapping']),
            robot_observed_to_model_ids_1based=' '.join(str(j+1) for j in best['mapping']),
            robot_missing_model_ids_1based=' '.join(str(j+1) for j in range(4) if j not in best['mapping']),
            wing_fit_rms_mm=wing['rms_mm'], robot_fit_rms_mm=best['rms_mm'],
            robot_runner_up_front_rms_mm=runner['rms_mm'],
            robot_front_margin_mm=runner['rms_mm']-best['rms_mm'],
            unrestricted_fit_rms_mm=robot_fits[0]['rms_mm'],
            unrestricted_fit_front_facing=robot_fits[0]['front_facing'],
            vision_status=row.vision_status, vision_valid=False,
            ir_valid=False, ir_distance_gt_mm=abs(robot_gt[2]-PLANE_Z_MM),
            ir_distance_measured_mm=np.nan, ir_distance_error_signed_mm=np.nan,
            ir_distance_error_abs_mm=np.nan, ir_normal_error_deg=np.nan,
            camera_position_error_mm=np.nan, camera_rotation_error_deg=np.nan)
        for axis, gt in zip('xyz', camera_gt):
            record[f'camera_gt_{axis}_mm'] = gt
            record[f'camera_vision_{axis}_mm'] = np.nan
            record[f'camera_error_{axis}_mm'] = np.nan
        camera_v = csv_transform(row, 'T_W_C')
        robot_v = csv_transform(row, 'T_W_R')
        valid_vision = row.vision_status == 'VALID' and np.isfinite(camera_v).all()
        if valid_vision:
            record['vision_valid'] = True
            error = camera_v[:3, 3] - camera_gt
            record['camera_position_error_mm'] = np.linalg.norm(error)
            error_rot = camera_v[:3, :3] @ T_W_C[:3, :3].T
            record['camera_rotation_error_deg'] = np.degrees(Rotation.from_matrix(error_rot).magnitude())
            rotvec = Rotation.from_matrix(error_rot).as_rotvec(degrees=True)
            for j, axis in enumerate('xyz'):
                record[f'camera_vision_{axis}_mm'] = camera_v[j, 3]
                record[f'camera_error_{axis}_mm'] = error[j]
                record[f'camera_rotation_vector_{axis}_deg'] = rotvec[j]
            checks['max_csv_extrinsic_chain_error_mm'] = max(checks['max_csv_extrinsic_chain_error_mm'],
                np.max(np.abs(csv_transform(row, 'T_R_C')-T_R_C_MM)))
            checks['max_csv_robot_chain_error_mm'] = max(checks['max_csv_robot_chain_error_mm'],
                np.max(np.abs(robot_v @ T_R_C_MM-camera_v)))
        pose = dict(group_id=gid, T_W_M_mm=T_W_M, T_W_R_gt_mm=T_W_R, T_W_C_gt_mm=T_W_C,
            T_W_R_vision_mm=robot_v if valid_vision else None,
            T_W_C_vision_mm=camera_v if valid_vision else None,
            wing_observed_W_mm=transform_points(T_W_M, group['wing']),
            robot_observed_W_mm=transform_points(T_W_M, group['robot']),
            robot_fitted_W_mm=transform_points(T_W_R, ROBOT_MARKERS_MM[list(best['mapping'])]))
        normal = row[['plane_nx_R', 'plane_ny_R', 'plane_nz_R']].to_numpy(float)
        valid_ir = (str(row.plane_valid).lower() == 'true' and np.isfinite(normal).all()
                    and np.isfinite(row.plane_d_R_m) and np.isfinite(row.plane_distance_mm)
                    and np.linalg.norm(normal) > 1e-12)
        if valid_ir:
            norm = np.linalg.norm(normal)
            normal /= norm
            distance_coefficient = row.plane_d_R_m * 1000 / norm
            normal_gt_R = T_W_R[:3, :3].T @ np.array([0., 0., 1.])
            if np.dot(normal, normal_gt_R) < 0:
                normal, distance_coefficient = -normal, -distance_coefficient
            measured_distance = float(row.plane_distance_mm)
            angle = np.degrees(np.arccos(np.clip(abs(np.dot(normal, normal_gt_R)), 0, 1)))
            signed_error = measured_distance-record['ir_distance_gt_mm']
            record.update(ir_valid=True, ir_distance_measured_mm=measured_distance,
                ir_distance_error_signed_mm=signed_error, ir_distance_error_abs_mm=abs(signed_error),
                ir_normal_error_deg=angle)
            points_R = np.array([[row[f's{s}_point_{a}_m']*1000 for a in 'xyz'] for s in (1,2,3)])
            assert np.isfinite(points_R).all()
            checks['all_ir_hits_positive_robot_z'] &= bool((points_R[:, 2] > 0).all())
            checks['max_ir_normal_norm_difference'] = max(checks['max_ir_normal_norm_difference'], abs(norm-1))
            checks['max_ir_plane_point_residual_mm'] = max(checks['max_ir_plane_point_residual_mm'],
                np.max(np.abs(points_R @ normal + distance_coefficient)))
            checks['max_ir_distance_coefficient_difference_mm'] = max(checks['max_ir_distance_coefficient_difference_mm'],
                abs(measured_distance-abs(distance_coefficient)))
            normal_W = T_W_R[:3, :3] @ normal
            pose.update(ir_normal_W=normal_W, ir_normal_gt_R=normal_gt_R,
                ir_projection_W_mm=robot_gt-distance_coefficient*normal_W,
                ir_true_projection_W_mm=np.array([robot_gt[0],robot_gt[1],PLANE_Z_MM]),
                ir_points_W_mm=transform_points(T_W_R, points_R))
        roundtrip = transform_points(inverse_rigid(T_W_R), transform_points(T_W_R, ROBOT_MARKERS_MM))
        checks['max_inverse_roundtrip_error_mm'] = max(checks['max_inverse_roundtrip_error_mm'],
            np.max(np.abs(roundtrip-ROBOT_MARKERS_MM)))
        # Reorder measured wing points into nominal marker order for edge diagnostics.
        wing_reordered = np.empty((3,3))
        wing_reordered[list(wing['mapping'])] = group['wing']
        for a,b in ((0,1),(0,2),(1,2)):
            nominal = np.linalg.norm(WING_MARKERS_MM[a]-WING_MARKERS_MM[b])
            measured = np.linalg.norm(wing_reordered[a]-wing_reordered[b])
            wing_edges.append(dict(group_id=gid, edge=f'W{a+1}-W{b+1}', nominal_mm=nominal,
                measured_mm=measured, difference_mm=measured-nominal))
        pose['metrics'] = record
        row_results.append(record)
        poses.append(pose)

    frame = pd.DataFrame(row_results)
    candidate_frame = pd.DataFrame(candidate_rows)
    assert (candidate_frame.groupby('group_id').size() == 24).all()
    for gid in frame.loc[frame.unique_robot_marker_count == 3, 'group_id']:
        mappings = [tuple(map(int, mapping.split())) for mapping in
                    candidate_frame.loc[candidate_frame.group_id == gid,
                                        'observed_to_robot_marker_ids_1based']]
        subsets = {tuple(sorted(mapping)) for mapping in mappings}
        assert len(subsets) == 4
        assert all(sum(tuple(sorted(mapping)) == subset for mapping in mappings) == 6
                   for subset in subsets)
    validation['all_groups_have_24_candidates'] = True
    validation['all_3_marker_groups_have_4_subsets_times_6_permutations'] = True
    metric_keys = ['camera_position_error_mm','camera_rotation_error_deg',
                   'ir_distance_error_abs_mm','ir_normal_error_deg']
    summary = {key: statistics(frame[key]) for key in metric_keys}
    summary['ir_distance_signed_bias_mm'] = frame.ir_distance_error_signed_mm.mean()
    summary['camera_signed_bias_xyz_mm'] = [frame[f'camera_error_{a}_mm'].mean() for a in 'xyz']
    summary['wing_fit_rms_mm'] = statistics(frame.wing_fit_rms_mm)
    summary['robot_fit_rms_mm'] = statistics(frame.robot_fit_rms_mm)
    summary['counts'] = dict(total=30, vision_valid=int(frame.vision_valid.sum()),
        ir_valid=int(frame.ir_valid.sum()), unique_3_marker_groups=frame.loc[frame.unique_robot_marker_count==3,'group_id'].tolist(),
        unrestricted_flip_groups=frame.loc[~frame.unrestricted_fit_front_facing,'group_id'].tolist())
    # Diagnostic only: do not silently remove outliers from the headline metric.
    summary['ir_without_group17_diagnostic_only'] = {
        key: statistics(frame.loc[frame.group_id!=17,key])
        for key in ['ir_distance_error_abs_mm','ir_normal_error_deg']}
    validation.update(checks)
    validation['checks_passed'] = bool(
        checks['max_csv_robot_chain_error_mm'] < 1e-5 and
        checks['max_csv_extrinsic_chain_error_mm'] < 1e-8 and
        checks['max_so3_orthogonality_error'] < 1e-8 and
        checks['max_ir_plane_point_residual_mm'] < 1e-5 and
        checks['max_ir_distance_coefficient_difference_mm'] < 1e-5 and
        checks['max_inverse_roundtrip_error_mm'] < 1e-8 and
        checks['all_ir_hits_positive_robot_z'])
    assert validation['checks_passed'], validation
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output/'per_group_errors.csv',index=False,encoding='utf-8-sig')
    candidate_frame.to_csv(output/'robot_correspondence_candidates.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(wing_edges).to_csv(output/'wing_marker_edge_diagnostics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame([dict(metric=key,**summary[key]) for key in metric_keys]).to_csv(
        output/'summary_metrics.csv',index=False,encoding='utf-8-sig')
    save_json(output/'summary.json',summary)
    save_json(output/'poses.json',poses)
    save_json(output/'validation_checks.json',validation)
    save_json(output/'software_versions.json', dict(python=platform.python_version(),
        **{name: importlib.metadata.version(name) for name in
           ['numpy','scipy','pandas','matplotlib','pyvista','plotly']}))
    save_json(output/'input_manifest.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [
        root/'README.md',root/'positioning_records.csv',root/'markerPos_test_1.mat',
        root/'rigid_body_test_1.csv',root/'robot_general_simplified.stl',root/'wing_general_simplified.STL']})
    return frame, summary, poses, pd.DataFrame(wing_edges)


def make_plots(frame, summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
        'axes.unicode_minus':False, 'font.size':11, 'axes.spines.top':False,
        'axes.spines.right':False, 'axes.grid':True, 'grid.alpha':.18,
        'figure.facecolor':'#f6f8fb','axes.facecolor':'white','savefig.facecolor':'#f6f8fb'})
    blue, orange, red = '#087ea4','#e88b26','#c44755'
    fig, axes = plt.subplots(2,2,figsize=(15,9),layout='constrained')
    names = [('camera_position_error_mm','相机位置误差','mm',blue),
             ('camera_rotation_error_deg','相机姿态误差（SO(3) 旋转角）','°',blue),
             ('ir_distance_error_abs_mm','红外距离绝对误差','mm',orange),
             ('ir_normal_error_deg','红外平面法向夹角误差','°',orange)]
    for ax,(key,title,unit,color) in zip(axes.flat,names):
        valid = frame[key].notna()
        stats = summary[key]
        ax.vlines(frame.loc[valid,'group_id'],0,frame.loc[valid,key],color=color,alpha=.22,lw=2)
        ax.scatter(frame.loc[valid,'group_id'],frame.loc[valid,key],s=43,c=color,zorder=3)
        ax.axhline(stats['mean'],c=red,ls='--',lw=1.4,label=f"平均 {stats['mean']:.2f} {unit}")
        ax.set(title=f"{title}   ·   n={stats['n']}",xlabel='采集组号',ylabel=f'误差 / {unit}',xlim=(.3,30.7))
        ax.set_xticks([1,5,10,15,20,25,30]); ax.set_ylim(bottom=0)
        ax.legend(loc='upper right',framealpha=.95)
        missing = frame.loc[~valid,'group_id']
        ax.scatter(missing,np.zeros(len(missing)),marker='x',c='#9199a4',s=25,zorder=4)
        if key.startswith('ir_'):
            y = frame.loc[frame.group_id==17,key].iloc[0]
            ax.annotate('第17组',xy=(17,y),xytext=(-48,-24),textcoords='offset points',
                arrowprops={'arrowstyle':'-','color':red},color=red)
    fig.suptitle('视觉与红外定位精度 · 以动捕为基准',fontsize=20,fontweight='bold')
    fig.supxlabel('灰色 × 表示无有效测量；保留全部有效组，包括第17组。',fontsize=11,color='#596575')
    fig.savefig(output/'error_dashboard.png',dpi=180)
    fig.savefig(output/'error_dashboard.svg')
    plt.close(fig)

    fig, axes = plt.subplots(2,2,figsize=(15,9),layout='constrained')
    ax=axes[0,0]
    for axis,color in zip('xyz',[blue,orange,'#8a65b5']):
        ax.plot(frame.group_id,frame[f'camera_error_{axis}_mm'],'o-',ms=4,lw=1.2,c=color,label=axis.upper())
    ax.axhline(0,c='#778899',lw=1); ax.set(title='相机位置分量误差（视觉 − 动捕）',xlabel='采集组号',ylabel='误差 / mm'); ax.legend(ncol=3)
    ax=axes[0,1]
    for axis,color in zip('xyz',[blue,orange,'#8a65b5']):
        ax.plot(frame.group_id,frame[f'camera_rotation_vector_{axis}_deg'],'o-',ms=4,lw=1.2,c=color,label=axis.upper())
    ax.axhline(0,c='#778899',lw=1); ax.set(title='姿态误差旋转向量（在机翼坐标系中）',xlabel='采集组号',ylabel='分量 / °'); ax.legend(ncol=3)
    ax=axes[1,0]; ir=frame[frame.ir_valid]
    ax.scatter(ir.ir_distance_gt_mm,ir.ir_distance_measured_mm,c=orange,s=48)
    bounds=[min(ir.ir_distance_gt_mm.min(),ir.ir_distance_measured_mm.min())-5,max(ir.ir_distance_gt_mm.max(),ir.ir_distance_measured_mm.max())+5]
    ax.plot(bounds,bounds,'--',c='#7e8998'); ax.set(xlim=bounds,ylim=bounds,title='红外垂直距离：测量与动捕基准',xlabel='动捕基准距离 / mm',ylabel='红外测量距离 / mm')
    for gid in (1,17,19):
        r=ir[ir.group_id==gid].iloc[0]; ax.annotate(str(gid),(r.ir_distance_gt_mm,r.ir_distance_measured_mm),xytext=(5,5),textcoords='offset points')
    ax=axes[1,1]
    ax.bar(ir.group_id,ir.ir_distance_error_signed_mm,color=[red if x<0 else blue for x in ir.ir_distance_error_signed_mm])
    ax.axhline(0,c='#778899',lw=1); ax.axhline(summary['ir_distance_signed_bias_mm'],ls='--',c=orange,label=f"平均偏差 {summary['ir_distance_signed_bias_mm']:.2f} mm")
    ax.set(title='红外距离有符号误差（红外 − 动捕）',xlabel='采集组号',ylabel='误差 / mm',xlim=(.3,30.7)); ax.legend()
    fig.suptitle('误差分解与距离一致性',fontsize=20,fontweight='bold')
    fig.savefig(output/'error_components.png',dpi=180); plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    axes[0].plot(frame.group_id,frame.wing_fit_rms_mm,'o-',c=blue,label='机翼（三点）')
    axes[0].plot(frame.group_id,frame.robot_fit_rms_mm,'o-',c=orange,label='机器人（去重后）')
    three=frame[frame.unique_robot_marker_count==3]
    axes[0].scatter(three.group_id,three.robot_fit_rms_mm,marker='s',s=85,facecolors='none',edgecolors=red,label='仅3个机器人点')
    axes[0].set(title='标记点拟合残差（不等于定位误差）',xlabel='采集组号',ylabel='点残差 RMS / mm'); axes[0].legend()
    axes[1].plot(frame.group_id,frame.robot_front_margin_mm,'o-',c=blue)
    axes[1].set(title='正向候选：次优与最优拟合残差之差',xlabel='采集组号',ylabel='残差差值 / mm')
    fig.suptitle('动捕注册质量与对应关系诊断',fontsize=19,fontweight='bold')
    fig.savefig(output/'registration_diagnostics.png',dpi=180); plt.close(fig)


def make_report(frame, summary, edges, output):
    names = [('camera_position_error_mm','相机位置（三维欧氏距离）','mm'),
             ('camera_rotation_error_deg','相机姿态（SO(3) 旋转角）','°'),
             ('ir_distance_error_abs_mm','红外距离绝对误差','mm'),
             ('ir_normal_error_deg','红外平面法向夹角误差','°')]
    lines=['# 单目 AprilTag 与红外定位精度分析','',
        '实验日期：2026-11-01。按 README 给出的几何参数与用户确认的组序对应关系计算；没有用视觉结果校准或挑选动捕姿态。','',
        '## 结果','', '| 指标 | 有效组数 | 平均误差 | RMSE | 中位数 | P95 | 最大值 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for key,name,unit in names:
        s=summary[key]
        lines.append(f"| {name} | {s['n']} | {s['mean']:.3f} {unit} | {s['rmse']:.3f} {unit} | {s['median']:.3f} {unit} | {s['p95']:.3f} {unit} | {s['maximum']:.3f} {unit} |")
    bias=summary['camera_signed_bias_xyz_mm']
    lines += ['',f"相机位置的平均有符号偏差（视觉 − 动捕，机翼 XYZ）为 ({bias[0]:.3f}, {bias[1]:.3f}, {bias[2]:.3f}) mm。",
        f"红外距离的平均有符号偏差为 {summary['ir_distance_signed_bias_mm']:.3f} mm；负号表示距离测短。",'',
        '以上平均值均为逐组误差的算术平均。位置平均误差是三维误差模长的均值，不是平均偏差向量的模长；姿态使用相对旋转角，不直接相减欧拉角。P95 使用 NumPy 默认线性插值。', '',
        '![误差总览](error_dashboard.png)','',
        '## 配准、单位与误差定义','',
        '1. MAT 中实际有 30 个非空组，位于第 1～30 行第 1 列，其余单元为空。MAT 坐标以 m 记录，统一乘 1000 变成 mm；README 标记坐标、STL 和基准平面均按 mm 使用。CSV 的变换矩阵平移也统一转为 mm。',
        '2. 用户已确认：按组序对应 group_id 1～30，视觉世界系 W 即机翼模型系。前三个动捕点枚举 3! 种对应，用只允许 det(R)=+1 的 Kabsch 刚体拟合求 T_M_W；每组分别求解，再取逆得到 T_W_M。没有缩放、反射或用视觉结果做二次对齐。',
        '3. 机器人点先去除完全重复的记录。4 个唯一点枚举 4!=24 种对应；3 个唯一点明确枚举全部 C(4,3)=4 个子集，每个子集再枚举 3!=6 种对应，总计 24 个候选。每组全部候选及其残差均保存在 robot_correspondence_candidates.csv。',
        '4. 机器人标记点接近共面且近似左右对称，残差最小的无约束解常将机器人翻转约 180°。已记录的红外命中点全部位于 +z_R，README 指出测量平面在机器人前方，因此添加物理约束：从机器人原点沿 +z_R 发出的射线必须在正向与 z_W=-75 相交，即 s=(-75-t_z)/R_zz > 0。在满足此条件的候选中选残差最小者。这一步只用标记点、已知平面及传感器朝向，未引用视觉姿态。若实验中基准面并非位于 +z_R 一侧，应修改此约束后重算。',
        '5. T_A_B 表示把 B 系坐标变换到 A 系。T_W_R=T_W_M·T_M_R，T_W_C=T_W_R·T_R_C；相机偏移为 (0,-29.5,4) mm，姿态与机器人相同。',
        '6. 相机位置误差 e_p=||t_W_C(vision)-t_W_C(mocap)||₂。姿态误差 e_R=acos(clip((tr(R_vision·R_gtᵀ)-1)/2,-1,1))，以度表示；代码用 SciPy 旋转角实现以提高数值稳定性。',
        '7. 红外基准是无限平面 z_W=-75 mm。原点至平面真实垂直距离 d_gt=|t_W_R,z+75|。真实法向在机器人系中为 n_gt,R=R_W_Rᵀ·(0,0,1)。距离误差为 |d_IR-d_gt|；角度误差为 acos(|n_IR·n_gt,R|)，归一化法向后计算，故平面法向的正负号不影响结果。这里统计的是三传感器拟合平面的垂直距离，不是单个器件的沿射线量程。', '',
        '## 数据质量与适用范围','',
        '- 第 8 组视觉状态为 PnP failed：不计入视觉误差，仍参与有效红外统计。',
        '- 红外有效组为（由实际 CSV 自动生成）。',
        '- 第 15 组有 8 个非空标记记录：最后两个点分别重复前面机器人点；第 26 组最后一个点重复。删除重复后，只有 3 个机器人点的组为 13、14、15、26、29；均遍历全部缺失点可能性。',
        f"- 无约束最小残差落入翻转分支的组有 {len(summary['counts']['unrestricted_flip_groups'])} 组；加入前向约束后，最小次优残差差值为 {frame.robot_front_margin_mm.min():.3f} mm。该差值只用于诊断，不是概率或置信区间。", '',
        '### 机翼标记几何存在系统性不一致','',
        '| 边 | README 名义边长 / mm | 动捕平均边长 / mm | 平均差值 / mm |','|---|---:|---:|---:|']
    # Replace the validity paragraph with the actual valid group list.
    ir_groups='、'.join(map(str,frame.loc[frame.ir_valid,'group_id']))
    lines=[f'- 红外有效组共 {int(frame.ir_valid.sum())} 组：{ir_groups}；其余 {len(frame)-int(frame.ir_valid.sum())} 组不补值，不计入红外统计。' if s.startswith('- 红外有效组为') else s for s in lines]
    for edge,g in edges.groupby('edge',sort=False):
        lines.append(f"| {edge} | {g.nominal_mm.iloc[0]:.3f} | {g.measured_mm.mean():.3f} | {g.difference_mm.mean():.3f} |")
    wing=summary['wing_fit_rms_mm']; robot=summary['robot_fit_rms_mm']
    ir17=frame[frame.group_id==17].iloc[0]
    lines += ['',f"机翼刚体拟合的平均点残差 RMS 为 {wing['mean']:.3f} mm（{wing['minimum']:.3f}～{wing['maximum']:.3f} mm），机器人为 {robot['mean']:.3f} mm。边长不一致不能由刚体变换消除。这表示当前结果包含名义标记坐标、标记安装、动捕几何及组间采集变化等共同影响，不能将误差全部归因于视觉或红外本身。建议复核三处机翼标记的实际球心坐标，再使用同一脚本重算。",'',
        '### 时间与异常组','',
        f"rigid_body_test_1.csv 有 30 条记录，其时间戳比对应视觉快照晚 {frame.rigid_body_lag_s.min():.3f}～{frame.rigid_body_lag_s.max():.3f} s（中位数 {frame.rigid_body_lag_s.median():.3f} s）。MAT 本身没有时间戳，因此这里只按用户确认的采集组序配对，不能进行插值或验证同步。需要每组两次采集之间机器人保持静止；若发生运动，组间位移也会计入误差。该刚体 CSV 的原点/外参未给出，故只用于采集顺序和时间差诊断，不把其中位姿直接当作机翼/机器人模型系真值。",'',
        f"第 17 组红外距离测量 {ir17.ir_distance_measured_mm:.3f} mm，动捕基准 {ir17.ir_distance_gt_mm:.3f} mm，有符号误差 {ir17.ir_distance_error_signed_mm:.3f} mm，法向误差 {ir17.ir_normal_error_deg:.3f}°，是明显异常组，已保留在主统计中。需要结合遮挡、是否打到其他表面、传感器标定及组间是否移动排查，现有文件不能判定具体原因。",'',
        f"仅作为敏感性诊断，去掉第 17 组后，红外距离 MAE 为 {summary['ir_without_group17_diagnostic_only']['ir_distance_error_abs_mm']['mean']:.3f} mm，法向平均误差为 {summary['ir_without_group17_diagnostic_only']['ir_normal_error_deg']['mean']:.3f}°（各 n={summary['ir_without_group17_diagnostic_only']['ir_distance_error_abs_mm']['n']}）；这不替代上面的主统计。",'',
        '![误差分解](error_components.png)','', '![配准诊断](registration_diagnostics.png)','',
        '## 三维可视化与复现','',
        '- 打开 [interactive_3d.html](interactive_3d.html)：离线交互式 STL 场景，拖动组号滑块或按左右键逐组查看；青色为动捕，橙色为视觉，红色为红外估计平面。红外平面通过动捕姿态放到 W 系，以单独显示红外误差。第 8 组隐藏视觉模型，无效红外组隐藏红外平面。可以切换局部/整翼视角和各层显示。',
        '- [scene_overview.png](scene_overview.png) 为机翼模型、参考平面与相机位置分布；[scene_group17.png](scene_group17.png) 为第 17 组三维局部对比。STL 显示使用 PyVista 读取与简化；原始 STL 和输入数据均未修改。简化只影响显示，不参与任何误差计算。',
        '- CSV 使用 UTF-8 BOM，可用 Excel 打开。per_group_errors.csv 是逐组结果；summary_metrics.csv 是统计表；poses.json 包含所有动捕/视觉变换（平移 mm）、标记点与红外平面，方便复核。',
        '- input_manifest.json 记录全部输入文件的 SHA-256。validation_checks.json 记录 48 次合成数据对应关系恢复检验以及外参链、正交性、逆变换、红外平面点一致性检验。', '',
        '在项目目录运行：','', '```powershell','python -m pip install -r requirements.txt','python analyze_localization.py','```','',
        '默认会重新生成 results/。只重算数值与二维图：`python analyze_localization.py --skip-3d`。独立桌面三维查看器：`python view_3d.py --group 17`。','',
        '![三维总览](scene_overview.png)','', '![第17组局部对比](scene_group17.png)','']
    (output/'分析报告.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='results')
    parser.add_argument('--skip-3d',action='store_true')
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    output=(root/args.output).resolve()
    frame,summary,poses,edges=analyse(root,output)
    make_plots(frame,summary,output)
    if not args.skip_3d:
        from visualization_3d import build_visualizations
        build_visualizations(root,output,poses)
    make_report(frame,summary,edges,output)
    print(json.dumps(native(summary),ensure_ascii=False,indent=2))
    print(f'Results: {output}')


if __name__=='__main__':
    main()
