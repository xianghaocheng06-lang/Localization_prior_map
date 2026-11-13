"""Fit infrared calibration to motion capture: python calibrate_infrared.py.

Exclude groups using the ORIGINAL plane-distance absolute error (>10 mm),
before fitting. Produce a plane-output correction and independent raw-range
calibrations for all three sensors; these are alternative pipelines.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from marker_registration import (
    PLANE_Z_MM, ROBOT_MARKERS_MM, WING_MARKERS_MM, enumerate_fits,
    faces_reference_plane, inverse_rigid, load_marker_groups,
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
        return {key: native(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [native(item) for item in value]
    return value


def save_json(path, value):
    path.write_text(json.dumps(native(value), ensure_ascii=False, indent=2,
                               allow_nan=False), encoding="utf-8")


def describe(error):
    error = np.asarray(error, float)
    error = error[np.isfinite(error)]
    if not len(error):
        raise ValueError("No finite errors to summarize")
    absolute = np.abs(error)
    return dict(n=len(error), bias=float(error.mean()), mae=float(absolute.mean()),
                rmse=float(np.sqrt(np.mean(error**2))), median_abs=float(np.median(absolute)),
                p95_abs=float(np.percentile(absolute, 95)), max_abs=float(absolute.max()))


def fit_model(x, y, model="linear"):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if model == "offset":
        return np.array([1., np.mean(y-x)])
    degree = {"linear": 1, "quadratic": 2}[model]
    if len(np.unique(x)) <= degree:
        raise ValueError("Insufficient distinct readings to fit calibration")
    return np.polyfit(x, y, degree)


def leave_one_out(x, y, model="linear"):
    # Each held-out group's motion-capture target is absent from its fit.
    return np.array([np.polyval(fit_model(np.delete(x, k), np.delete(y, k), model), x[k])
                     for k in range(len(x))])


def plane_from_points(points):
    points = np.asarray(points, float)
    normal = np.cross(points[:, 1]-points[:, 0], points[:, 2]-points[:, 0])
    lengths = np.linalg.norm(normal, axis=1)
    if np.any(lengths < 1e-8):
        raise ValueError("Infrared hit points cannot determine a plane")
    normal /= lengths[:, None]
    coefficient = -np.einsum("ij,ij->i", normal, points[:, 0])
    flip = coefficient < 0
    normal[flip] *= -1
    coefficient[flip] *= -1
    return normal, coefficient


def angle_error(normals, truth):
    dot = np.einsum("ij,ij->i", normals, truth)
    return np.degrees(np.arccos(np.clip(np.abs(dot), 0., 1.)))


def load_reference(root, raw):
    """Rebuild the existing mocap reference directly from immutable markers."""
    groups = load_marker_groups(root / "markerPos_test_1.mat")
    if len(groups) != len(raw) or raw.group_id.tolist() != list(range(1, len(raw)+1)):
        raise ValueError("Raw records and marker groups must pair in acquisition order")
    transforms = []
    for group in groups:
        wing = enumerate_fits(WING_MARKERS_MM, group["wing"])[0]
        T_W_M = inverse_rigid(wing["T"])
        candidates = enumerate_fits(ROBOT_MARKERS_MM, group["robot"])
        forward = [T_W_M @ c["T"] for c in candidates
                   if faces_reference_plane(T_W_M @ c["T"])]
        if not forward:
            raise ValueError("No motion-capture robot pose faces the reference plane")
        transforms.append(forward[0])
    transforms = np.asarray(transforms)
    normals = transforms[:, 2, :3]
    coefficients = transforms[:, 2, 3]-PLANE_Z_MM
    return transforms, normals, coefficients


def recover_geometry(raw, included, sensor):
    """Recover the fixed recorded ray p_R = origin + calibrated_range*direction.

    This recovers existing mounting geometry, not new extrinsics. Only retained
    groups are used. Independent available rows check the recovered geometry.
    """
    prefix = f"s{sensor}_"
    columns = [prefix+f"point_{axis}_m" for axis in "xyz"]
    old = raw.loc[included, prefix+"calibrated_mm"].to_numpy(float)
    points = raw.loc[included, columns].to_numpy(float)*1000
    direction, origin = np.linalg.lstsq(np.column_stack([old, np.ones(len(old))]),
                                       points, rcond=None)[0]
    original_length = np.linalg.norm(direction)
    direction /= original_length
    available = (raw[prefix+"status"].eq("VALID") &
                 raw[[prefix+"calibrated_mm", *columns]].notna().all(axis=1))
    recorded = raw.loc[available, columns].to_numpy(float)*1000
    recovered = origin + raw.loc[available, prefix+"calibrated_mm"].to_numpy()[:, None]*direction
    residual = np.max(np.abs(recorded-recovered))
    if residual > 1e-5 or abs(original_length-1) > 1e-6:
        raise ValueError(f"Sensor {sensor}: inconsistent recorded mounting geometry")
    return origin, direction, float(residual)


def calibrate(root, output, threshold):
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("Exclusion threshold must be finite and nonnegative")
    raw = pd.read_csv(root / "positioning_records.csv")
    transforms, normals_gt, coefficients_gt = load_reference(root, raw)
    gt = np.abs(coefficients_gt)
    normals_old = raw[[f"plane_n{axis}_R" for axis in "xyz"]].to_numpy(float)
    measured = raw.plane_distance_mm.to_numpy(float)
    normal_lengths = np.linalg.norm(normals_old, axis=1)
    valid = (raw.plane_valid.astype(str).str.lower().eq("true").to_numpy() &
             np.isfinite(normals_old).all(axis=1) & np.isfinite(measured) &
             np.isfinite(raw.plane_d_R_m.to_numpy(float)) & (normal_lengths > 1e-12))
    error = measured-gt
    included = valid & (np.abs(error) <= threshold)
    excluded = valid & (np.abs(error) > threshold)
    if included.sum() < 4:
        raise ValueError("At least four retained groups are needed for fitting and validation")
    ids = raw.loc[included, "group_id"].to_numpy(int)
    x, y = measured[included], gt[included]
    normal_old = normals_old[included]/normal_lengths[included, None]
    angles_old = angle_error(normal_old, normals_gt[included])
    output.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(dict(group_id=raw.group_id, ir_valid=valid,
        included_in_fit=included, exclusion_reason=np.where(excluded, "original_abs_error_above_threshold",
            np.where(~valid, "invalid_infrared_plane", "")),
        gt_plane_distance_mm=gt, original_plane_distance_mm=measured,
        original_error_signed_mm=np.where(valid, error, np.nan),
        original_error_abs_mm=np.where(valid, np.abs(error), np.nan)))
    for name in ["original_normal_error_deg", "distance_corrected_mm", "distance_corrected_error_signed_mm",
                 "distance_corrected_error_abs_mm", "distance_loo_mm", "distance_loo_error_signed_mm",
                 "distance_loo_error_abs_mm", "sensors_refit_plane_distance_mm",
                 "sensors_refit_plane_error_signed_mm", "sensors_refit_plane_error_abs_mm",
                 "sensors_refit_normal_error_deg", "sensors_loo_plane_distance_mm",
                 "sensors_loo_plane_error_signed_mm", "sensors_loo_plane_error_abs_mm",
                 "sensors_loo_normal_error_deg"]:
        frame[name] = np.nan
    frame.loc[included, "original_normal_error_deg"] = angles_old
    distance_coefficients = fit_model(x, y)
    distance_fitted = np.polyval(distance_coefficients, x)
    distance_loo = leave_one_out(x, y)
    frame.loc[included, "distance_corrected_mm"] = distance_fitted
    frame.loc[included, "distance_loo_mm"] = distance_loo
    for prefix, values in [("distance_corrected", distance_fitted), ("distance_loo", distance_loo)]:
        frame.loc[included, prefix+"_error_signed_mm"] = values-y
        frame.loc[included, prefix+"_error_abs_mm"] = np.abs(values-y)

    distance_models = []
    for model in ("offset", "linear", "quadratic"):
        coefficients = fit_model(x, y, model)
        train = describe(np.polyval(coefficients, x)-y)
        loo = describe(leave_one_out(x, y, model)-y)
        distance_models.append(dict(model=model, coefficients_descending=coefficients,
                                    training=train, leave_one_out=loo))

    sensor_models, sensor_rows, points_fit, points_loo = [], [], [], []
    max_geometry_residual, max_ray_plane_residual = 0., 0.
    old_points = []
    fold_records = []
    for sensor in (1, 2, 3):
        prefix = f"s{sensor}_"
        origin, direction, residual = recover_geometry(raw, included, sensor)
        max_geometry_residual = max(max_geometry_residual, residual)
        raw_range = raw.loc[included, prefix+"raw_mm"].to_numpy(float)
        old_range = raw.loc[included, prefix+"calibrated_mm"].to_numpy(float)
        if not raw.loc[included, prefix+"status"].eq("VALID").all():
            raise ValueError("Retained plane group has an invalid sensor")
        denominator = normals_gt[included] @ direction
        if np.any(np.abs(denominator) < 1e-8):
            raise ValueError("Sensor ray is parallel to the reference plane")
        true_range = -(normals_gt[included] @ origin+coefficients_gt[included])/denominator
        if not np.isfinite(raw_range).all() or (true_range <= 0).any():
            raise ValueError("Invalid sensor reading or reference intersection")
        true_points = origin+true_range[:, None]*direction
        max_ray_plane_residual = max(max_ray_plane_residual, float(np.max(np.abs(
            np.einsum("ij,ij->i", normals_gt[included], true_points)+coefficients_gt[included]))))
        coefficients = fit_model(raw_range, true_range)
        fitted = np.polyval(coefficients, raw_range)
        loo = leave_one_out(raw_range, true_range)
        old_coefficients = fit_model(raw_range, old_range)
        if np.max(np.abs(np.polyval(old_coefficients, raw_range)-old_range)) > 1e-5:
            raise ValueError("Recorded old calibration is not affine")
        points_fit.append(origin+fitted[:, None]*direction)
        points_loo.append(origin+loo[:, None]*direction)
        old_points.append(raw.loc[included, [prefix+f"point_{axis}_m" for axis in "xyz"]].to_numpy()*1000)
        sensor_models.append(dict(sensor=sensor, model="linear", input="raw_mm", output="ray_range_mm",
            slope=float(coefficients[0]), intercept_mm=float(coefficients[1]),
            old_slope=float(old_coefficients[0]), old_intercept_mm=float(old_coefficients[1]),
            origin_R_mm=origin, direction_R_unit=direction, geometry_max_abs_residual_mm=residual,
            raw_training_range_mm=[float(raw_range.min()), float(raw_range.max())],
            gt_training_range_mm=[float(true_range.min()), float(true_range.max())],
            original=describe(old_range-true_range), training=describe(fitted-true_range),
            leave_one_out=describe(loo-true_range)))
        for k, gid in enumerate(ids):
            sensor_rows.append(dict(group_id=int(gid), sensor=sensor, raw_mm=raw_range[k],
                original_calibrated_mm=old_range[k], mocap_ray_range_mm=true_range[k],
                fitted_range_mm=fitted[k], loo_range_mm=loo[k],
                original_error_mm=old_range[k]-true_range[k], fitted_error_mm=fitted[k]-true_range[k],
                loo_error_mm=loo[k]-true_range[k]))
            training = np.arange(len(ids)) != k
            fold_coefficient = fit_model(raw_range[training], true_range[training])
            fold_records.append(dict(held_out_group_id=int(gid), sensor=sensor,
                training_group_ids=ids[training], coefficients_descending=fold_coefficient))

    normal_fit, plane_fit = plane_from_points(np.stack(points_fit, axis=1))
    normal_loo, plane_loo = plane_from_points(np.stack(points_loo, axis=1))
    normal_rebuilt, plane_rebuilt = plane_from_points(np.stack(old_points, axis=1))
    original_reconstruction_error = float(np.max(np.abs(plane_rebuilt-x)))
    original_angle_reconstruction_error = float(np.max(np.abs(
        angle_error(normal_rebuilt, normals_gt[included])-angles_old)))
    for prefix, values, normal in [("sensors_refit", plane_fit, normal_fit),
                                   ("sensors_loo", plane_loo, normal_loo)]:
        frame.loc[included, prefix+"_plane_distance_mm"] = values
        frame.loc[included, prefix+"_plane_error_signed_mm"] = values-y
        frame.loc[included, prefix+"_plane_error_abs_mm"] = np.abs(values-y)
        frame.loc[included, prefix+"_normal_error_deg"] = angle_error(normal, normals_gt[included])

    summary = dict(reference_plane_z_W_mm=PLANE_Z_MM, exclusion_threshold_mm=threshold,
        selection_rule="original plane distance absolute error <= threshold, frozen before fitting",
        total_groups=len(raw), valid_ir_groups=int(valid.sum()), retained_groups=ids,
        excluded_groups=raw.loc[excluded, "group_id"].to_numpy(int),
        invalid_groups=raw.loc[~valid, "group_id"].to_numpy(int),
        original_all_valid=describe(error[valid]), original_retained=describe(x-y),
        plane_distance_fit=describe(distance_fitted-y), plane_distance_leave_one_out=describe(distance_loo-y),
        original_retained_normal_error_deg=describe(angles_old),
        sensors_refit_plane=describe(plane_fit-y), sensors_leave_one_out_plane=describe(plane_loo-y),
        sensors_refit_normal_error_deg=describe(angle_error(normal_fit, normals_gt[included])),
        sensors_leave_one_out_normal_error_deg=describe(angle_error(normal_loo, normals_gt[included])),
        plane_distance_model_comparison=distance_models)
    config = dict(units="mm", reference_plane_z_W_mm=PLANE_Z_MM, retained_groups=ids,
        excluded_groups=summary["excluded_groups"],
        plane_distance_output=dict(model="linear", input="original_plane_distance_mm",
            output="corrected_plane_distance_mm", slope=float(distance_coefficients[0]),
            intercept_mm=float(distance_coefficients[1]),
            training_input_range_mm=[float(x.min()), float(x.max())],
            training_reference_range_mm=[float(y.min()), float(y.max())],
            preserves_original_normal=True,
            usage="Apply once to the original plane-distance output; do not stack with sensor refits"),
        sensors=sensor_models,
        limitations=["Leave-one-out validation is conditional on original-error filtering, not an independent test set",
            "Mounting geometry recovered from recorded points is held fixed",
            "Sensor 2 raw readings do not uniquely explain the mocap ranges in this batch",
            "Coefficients use millimetres; apply to original raw readings, replacing old range calibration"])
    validation = dict(max_recorded_ray_geometry_residual_mm=max_geometry_residual,
        max_true_ray_plane_residual_mm=max_ray_plane_residual,
        max_original_plane_reconstruction_error_mm=original_reconstruction_error,
        max_original_normal_angle_reconstruction_error_deg=original_angle_reconstruction_error,
        retained_groups_all_within_original_threshold=bool((np.abs(error[included]) <= threshold).all()),
        excluded_groups_all_above_original_threshold=bool((np.abs(error[excluded]) > threshold).all()),
        excluded_groups_have_no_corrected_predictions=bool(frame.loc[~included,
            ["distance_corrected_mm", "distance_loo_mm", "sensors_refit_plane_distance_mm",
             "sensors_loo_plane_distance_mm"]].isna().all().all()),
        loo_training_groups_per_fold=len(ids)-1,
        all_loo_folds_exclude_held_out_group=all(f["held_out_group_id"] not in f["training_group_ids"]
                                              for f in fold_records))
    previous = root / "results" / "per_group_errors.csv"
    if previous.exists():
        old = pd.read_csv(previous).set_index("group_id").loc[raw.group_id]
        validation["max_previous_mocap_distance_difference_mm"] = float(np.max(np.abs(
            old.ir_distance_gt_mm.to_numpy()-gt)))
        if validation["max_previous_mocap_distance_difference_mm"] > 1e-6:
            raise ValueError("Rebuilt reference differs from the previous analysis")
    validation["checks_passed"] = bool(max_geometry_residual < 1e-5 and max_ray_plane_residual < 1e-8
        and original_reconstruction_error < 1e-5 and original_angle_reconstruction_error < 1e-4
        and validation["excluded_groups_have_no_corrected_predictions"]
        and validation["all_loo_folds_exclude_held_out_group"])
    if not validation["checks_passed"]:
        raise AssertionError(validation)
    frame.to_csv(output / "per_group_calibration.csv", index=False, encoding="utf-8-sig")
    sensor_frame = pd.DataFrame(sensor_rows).sort_values(["group_id", "sensor"])
    sensor_frame.to_csv(output / "per_sensor_calibration.csv", index=False, encoding="utf-8-sig")
    metric_rows = []
    for key in ["original_all_valid", "original_retained", "plane_distance_fit",
                "plane_distance_leave_one_out", "sensors_refit_plane", "sensors_leave_one_out_plane",
                "original_retained_normal_error_deg", "sensors_refit_normal_error_deg",
                "sensors_leave_one_out_normal_error_deg"]:
        metric_rows.append(dict(metric=key, unit="deg" if "normal" in key else "mm", **summary[key]))
    pd.DataFrame(metric_rows).to_csv(output / "summary_metrics.csv", index=False, encoding="utf-8-sig")
    save_json(output / "calibration_coefficients.json", config)
    save_json(output / "summary.json", summary)
    save_json(output / "validation_checks.json", validation)
    save_json(output / "leave_one_out_folds.json", fold_records)
    paths = [root / name for name in ["positioning_records.csv", "markerPos_test_1.mat",
                                      "README.md", "marker_registration.py", "calibrate_infrared.py"]]
    save_json(output / "input_manifest.json", {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    save_json(output / "software_versions.json", dict(python=platform.python_version(),
        numpy=np.__version__, pandas=pd.__version__))
    make_plots(output, frame, sensor_frame, config, summary)
    write_report(output, frame, config, summary)
    return summary, config


def make_plots(output, frame, sensors, config, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False, "font.size": 10, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.alpha": .18,
        "figure.facecolor": "#f5f8fc", "axes.facecolor": "white", "savefig.facecolor": "#f5f8fc"})
    blue, orange, red, gray = "#087ea4", "#e88b26", "#bd4055", "#788697"
    used = frame.loc[frame.included_in_fit]
    excluded = frame.loc[frame.ir_valid & ~frame.included_in_fit]
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 10), layout="constrained")
    ax = axes[0, 0]
    ax.scatter(used.original_plane_distance_mm, used.gt_plane_distance_mm, c=blue, s=45, label="保留的 12 组")
    ax.scatter(excluded.original_plane_distance_mm, excluded.gt_plane_distance_mm,
               c=red, marker="x", s=90, linewidths=2, label="剔除组（拟合前误差 > 10 mm）")
    for row in frame.loc[frame.ir_valid].itertuples():
        ax.annotate(str(row.group_id), (row.original_plane_distance_mm, row.gt_plane_distance_mm),
                    xytext=(4, 5), textcoords="offset points", fontsize=8)
    xx = np.linspace(used.original_plane_distance_mm.min(), used.original_plane_distance_mm.max(), 200)
    c = config["plane_distance_output"]
    ax.plot(xx, c["slope"]*xx+c["intercept_mm"], c=blue, label="一次线性拟合")
    ax.plot([35, 135], [35, 135], "--", c=gray, label="理想 y = x")
    ax.set(xlabel="原始红外平面距离 / mm", ylabel="动捕基准垂直距离 / mm", title="平面距离输出标定")
    ax.legend(fontsize=8, loc="upper left")
    ax = axes[0, 1]
    ax.axhline(0, c=gray, lw=1)
    ax.plot(used.group_id, used.original_error_signed_mm, "o-", c=orange, label="拟合前")
    ax.plot(used.group_id, used.distance_corrected_error_signed_mm, "o-", c=blue, label="同批拟合后")
    ax.plot(used.group_id, used.distance_loo_error_signed_mm, "s--", c=red, label="留一验证")
    ax.set(xlabel="采集组号", ylabel="红外 − 动捕 / mm", title="同一批 12 组的有符号距离误差")
    ax.set_xticks(used.group_id)
    ax.legend(fontsize=9)
    ax = axes[1, 0]
    keys = ["original_retained", "plane_distance_fit", "plane_distance_leave_one_out"]
    labels = ["拟合前", "同批拟合后", "留一验证"]
    positions = np.arange(3)
    for offset, metric, color in [(-.19, "mae", blue), (.19, "rmse", orange)]:
        values = [summary[key][metric] for key in keys]
        bars = ax.bar(positions+offset, values, width=.36, color=color, label=metric.upper())
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_xticks(positions, labels)
    ax.set(ylabel="误差 / mm", title="平面距离输出校正：MAE 与 RMSE", ylim=(0, 5.3))
    ax.legend(fontsize=9)
    ax = axes[1, 1]
    for sensor, color in zip(config["sensors"], [blue, orange, red]):
        s = sensors.loc[sensors.sensor == sensor["sensor"]]
        ax.scatter(s.raw_mm, s.mocap_ray_range_mm, color=color, alpha=.85, s=28,
                   label=f'S{sensor["sensor"]}（留一 MAE {sensor["leave_one_out"]["mae"]:.2f} mm）')
        xx = np.linspace(s.raw_mm.min(), s.raw_mm.max(), 100)
        ax.plot(xx, sensor["slope"]*xx+sensor["intercept_mm"], color=color)
    ax.set(xlabel="传感器原始读数 / mm", ylabel="动捕推算的沿射线距离 / mm", title="三个器件分别拟合（与输出校正分开使用）")
    ax.legend(fontsize=8)
    fig.suptitle("动捕基准下的红外重新标定", fontsize=18, fontweight="bold")
    fig.savefig(output / "calibration_dashboard.png", dpi=175)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(15.8, 8.5), layout="constrained")
    for j, sensor in enumerate(config["sensors"]):
        s = sensors.loc[sensors.sensor == sensor["sensor"]]
        ax = axes[0, j]
        ax.scatter(s.raw_mm, s.mocap_ray_range_mm, c=blue, s=35, label="动捕射线交点")
        xx = np.linspace(s.raw_mm.min(), s.raw_mm.max(), 150)
        ax.plot(xx, sensor["slope"]*xx+sensor["intercept_mm"], c=blue, label="重新标定")
        ax.plot(xx, sensor["old_slope"]*xx+sensor["old_intercept_mm"], "--", c=orange, label="原标定")
        for row in s.itertuples():
            ax.annotate(str(row.group_id), (row.raw_mm, row.mocap_ray_range_mm),
                        xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.set(title=f'S{sensor["sensor"]}：r = {sensor["slope"]:.6f} x {sensor["intercept_mm"]:+.6f}',
               xlabel="原始读数 x / mm", ylabel="沿射线基准距离 r / mm")
        ax.legend(fontsize=8)
        ax = axes[1, j]
        for column, color, label in [("original_error_mm", orange, "原标定"),
                                    ("fitted_error_mm", blue, "同批拟合"),
                                    ("loo_error_mm", red, "留一验证")]:
            ax.plot(s.group_id, s[column], "o-", c=color, label=label, markersize=3)
        ax.axhline(0, c=gray, lw=1)
        ax.set(title=f'S{sensor["sensor"]}：留一 MAE {sensor["leave_one_out"]["mae"]:.3f} mm',
               xlabel="采集组号", ylabel="沿射线距离误差 / mm")
        ax.set_xticks(s.group_id)
        ax.legend(fontsize=8)
    fig.suptitle("逐传感器标定诊断（全部使用相同的 12 组）", fontsize=17, fontweight="bold")
    fig.savefig(output / "per_sensor_diagnostics.png", dpi=175)
    plt.close(fig)


def write_report(output, frame, config, summary):
    c = config["plane_distance_output"]
    used = frame.loc[frame.included_in_fit]
    excluded = frame.loc[frame.ir_valid & ~frame.included_in_fit]
    retained = "、".join(map(str, summary["retained_groups"]))
    lines = ["# 以动捕为基准的红外重新标定", "",
        "## 结果与使用建议", "",
        f"按**拟合前**红外平面距离绝对误差大于 {summary['exclusion_threshold_mm']:g} mm 剔除整组。"
        f"原有 {summary['valid_ir_groups']} 组有效平面，剔除第 "
        f"{'、'.join(map(str, summary['excluded_groups']))} 组，保留 {len(used)} 组：{retained}。"
        "其余无效平面组不补值、不纳入标定。第 8 组虽然视觉失败，红外与动捕有效，仍保留。", "",
        "最终平面距离的一次拟合公式为：", "", "```text",
        f"d_corrected_mm = {c['slope']:.9f} * d_original_mm {c['intercept_mm']:+.9f}", "```", "",
        "公式输入为现有三传感器融合后输出的 `plane_distance_mm`。校正一次即可。"
        "下表始终比较同一批保留组，以区分异常组剔除与拟合各自带来的变化。", "",
        "| 平面距离结果 | 组数 | MAE / mm | RMSE / mm | 有符号偏差 / mm | 最大绝对误差 / mm |",
        "|---|---:|---:|---:|---:|---:|"]
    for key, label in [("original_all_valid", "原始全部有效组（仅作背景）"),
                       ("original_retained", "剔除后、拟合前"),
                       ("plane_distance_fit", "平面距离线性校正（同批拟合）"),
                       ("plane_distance_leave_one_out", "平面距离线性校正（留一验证）")]:
        s = summary[key]
        lines.append(f"| {label} | {s['n']} | {s['mae']:.3f} | {s['rmse']:.3f} | {s['bias']:+.3f} | {s['max_abs']:.3f} |")
    improvement = 100*(1-summary["plane_distance_fit"]["mae"]/summary["original_retained"]["mae"])
    normal_mean = summary["original_retained_normal_error_deg"]["mae"]
    lines += ["", f"保留组同批拟合的距离 MAE 降低 {improvement:.1f}%。每次留一验证使用另外 {len(used)-1} 组"
        "重新拟合，再预测被留出的组。筛选名单在拟合前固定，未按拟合后残差再次删组。"
        "留一结果针对本次筛选后的样本，仍需独立采集验证。", "",
        f"距离输出校正保持原法向，保留组法向平均误差仍为 **{normal_mean:.3f}°**。"
        "若需要同步校正法向，应检查下文的逐传感器结果。", "",
        f"本批公式输入覆盖 {c['training_input_range_mm'][0]:.3f}～{c['training_input_range_mm'][1]:.3f} mm，"
        f"对应动捕垂直距离 {c['training_reference_range_mm'][0]:.3f}～{c['training_reference_range_mm'][1]:.3f} mm。"
        "超出此范围的表现尚未验证。", "",
        "![标定总览](calibration_dashboard.png)", "", "## 剔除记录", "",
        "| 组号 | 原始红外距离 / mm | 动捕距离 / mm | 绝对误差 / mm |",
        "|---|---:|---:|---:|"]
    for row in excluded.itertuples():
        lines.append(f"| {row.group_id} | {row.original_plane_distance_mm:.3f} | {row.gt_plane_distance_mm:.3f} | {row.original_error_abs_mm:.3f} |")
    lines += ["", "阈值针对融合平面的垂直距离误差，不针对单个传感器沿射线误差。等于阈值的组保留。", "",
        "## 三个传感器分别重新拟合", "",
        "分别拟合 `r_new_mm = a * raw_mm + b`，输入为 `s1_raw_mm`、`s2_raw_mm`、`s3_raw_mm`。"
        "新公式替换原来的器件距离标定，随后沿现有安装射线重建三个命中点，并重新求平面。"
        "此流程与上面的最终距离输出校正是两种替代方案，不能叠加使用。", "",
        "| 传感器 | a | b / mm | 原始读数范围 / mm | 原标定 MAE / mm | 同批拟合 MAE / mm | 留一 MAE / mm |",
        "|---|---:|---:|---|---:|---:|---:|"]
    for s in config["sensors"]:
        lo, hi = s["raw_training_range_mm"]
        lines.append(f"| S{s['sensor']} | {s['slope']:.9f} | {s['intercept_mm']:+.9f} | {lo:g}～{hi:g} | "
                     f"{s['original']['mae']:.3f} | {s['training']['mae']:.3f} | {s['leave_one_out']['mae']:.3f} |")
    lines += ["", "以上为沿传感器射线距离的误差，不能直接与垂直平面距离误差混用。", "",
        "| 三点重建后的结果 | 平面距离 MAE / mm | 平面距离 RMSE / mm | 法向平均误差 / ° |",
        "|---|---:|---:|---:|"]
    for label, distance, angle in [("原标定", "original_retained", "original_retained_normal_error_deg"),
            ("逐传感器线性重标定（同批）", "sensors_refit_plane", "sensors_refit_normal_error_deg"),
            ("逐传感器线性重标定（留一）", "sensors_leave_one_out_plane", "sensors_leave_one_out_normal_error_deg")]:
        lines.append(f"| {label} | {summary[distance]['mae']:.3f} | {summary[distance]['rmse']:.3f} | {summary[angle]['mae']:.3f} |")
    lines += ["", "S1、S3 的沿射线距离得到较好校正。S2 在第 1、10 组的原始读数均为 122 mm，"
        "动捕推算的射线距离却分别为约 126.251、147.501 mm。仅靠读数的一次函数无法同时拟合这两点。"
        "第 5、10 组的 S2 原标定沿射线误差也接近 40 mm，但整组三点融合距离误差未超过 10 mm，"
        "因此按本次规则仍保留。", "",
        "逐传感器重标定后的平面距离留一误差高于原标定，当前数据不支持把这套三器件参数作为已验证的整体改进。"
        "若目标是改善现有距离输出，优先使用上面的平面距离线性校正；若目标包括法向精度，"
        "需复核 S2 的命中表面、安装射线、组间静止状态及动捕基准，再采集数据验证。", "",
        "![逐传感器诊断](per_sensor_diagnostics.png)", "", "## 计算方法与模型选择", "",
        "1. 直接从 `markerPos_test_1.mat` 重建动捕机器人位姿，沿用已有的标记点枚举与正向命中约束，"
        "不使用视觉位姿参与拟合。机器人三点组枚举全部 4 选 3 子集及排列。", 
        "2. 基准面为 `z_W = -75 mm`。`T_W_R` 的旋转和平移记为 `R,t`，"
        "则机器人系平面 `n_R · p_R + c_R = 0` 中，`n_R = Rᵀ[0,0,1]`，`c_R = t_z + 75`，"
        "真实垂直距离为 `abs(c_R)`。", 
        "3. 从记录的 `s*_calibrated_mm` 与 `s*_point_*_m` 恢复固定射线 `p_R = o_R + r*u_R`。"
        "只用保留组恢复既有安装关系，再用其他可用点验证。没有重新估计安装外参。"
        "动捕射线距离为 `r_gt = -(n_R·o_R + c_R)/(n_R·u_R)`。", 
        "4. 三器件分别以原始读数拟合真实射线距离。重建三个命中点后，叉积求单位法向，"
        "再计算原点到平面的垂直距离与 `acos(abs(n_fit·n_gt))` 法向误差。", 
        "5. 所有拟合均为普通最小二乘。平面距离输出比较偏置、一次和二次模型；"
        "采用一次模型，因其留一表现优于偏置和二次模型。逐传感器统一采用一次模型，避免小样本高阶拟合。", "",
        "| 平面输出模型 | 同批 MAE / mm | 同批 RMSE / mm | 留一 MAE / mm | 留一 RMSE / mm |",
        "|---|---:|---:|---:|---:|"]
    labels = {"offset": "仅偏置", "linear": "一次线性（采用）", "quadratic": "二次"}
    for model in summary["plane_distance_model_comparison"]:
        a, b = model["training"], model["leave_one_out"]
        lines.append(f"| {labels[model['model']]} | {a['mae']:.3f} | {a['rmse']:.3f} | {b['mae']:.3f} | {b['rmse']:.3f} |")
    lines += ["", "原有机翼名义标记几何与动捕边长存在约 1.5～10.4 mm 的不一致，"
        "机翼配准平均点残差约 4.945 mm。这里的拟合吸收了当前基准中的系统误差，"
        "结果依赖该安装、基准面及按组序静态配对条件。", "", "## 逐组平面距离结果", "",
        "| 组号 | 动捕 / mm | 原始红外 / mm | 校正后 / mm | 校正有符号误差 / mm | 留一有符号误差 / mm |",
        "|---|---:|---:|---:|---:|---:|"]
    for row in used.itertuples():
        lines.append(f"| {row.group_id} | {row.gt_plane_distance_mm:.3f} | {row.original_plane_distance_mm:.3f} | "
            f"{row.distance_corrected_mm:.3f} | {row.distance_corrected_error_signed_mm:+.3f} | {row.distance_loo_error_signed_mm:+.3f} |")
    lines += ["", "## 输出与复现", "",
        "- `calibration_coefficients.json`：距离输出校正与三个器件的完整参数、射线几何和适用范围。",
        "- `per_group_calibration.csv`：全部 30 组及筛选原因，保留组的两种校正与留一结果。",
        "- `per_sensor_calibration.csv`：保留组的逐器件原始读数、动捕射线距离及误差。",
        "- `summary_metrics.csv`、`summary.json`：数值统计。",
        "- `validation_checks.json`：原始平面复原、射线交点、阈值筛选、留一排除检查。",
        "- `leave_one_out_folds.json`：逐器件每折的训练组号和系数。",
        "- `input_manifest.json`：原始数据与复现脚本的 SHA-256。", "",
        "```powershell", "python calibrate_infrared.py", "```", "",
        "默认仅生成 `results/infrared_calibration/`，原始采集文件与之前的分析结果保持不变。",
        "CSV 以 UTF-8 BOM 保存。"]
    (output / "红外重新标定报告.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold-mm", type=float, default=10.)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = args.output.resolve() if args.output else root / "results" / "infrared_calibration"
    summary, config = calibrate(root, output, args.threshold_mm)
    print(json.dumps(native(dict(retained_groups=summary["retained_groups"],
        excluded_groups=summary["excluded_groups"], plane_distance_output=config["plane_distance_output"],
        before=summary["original_retained"], after=summary["plane_distance_fit"],
        leave_one_out=summary["plane_distance_leave_one_out"], output=str(output))), ensure_ascii=False, indent=2))
