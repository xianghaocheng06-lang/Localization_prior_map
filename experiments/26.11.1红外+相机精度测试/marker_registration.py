"""Rigid registration in millimetres. T_A_B maps B coordinates into A."""
from itertools import combinations, permutations

import numpy as np
from scipy.io import loadmat

WING_MARKERS_MM = np.array([[553, 241, 128], [365, 7.5, 128], [638, 3, 128]], float)
ROBOT_MARKERS_MM = np.array([
    [-34.34, -10.29, -1.2], [-17, -27.2, 1.5],
    [17, -27.2, 1.5], [34.34, -10.29, -4.2],
], float)
T_R_C_MM = np.eye(4)
T_R_C_MM[:3, 3] = [0, -29.5, 4]
PLANE_Z_MM = -75.0


def rigid_fit(source, target):
    """Kabsch, proper rotation only, unweighted point RMS in mm."""
    a, b = np.asarray(source, float), np.asarray(target, float)
    if a.shape != b.shape or a.shape[0] < 3 or a.shape[1] != 3:
        raise ValueError('Matched point arrays must have the same (N>=3,3) shape')
    ca, cb = a.mean(0), b.mean(0)
    if np.linalg.matrix_rank(a - ca, tol=1e-8) < 2:
        raise ValueError('Collinear markers cannot determine a pose')
    u, _, vt = np.linalg.svd((a - ca).T @ (b - cb))
    rotation = vt.T @ np.diag([1, 1, np.linalg.det(vt.T @ u.T)]) @ u.T
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = cb - rotation @ ca
    residuals = a @ rotation.T + transform[:3, 3] - b
    rms = float(np.sqrt(np.mean(np.sum(residuals**2, axis=1))))
    return transform, rms, residuals


def enumerate_fits(model, observed):
    """Explicitly enumerate all C(N,k) subsets AND k! correspondences."""
    found = []
    for subset in combinations(range(len(model)), len(observed)):
        for correspondence in permutations(subset):
            transform, rms, residuals = rigid_fit(model[list(correspondence)], observed)
            found.append(dict(mapping=correspondence, T=transform, rms_mm=rms,
                              residuals_mm=residuals))
    return sorted(found, key=lambda c: c['rms_mm'])


def inverse_rigid(transform):
    result = np.eye(4)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ transform[:3, 3]
    return result


def transform_points(transform, points):
    return np.asarray(points) @ transform[:3, :3].T + transform[:3, 3]


def load_marker_groups(path):
    """Read MATLAB cells; keep acquisition order, remove empty point cells."""
    raw = loadmat(path, squeeze_me=True)['markerPos_array']
    result = []
    for row in range(raw.shape[0]):
        occupied = [col for col in range(raw.shape[1]) if np.size(raw[row, col])]
        if not occupied:
            continue
        if occupied != [0]:
            raise ValueError(f'Unexpected MATLAB cell layout at row {row + 1}: {occupied}')
        points = [np.asarray(p, float).reshape(3) * 1000
                  for p in raw[row, 0].ravel() if np.size(p)]
        points = np.asarray(points)
        if not np.isfinite(points).all():
            raise ValueError(f'Non-finite marker at MATLAB row {row + 1}')
        unique_robot, kept, duplicates = [], [], []
        for point_index, point in enumerate(points[3:], start=4):
            if any(np.linalg.norm(point - previous) < 1e-6 for previous in unique_robot):
                duplicates.append(point_index)
            else:
                unique_robot.append(point)
                kept.append(point_index)
        if len(unique_robot) not in (3, 4):
            raise ValueError(f'Need 3 or 4 unique robot markers at row {row + 1}')
        result.append(dict(mat_row_1based=row + 1, raw=points, wing=points[:3],
                           robot=np.asarray(unique_robot), kept_marker_indices=kept,
                           duplicate_marker_indices=duplicates))
    return result


def faces_reference_plane(T_W_R):
    """Require z=-75 to be in the +z_R half-space, independent of vision.

    The recorded IR hit points are on +z_R; README says their plane is in
    front of the robot. This resolves the near-planar marker flip ambiguity.
    Equivalent: a ray from the robot origin along +z_R hits z=-75 at s>0.
    """
    denominator = T_W_R[2, 2]
    return bool(abs(denominator) > 1e-9 and
                (PLANE_Z_MM - T_W_R[2, 3]) / denominator > 0)
