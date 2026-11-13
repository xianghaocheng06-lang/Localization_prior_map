"""Lookup and linear estimation for focus-dependent OpenCV calibration."""
import json
from pathlib import Path

import numpy as np


class FocusStability:
    """Require matching before/after-read values and three consecutive stable frames."""
    def __init__(self, frames=3):
        self.frames, self.last, self.count = frames, None, 0

    def update(self, before, after):
        if not np.isfinite([before, after]).all() or min(before, after) < 0:
            self.last, self.count = None, 0
            raise ValueError('驱动未提供有效焦点回读，无法动态选择内参')
        if abs(before-after) > 1e-6:
            self.last, self.count = None, 0
            return None
        self.count = self.count+1 if self.last is not None and abs(after-self.last)<=1e-6 else 1
        self.last = after
        return after if self.count >= self.frames else None


class FocusMap:
    def __init__(self, path, image_size, flip_180):
        self.path = Path(path)
        self.data = json.loads(self.path.read_text(encoding='utf-8-sig'))
        if (self.data.get('model') != 'opencv_pinhole_focus_map'
                or self.data.get('image_size') != list(image_size)
                or type(self.data.get('flip_camera_180')) is not bool
                or self.data['flip_camera_180'] != flip_180):
            raise ValueError('焦点参数表的模型、分辨率或旋转方向与定位配置不符')
        entries = self.data['entries']
        if len(entries) < 2:
            raise ValueError('焦点参数表至少需要两档')
        self.focus = np.array([e['focus_position'] for e in entries], float)
        self.matrices = np.array([e['camera_matrix'] for e in entries], float)
        self.distortions = np.array([e['dist_coeffs'] for e in entries], float)
        if (not np.isfinite(self.focus).all() or np.any(np.diff(self.focus) <= 0)
                or self.matrices.shape != (len(entries), 3, 3)
                or not np.isfinite(self.matrices).all()
                or np.any(self.matrices[:, [0, 1], [0, 1]] <= 0)
                or not np.allclose(self.matrices[:, 2, :], [0, 0, 1])
                or not np.allclose(self.matrices[:, [0, 1], [1, 0]], 0)
                or self.distortions.shape != (len(entries), 5)
                or not np.isfinite(self.distortions).all()):
            raise ValueError('焦点参数表存在无效参数或未按焦点严格递增排列')

    def lookup(self, focus, require_validation=True):
        """Use boundary segments for extrapolation when validation is optional."""
        if focus is None or not np.isfinite(focus):
            raise ValueError('缺少有效焦点回读')
        exact = np.flatnonzero(np.isclose(self.focus, focus, atol=1e-6, rtol=0))
        if len(exact):
            i = exact[0]
            if require_validation and 'validation' in self.data:
                if not any(v.get('kind') == 'anchor' and v.get('focus') == self.focus[i] and v.get('passed') is True
                           for v in self.data['validation']):
                    raise ValueError(f'焦点 {focus:g} 尚未通过独立验证')
            return self.matrices[i].copy(), self.distortions[i].copy()
        j = int(np.searchsorted(self.focus, focus))
        if require_validation:
            if j == 0 or j == len(self.focus):
                raise ValueError(f'焦点 {focus:g} 超出已标定范围 [{self.focus[0]:g}, {self.focus[-1]:g}]')
            intervals = self.data.get('validated_intervals', [])
            if not any(np.allclose(pair, [self.focus[j-1], self.focus[j]], atol=1e-6, rtol=0) for pair in intervals):
                raise ValueError(f'焦点区间 [{self.focus[j-1]:g}, {self.focus[j]:g}] 尚未通过中间焦点验证')
        j = min(max(j, 1), len(self.focus) - 1)
        alpha = (focus - self.focus[j - 1]) / (self.focus[j] - self.focus[j - 1])
        matrix = self.matrices[j - 1] + alpha * (self.matrices[j] - self.matrices[j - 1])
        distortion = self.distortions[j - 1] + alpha * (self.distortions[j] - self.distortions[j - 1])
        if not np.isfinite(matrix).all() or not np.isfinite(distortion).all() or min(matrix[0, 0], matrix[1, 1]) <= 0:
            raise ValueError(f'焦点 {focus:g} 计算出的相机内参无效')
        return matrix, distortion
