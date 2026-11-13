import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from calibration.camera_calibration import write_json
from calibration.focus_probe import analyze, capture, corner_difference, autofocus_mode


class FocusProbeTests(unittest.TestCase):
    def test_autofocus_modes_are_backend_specific(self):
        self.assertEqual(autofocus_mode(2, 'DSHOW'), 'MANUAL')
        self.assertEqual(autofocus_mode(1, 'DSHOW'), 'AUTO')
        for value in [-1, 0, 3, 2.5, None, float('nan')]:
            self.assertEqual(autofocus_mode(value, 'DSHOW'), 'UNKNOWN')
        self.assertEqual(autofocus_mode(0, 'MSMF'), 'MANUAL')
        self.assertEqual(autofocus_mode(1, 'MSMF'), 'AUTO')
        self.assertEqual(autofocus_mode(2, 'MSMF'), 'UNKNOWN')

    def points(self):
        return np.mgrid[0:9, 0:6].T.reshape(-1, 2).astype(float) * 20 + [100, 100]

    def test_repeatability_does_not_align_away_scale_or_translation(self):
        a = self.points()
        zero = corner_difference(a, a[::-1])
        self.assertAlmostEqual(zero['corner_rms_px'], 0)
        translated = corner_difference(a, a + [3, 4])
        self.assertAlmostEqual(translated['corner_rms_px'], 5)
        self.assertAlmostEqual(translated['center_shift_px'], 5)
        scaled = corner_difference(a, (a-a.mean(0))*1.01+a.mean(0))
        self.assertAlmostEqual(scaled['relative_spread_change_percent'], 1)

    def test_report_only_compares_manual_same_readback_detected_boards(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            samples = []
            for i, (q, af, points) in enumerate([(20,0,self.points()), (25,0,self.points()),
                    (20,1,self.points()), (20,0,None), (20,0,self.points()+[.3,.4])], 1):
                samples.append(dict(sample=i, focus_readback=q, autofocus_readback=af,
                    corners=None if points is None else points.tolist(), sharpness=100))
            write_json(folder/'samples.json', samples)
            write_json(folder/'events.json', [])
            with contextlib.redirect_stdout(io.StringIO()):
                summary = analyze(folder)
            self.assertEqual(summary['same_focus_comparisons'], 1)
            self.assertAlmostEqual(summary['comparisons'][0]['corner_rms_px'], .5)

    def test_old_dshow_sessions_include_manual_flag_two(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            write_json(folder/'session.json', {'backend': 'DSHOW'})
            write_json(folder/'events.json', [])
            write_json(folder/'samples.json', [dict(sample=i,focus_readback=144,autofocus_readback=af,
                corners=self.points().tolist(),sharpness=100) for i,af in enumerate([2,1,2,0,-1],1)])
            with contextlib.redirect_stdout(io.StringIO()):
                summary = analyze(folder)
            self.assertEqual(summary['same_focus_comparisons'], 1)
            self.assertEqual(summary['comparisons'][0]['reference_sample'], 1)
            self.assertEqual(summary['comparisons'][0]['sample'], 3)

    def test_interactive_capture_saves_requests_and_delayed_readback(self):
        class FakeCamera:
            def __init__(self):
                self.focus, self.af, self.released = 20., 1., False
            def isOpened(self): return True
            def getBackendName(self): return 'MOCK'
            def get(self, prop):
                return self.focus if prop == cv2.CAP_PROP_FOCUS else self.af
            def set(self, prop, value):
                if prop == cv2.CAP_PROP_FOCUS: self.focus = value
                if prop == cv2.CAP_PROP_AUTOFOCUS: self.af = value
                return True
            def read(self): return True, np.full((480,640,3), 128, np.uint8)
            def release(self): self.released = True
        # Simulate both boolean and DSHOW flag readbacks through the whole UI.
        for actual_backend in ['MOCK', 'DSHOW']:
          camera = FakeCamera()
          original_set = camera.set
          def set_property(prop, value):
              result = original_set(prop, value)
              if prop == cv2.CAP_PROP_AUTOFOCUS and actual_backend == 'DSHOW':
                  camera.af = 1 if value else 2
              return result
          camera.set = set_property
          camera.getBackendName = lambda: actual_backend
          with tempfile.TemporaryDirectory() as temp:
            self.run_capture_case(temp, camera)

    def run_capture_case(self, temp, camera):
            folder = Path(temp)
            cfg = folder/'config.json'
            write_json(cfg, dict(camera_index=1,camera_resolution=[640,480],camera_fps=15,flip_camera_180=True))
            args = SimpleNamespace(config=cfg,camera=None,backend='dshow',output=folder,step=5,settle=0,cols=9,rows=6)
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch('calibration.focus_probe.cv2.VideoCapture', return_value=camera))
                for name in ['namedWindow','resizeWindow','imshow','destroyAllWindows']:
                    stack.enter_context(patch('calibration.focus_probe.cv2.'+name))
                stack.enter_context(patch('calibration.focus_probe.cv2.getWindowProperty', return_value=1))
                stack.enter_context(patch('calibration.focus_probe.cv2.waitKey', side_effect=[ord(k) for k in 'mbs]srsq']))
                stack.enter_context(patch('calibration.focus_probe.detect_corners', return_value=self.points().astype(np.float32).reshape(-1,1,2)))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                capture(args)
            self.assertTrue(camera.released)
            session = next(folder.glob('session_*'))
            samples = json.loads((session/'samples.json').read_text(encoding='utf-8'))
            self.assertEqual([s['focus_readback'] for s in samples], [20,25,20])
            self.assertEqual(len(list(session.glob('sample_*.png'))), 3)
            summary = json.loads((session/'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['accepted_focus_requests'], 2)
            self.assertEqual(summary['same_focus_comparisons'], 1)


if __name__ == '__main__':
    unittest.main()
