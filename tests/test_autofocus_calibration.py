import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from calibration.autofocus_calibration import (capture, fit, focus_list, interval_is_valid, position_key,
                                    stabilized_frame, StopCapture)
from calibration.camera_calibration import write_json, write_image, object_points
from localization.focus_map import FocusMap, FocusStability
from localization.positioning_core import load_config, VisionEstimator, record_snapshot, save_record, CSV_FIELDS
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]


def sample_map():
    entries = [dict(focus_position=q,camera_matrix=[[f,0,320],[0,f,240],[0,0,1]],
                    dist_coeffs=[.1, -.2, .001, .002, .05]) for q,f in [(200,550),(300,530),(400,510)]]
    return dict(model='opencv_pinhole_focus_map',image_size=[640,480],flip_camera_180=True,
                entries=entries,validated_intervals=[[200,300]])


class AutofocusCalibrationTests(unittest.TestCase):
    def test_lookup_interpolates_and_rejects_unvalidated_or_outside_focus(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'map.json'
            write_json(path,sample_map())
            table=FocusMap(path,[640,480],True)
            self.assertEqual(table.lookup(200)[0][0,0],550)
            self.assertEqual(table.lookup(250)[0][0,0],540)
            np.testing.assert_allclose(table.lookup(250)[1],[.1,-.2,.001,.002,.05])
            for q in [199,401,350,None,float('nan')]:
                with self.assertRaises(ValueError): table.lookup(q)
            with self.assertRaises(ValueError): FocusMap(path,[320,240],True)
            with self.assertRaises(ValueError): FocusMap(path,[640,480],False)
            data=sample_map();data['validation']=[{'focus':200,'kind':'anchor','passed':False}]
            write_json(path,data)
            with self.assertRaises(ValueError): FocusMap(path,[640,480],True).lookup(200)
            data=sample_map();data['entries'][1]['camera_matrix'][0][0]=-1
            write_json(path,data)
            with self.assertRaises(ValueError): FocusMap(path,[640,480],True)

    def test_stable_frame_gate_resets_on_focus_change(self):
        gate=FocusStability()
        self.assertIsNone(gate.update(200,201))
        self.assertIsNone(gate.update(201,201))
        self.assertIsNone(gate.update(201,201))
        self.assertEqual(gate.update(201,201),201)
        self.assertIsNone(gate.update(210,210))
        with self.assertRaises(ValueError): gate.update(-1,-1)

    def test_focus_parser_and_holdout_interval_gate(self):
        self.assertEqual(focus_list('300,200,400'),[200,300,400])
        for value in ['200','200,200','200,nan']:
            with self.assertRaises(ValueError):focus_list(value)
        self.assertFalse(interval_is_valid([.2,.3],1))
        self.assertFalse(interval_is_valid([.2,.3,1.1],1))
        self.assertTrue(interval_is_valid([.2,.3,.4],1))

    def test_capture_writes_separate_focus_metadata_and_skips_bad_images(self):
        cap=MagicMock();cap.isOpened.return_value=True;cap.getBackendName.return_value='DSHOW'
        cap.read.return_value=(True,np.zeros((480,640,3),np.uint8));cap.get.return_value=200.
        state=dict(focus_readback=200.,autofocus_mode='MANUAL',autofocus_readback=2.)
        def stable(camera,q,args,settings,status,**kwargs):
            return None if q==300 else (np.zeros((480,640,3),np.uint8),{**state,'focus_readback':q},100.)
        with tempfile.TemporaryDirectory() as temp:
            config=Path(temp)/'config.json'
            write_json(config,dict(camera_index=1,camera_resolution=[640,480],camera_fps=15,flip_camera_180=True))
            args=SimpleNamespace(config=config,camera=1,backend='dshow',output=Path(temp),focus_values='200,300,400',
                cols=9,rows=6,square_mm=20,validation_every=5)
            with patch('calibration.autofocus_calibration.cv2.VideoCapture',return_value=cap), \
                    patch('calibration.autofocus_calibration.cv2.namedWindow'), \
                    patch('calibration.autofocus_calibration.cv2.destroyAllWindows'), \
                    patch('calibration.autofocus_calibration.show',side_effect=[ord('s'),StopCapture()]), \
                    patch('calibration.autofocus_calibration.stabilized_frame',side_effect=stable), contextlib.redirect_stdout(io.StringIO()):
                capture(args)
            session=next(Path(temp).glob('session_*'))
            data=json.loads((session/'session.json').read_text(encoding='utf-8'))
            self.assertEqual([r['saved'] for r in data['groups'][0]['results']],[True,False,True])
            self.assertFalse((session/'train/focus_300').exists())
            metadata=json.loads((session/'train/focus_400/capture.json').read_text(encoding='utf-8'))
            self.assertEqual(metadata['focus_samples']['pose_0001.png']['focus_readback'],400)
            cap.release.assert_called_once()

    def test_estimator_uses_map_and_missing_focus_never_reuses_pose(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'map.json';write_json(path,sample_map())
            config=load_config(ROOT/'config/positioning_config.json');config['focus_map_file']=path
            config["intrinsics_calibrated"] = True
            estimator=VisionEstimator(config)
            estimator.detector=SimpleNamespace(detect=lambda gray: [])
            frame=np.zeros((480,640,3),np.uint8)
            result=estimator.process(frame.copy(),focus_position=250)
            self.assertEqual(result.focus_position,250)
            self.assertEqual(result.camera_matrix[0][0],540)
            row = record_snapshot(result, {}, None, config)
            self.assertEqual((row["camera_fx_px"], row["camera_fy_px"], row["focus_position"]), (540, 540, 250))
            extrapolated = estimator.process(frame.copy(), focus_position=500)
            row = record_snapshot(extrapolated, {}, None, config)
            self.assertEqual((row["camera_fx_px"], row["camera_fy_px"], row["focus_position"]), (490, 490, 500))
            rejected = estimator.process(frame.copy(), focus_position=3000)
            row = record_snapshot(rejected, {}, None, config)
            self.assertEqual((row["camera_fx_px"], row["camera_fy_px"], row["focus_position"]), ("", "", 3000))
            result=estimator.process(frame.copy())
            self.assertIsNone(result.camera_pose)
            self.assertTrue(result.status.startswith('FOCUS:'))

    def test_project_default_applies_focus_dependent_intrinsics(self):
        config = load_config(ROOT / "config/positioning_config.json")
        config["intrinsics_calibrated"] = True
        estimator = VisionEstimator(config)
        estimator.detector = SimpleNamespace(detect=lambda gray: [])
        frame = np.zeros((480, 640, 3), np.uint8)
        table = estimator.focus_map
        # Include both outer ranges, an unvalidated interval, and its unvalidated anchor.
        for focus, left, right in ((208, 0, 1), (288, 0, 1), (336, 1, 2), (368, 1, 2), (400, 1, 2)):
            with self.subTest(focus=focus):
                result = estimator.process(frame, focus_position=focus)
                row = record_snapshot(result, {}, None, config)
                fraction = (focus - table.focus[left]) / (table.focus[right] - table.focus[left])
                expected_matrix = (1 - fraction) * table.matrices[left] + fraction * table.matrices[right]
                expected_distortion = (1 - fraction) * table.distortions[left] + fraction * table.distortions[right]
                np.testing.assert_allclose(result.camera_matrix, expected_matrix)
                np.testing.assert_allclose(result.dist_coeffs, expected_distortion)
                self.assertEqual(row["focus_position"], focus)
                self.assertAlmostEqual(row["camera_fx_px"], expected_matrix[0, 0])
                self.assertAlmostEqual(row["camera_fy_px"], expected_matrix[1, 1])

    def test_focus_sidecar_and_csv_roll_back_together(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp); row=dict.fromkeys(CSV_FIELDS,'')
            row['_camera_context']={'focus_position':250,'camera_matrix':np.eye(3).tolist()}
            path,image=save_record(folder/'records.csv',folder/'images','1',row,np.zeros((8,8,3),np.uint8),datetime.now())
            sidecar=image.with_suffix('.json')
            self.assertEqual(json.loads(sidecar.read_text(encoding='utf-8'))['focus_position'],250)
            with patch('localization.positioning_core.append_csv',side_effect=PermissionError('locked')):
                with self.assertRaises(PermissionError):
                    save_record(folder/'records.csv',folder/'images','2',row,np.zeros((8,8,3),np.uint8),datetime.now())
            self.assertEqual(len(list((folder/'images').glob('*.png'))),1)
            self.assertEqual(len(list((folder/'images').glob('*.json'))),1)

    def test_real_opencv_batch_fit_and_independent_midpoint_validation(self):
        # Known projected correspondences exercise actual per-anchor calibration,
        # held-out PnP and map export without requiring a camera or printed board.
        spec={'inner_corners':[9,6],'square_size_mm':20.}
        obj=object_points(spec);rng=np.random.default_rng(71);points={}
        poses=[(rng.uniform(-.5,.5,3),rng.uniform([-.1,-.07,.35],[.01,.01,.65])) for _ in range(15)]
        with tempfile.TemporaryDirectory() as temp:
            session=Path(temp)
            write_json(session/'session.json',{**spec,'focus_values':[200.,300.,400.],
                'image_size':[640,480],'flip_camera_180':True,'camera_index':1})
            for qi,q in enumerate([200.,250.,300.,350.,400.],1):
                focal=550-(q-200)*.2
                K=np.array([[focal,0,320],[0,focal,240],[0,0,1.]])
                for split,indices in [('train',range(12)),('validation',range(12,15))]:
                    if split=='train' and q not in [200.,300.,400.]:continue
                    folder=session/split/position_key(q)
                    metadata={**spec,'image_size':[640,480],'flip_camera_180':True,'images_already_oriented':True,
                        'camera_index':1,'focus_position':q,'focus_lock_verified':True,'backend':'DSHOW','focus_samples':{}}
                    for i in indices:
                        image=np.full((480,640,3),128,np.uint8);image[0,0]=[qi,i,0]
                        name=f'pose_{i:04d}.png';write_image(folder/name,image)
                        rv,tv=poses[i]
                        points[(qi,i)]=cv2.projectPoints(obj,rv,tv,K,np.zeros(5))[0].astype(np.float32)
                        metadata['focus_samples'][name]={'focus_readback':q,'autofocus_mode':'MANUAL'}
                    write_json(folder/'capture.json',metadata)
            def detector(image,*_):
                return points[tuple(int(v) for v in image[0,0,:2])]
            with patch('calibration.camera_calibration.detect_corners',side_effect=detector), \
                    patch('calibration.autofocus_calibration.detect_corners',side_effect=detector), contextlib.redirect_stdout(io.StringIO()):
                fit(SimpleNamespace(session=session,overwrite=False,max_validation_rms=1.))
            table=FocusMap(session/'calibrated/focus_map.json',[640,480],True)
            self.assertEqual(table.data['validated_intervals'],[[200.,300.],[300.,400.]])
            self.assertAlmostEqual(table.lookup(250)[0][0,0],540,places=2)
            self.assertTrue(all(len(v['errors_px'])==3 for v in table.data['validation']))
