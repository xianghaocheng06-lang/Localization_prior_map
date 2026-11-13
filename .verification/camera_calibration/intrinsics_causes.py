import json
import sys
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from localization.positioning_core import VisionEstimator, load_config, transform_points, inverse_transform
from experiments.replay_experiment import DEFAULT_EXPERIMENT, corner_only, pose_errors
from calibration.camera_calibration import read_image, detect_corners, object_points

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

root=DEFAULT_EXPERIMENT/'results/pnp_replay_20261106'
out=root/'intrinsics_cause_analysis'
out.mkdir(exist_ok=True)
cfg=load_config(PROJECT/'config/positioning_config.json')
est=VisionEstimator(cfg)
old=read(PROJECT/'calibration/charuco_calib.json')
new=read(cfg['calibration_file'])
KO,KN=np.array(old['camera_matrix']),np.array(new['camera_matrix'])
DO,DN=np.array(old['dist_coeffs']),np.array(new['dist_coeffs'])
diag=read(root/'replay_diagnostics.json')
truth={p['group_id']:np.array(p['T_W_C_gt_mm']) for p in read(DEFAULT_EXPERIMENT/'results/poses.json')}
result=[]
shifts=[]
for group in diag:
    gid=group['group_id']
    tags=[SimpleNamespace(tag_id=t['tag_id'],corners=np.array(t['corners']),center=np.array(t['center'])) for t in group['observations']]
    est.K,est.D=KO,DO
    cv2.setRNGSeed(20261101+gid)
    baseline=corner_only(est,tags)
    if baseline.camera_pose is None:
        continue
    world=np.vstack([transform_points(est.layout[str(t.tag_id)],est.local) for t in tags])
    image=np.vstack([t.corners for t in tags])
    inds=baseline.inlier_indices
    world,image=world[inds],image[inds]
    seed=inverse_transform(baseline.camera_pose)
    seed_rv=cv2.Rodrigues(seed[:3,:3])[0]
    seed_tv=seed[:3,3].copy()
    for bits in range(8):
        K=KO.copy()
        D=DN if bits&4 else DO
        if bits&1:
            K[0,0],K[1,1]=KN[0,0],KN[1,1]
        if bits&2:
            K[0,2],K[1,2]=KN[0,2],KN[1,2]
        rv,tv=cv2.solvePnPRefineLM(world,image,K,D,seed_rv.copy(),seed_tv.copy())
        T=np.eye(4)
        T[:3,:3]=cv2.Rodrigues(rv)[0]
        T[:3,3]=tv.ravel()
        pose=inverse_transform(T)
        projected=cv2.projectPoints(world,rv,tv,K,D)[0].reshape(-1,2)
        rmse=np.sqrt(np.mean(np.sum((projected-image)**2,1)))
        result.append(dict(group_id=gid,swap_focal=bool(bits&1),swap_principal=bool(bits&2),swap_distortion=bool(bits&4),bits=bits,pixel_rmse=rmse,**pose_errors(pose,truth[gid])))
    a=group['methods']['old_K_corners']['pose_m']
    b=group['methods']['new_K_corners']['pose_m']
    if a is not None and b is not None:
        a,b=np.array(a),np.array(b)
        R=truth[gid][:3,:3]
        shift=R.T@((b[:3,3]-a[:3,3])*1000)
        origin=est.layout[str(group['observations'][0]['tag_id'])][:3,3]
        z=(a[:3,:3].T@(origin-a[:3,3]))[2]*1000
        shifts.append([gid,*shift,z])
f=pd.DataFrame(result)
f.to_csv(out/'fixed_inlier_parameter_swaps.csv',index=False,encoding='utf-8-sig')
s=f.groupby('bits').agg(n=('group_id','size'),position_mm=('position_error_mm','mean'),rotation_deg=('rotation_error_deg','mean'),pixel_rmse=('pixel_rmse','mean'),bias_x=('error_x_mm','mean'),bias_y=('error_y_mm','mean'),bias_z=('error_z_mm','mean'))
s.to_csv(out/'parameter_swap_summary.csv',encoding='utf-8-sig')
print(s.to_string())
v=np.array(shifts)
save={'mean_old_to_new_camera_frame_shift_mm':v[:,1:4].mean(0).tolist(),
      'mean_tag_distance_mm':float(v[:,-1].mean()),
      'approx_focal_scale_depth_shift_mm':float(v[:,-1].mean()*(np.mean([KN[0,0]/KO[0,0],KN[1,1]/KO[1,1]])-1))}
(out/'depth_shift.json').write_text(json.dumps(save,indent=2),encoding='utf-8')
print(save)
calibration_rows=[]
obj=object_points(new)
source_directory = Path(new['source_directory'])
if not source_directory.is_dir():
    source_directory = PROJECT/'calibration/images'/source_directory.name
for view in new['views']:
    if view['status']!='used':
        continue
    corners=detect_corners(read_image(source_directory/view['file']),new['inner_corners'])
    for label,K,D in [('old',KO,DO),('new',KN,DN)]:
        ok,rv,tv=cv2.solvePnP(obj,corners,K,D,flags=cv2.SOLVEPNP_ITERATIVE)
        projected=cv2.projectPoints(obj,rv,tv,K,D)[0]
        rmse=float(np.sqrt(np.mean(np.sum((projected.reshape(-1,2)-corners.reshape(-1,2))**2,axis=1))))
        calibration_rows.append({'file':view['file'],'intrinsics':label,'rmse_px':rmse})
c=pd.DataFrame(calibration_rows)
c.to_csv(out/'same_calibration_images_fit.csv',index=False,encoding='utf-8-sig')
print('Calibration image fits',c.groupby('intrinsics').rmse_px.agg(['mean','median','max']).to_string())
