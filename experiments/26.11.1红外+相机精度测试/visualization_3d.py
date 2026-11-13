"""STL-based PyVista renders and an offline, interactive Plotly viewer."""
import json
from pathlib import Path

import numpy as np
import pyvista as pv

from marker_registration import transform_points, PLANE_Z_MM, WING_MARKERS_MM


def plane_patch(center, normal, half_size=36):
    normal=np.asarray(normal,float)
    axis=np.array([1.,0.,0.]) if abs(normal[0])<.9 else np.array([0.,1.,0.])
    u=np.cross(normal,axis); u/=np.linalg.norm(u)
    v=np.cross(normal,u)
    return np.array([center+half_size*(a*u+b*v) for a,b in [(-1,-1),(1,-1),(1,1),(-1,1)]])


def load_display_meshes(root):
    # Decimation is display-only; fits use the exact nominal marker coordinates.
    # Quadric decimation preserves large thin STL panels; decimate_pro can
    # collapse these panels and leave only isolated fastener details.
    wing=pv.read(root/'wing_general_simplified.STL').triangulate().decimate(.985)
    robot=pv.read(root/'robot_general_simplified.stl').triangulate().decimate(.98)
    return wing,robot


def add_scene(plotter,wing,robot,poses,selected=17,close=False):
    pose=poses[selected-1]
    gt=np.array([p['T_W_C_gt_mm'][:3,3] for p in poses])
    estimates=np.array([p['T_W_C_vision_mm'][:3,3] for p in poses if p['T_W_C_vision_mm'] is not None])
    plotter.set_background('#f3f6fa')
    shown_wing=wing.clip_box((180,455,-20,175,-130,160),invert=False) if close else wing
    plotter.add_mesh(shown_wing,color='#9caec2',opacity=.09 if close else .20,smooth_shading=True,label='Wing STL')
    if close:
        plane=pv.Plane(center=(320,70,-75),direction=(0,0,1),i_size=220,j_size=180)
    else:
        plane=pv.Plane(center=(745,120,-75),direction=(0,0,1),i_size=1500,j_size=260)
    plotter.add_mesh(plane,color='#7ca9c2',opacity=.18,label='Reference z = -75 mm')
    plotter.add_points(gt,color='#007f9d',point_size=10,render_points_as_spheres=True,label='Mocap camera')
    plotter.add_points(estimates,color='#e68a23',point_size=10,render_points_as_spheres=True,label='Vision camera')
    for p in poses:
        if p['T_W_C_vision_mm'] is not None:
            plotter.add_lines(np.array([p['T_W_C_gt_mm'][:3,3],p['T_W_C_vision_mm'][:3,3]]),color='#a2a9b5',width=1)
    if close:
        plotter.add_mesh(robot.transform(pose['T_W_R_gt_mm'],inplace=False),color='#007f9d',opacity=.65,
            smooth_shading=True,label='Mocap robot')
        if pose['T_W_R_vision_mm'] is not None:
            plotter.add_mesh(robot.transform(pose['T_W_R_vision_mm'],inplace=False),color='#e68a23',opacity=.4,
                smooth_shading=True,label='Vision robot')
        plotter.add_points(pose['robot_observed_W_mm'],color='#192c43',point_size=12,render_points_as_spheres=True,label='Observed markers')
        if 'ir_normal_W' in pose:
            corners=plane_patch(pose['ir_projection_W_mm'],pose['ir_normal_W'])
            plotter.add_mesh(pv.PolyData(corners,np.array([4,0,1,2,3])),color='#cf4556',opacity=.48,label='IR estimated plane')
            plotter.add_arrows(np.array([pose['ir_projection_W_mm']]),np.array([pose['ir_normal_W']]),mag=35,color='#cf4556')
            plotter.add_arrows(np.array([pose['ir_true_projection_W_mm']]),np.array([[0,0,1]]),mag=35,color='#007f9d')
            plotter.add_points(pose['ir_points_W_mm'],color='#cf4556',point_size=11,render_points_as_spheres=True)
            plotter.add_lines(np.array([pose['T_W_R_gt_mm'][:3,3],pose['ir_projection_W_mm']]),color='#cf4556',width=2)
        plotter.add_text(f"Group {selected:02d} | Ground truth / Vision / IR",position='upper_left',font_size=16,color='#20364e')
        m=pose['metrics']
        def fmt(x): return 'invalid' if x is None or not np.isfinite(x) else f'{x:.2f}'
        plotter.add_text(f"Camera: {fmt(m['camera_position_error_mm'])} mm, {fmt(m['camera_rotation_error_deg'])} deg\n"
            f"IR: {fmt(m['ir_distance_error_abs_mm'])} mm, {fmt(m['ir_normal_error_deg'])} deg",
            position='lower_left',font_size=13,color='#20364e')
        plotter.camera_position=[(575,-320,235),(320,68,-5),(0,0,1)]
        plotter.enable_parallel_projection(); plotter.camera.parallel_scale=125
    else:
        plotter.add_points(WING_MARKERS_MM,color='#4c5b70',point_size=13,render_points_as_spheres=True,label='Nominal wing markers')
        plotter.add_text('Wing model and 30 camera reference positions',position='upper_left',font_size=18,color='#20364e')
        plotter.camera_position=[(1780,-1600,1100),(735,115,0),(0,0,1)]
        plotter.enable_parallel_projection(); plotter.camera.parallel_scale=460
    plotter.add_legend(bcolor='white',face='rectangle',size=(.22,.24),loc='upper right')
    plotter.add_axes(xlabel='X',ylabel='Y',zlabel='Z')


def mesh_dict(mesh):
    return dict(points=np.round(mesh.points,5).tolist(),
                triangles=mesh.faces.reshape(-1,4)[:,1:].tolist())


def build_html(output,wing,robot,poses):
    from plotly.offline import get_plotlyjs
    from analyze_localization import native
    data=dict(wing=mesh_dict(wing),robot=mesh_dict(robot),poses=poses)
    payload=json.dumps(native(data),ensure_ascii=False,separators=(',',':'),allow_nan=False)
    html=r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>视觉与红外定位 · 三维误差</title><style>
*{box-sizing:border-box}body{margin:0;background:#f3f6fa;color:#22344a;font-family:"Microsoft YaHei",system-ui,sans-serif}
header{padding:22px 30px 16px;border-bottom:1px solid #dce3eb;background:white}h1{font-size:24px;margin:0 0 8px}p{line-height:1.65;margin:7px 0;color:#617084;font-size:13px}
main{display:grid;grid-template-columns:310px 1fr;gap:16px;padding:18px;max-width:1800px;margin:auto}.panel{background:white;border:1px solid #e0e6ed;border-radius:12px;padding:20px}#plot{height:calc(100vh - 158px);min-height:570px;background:white;border-radius:12px;overflow:hidden;border:1px solid #e0e6ed}
.row{display:flex;justify-content:space-between;align-items:center;margin:8px 0}h2{font-size:16px;margin:0 0 12px}input[type=range]{width:100%;accent-color:#087e9d}input[type=checkbox]{accent-color:#087e9d}label{font-size:13px}.cards{display:grid;grid-template-columns:1fr 1fr;gap:9px;margin:18px 0}.card{background:#f3f7fa;border-radius:8px;padding:12px 8px}.card small{display:block;font-size:11px;color:#617084}.card strong{display:block;font-size:21px;margin-top:5px}.card em{font-style:normal;font-size:11px;color:#617084}button{border:1px solid #d1dde8;border-radius:6px;background:white;padding:8px 12px;cursor:pointer;color:#22344a}button:hover{background:#eef6fa}.blue{color:#007f9d}.orange{color:#dc821e}.red{color:#cf4556}hr{border:0;border-top:1px solid #e7ebf0;margin:18px 0}#details{font-size:12px;line-height:1.8}a{color:#087e9d}#status{min-height:20px;font-size:12px;color:#a04d39}.legend{font-size:12px;line-height:2}@media(max-width:900px){main{grid-template-columns:1fr}#plot{height:600px}.cards{grid-template-columns:repeat(4,1fr)}}
</style><script>__PLOTLY_JS__</script></head><body>
<header><h1>视觉与红外定位精度 · 三维对照</h1><p>STL 模型与全部有效测量在机翼坐标系 W 中展示，坐标单位 mm。拖动旋转，滚轮缩放，左右键切换组号。</p></header>
<main><aside class="panel"><div class="row"><h2>采集组 <span id="groupLabel">17 / 30</span></h2><div><button id="prev">←</button> <button id="next">→</button></div></div>
<input id="group" aria-label="采集组号" type="range" min="1" max="30" value="17" step="1"><div id="status"></div>
<div class="cards"><div class="card"><small>相机位置误差</small><strong class="orange" id="pos">—</strong><em>mm</em></div><div class="card"><small>相机姿态误差</small><strong class="orange" id="rot">—</strong><em>° · SO(3)</em></div><div class="card"><small>红外距离绝对误差</small><strong class="red" id="dist">—</strong><em>mm</em></div><div class="card"><small>红外法向误差</small><strong class="red" id="angle">—</strong><em>°</em></div></div>
<div class="row"><button id="local">局部视角</button><button id="full">整翼视角</button></div><hr>
<div class="row"><label><input id="wing" type="checkbox" checked> 机翼 STL</label></div>
<div class="row"><label><input id="gtRobot" type="checkbox" checked> 动捕机器人与标记点</label></div>
<div class="row"><label><input id="visionRobot" type="checkbox" checked> 视觉机器人</label></div>
<div class="row"><label><input id="ir" type="checkbox" checked> 红外平面与法向</label></div>
<div class="row"><label><input id="all" type="checkbox" checked> 全部相机位置与误差连线</label></div>
<hr><div class="legend"><span class="blue">● 动捕基准 / 真实法向</span><br><span class="orange">● 视觉估计 / 位置误差</span><br><span class="red">● 红外估计平面 / 法向</span><br>淡蓝色水平面：z = −75 mm</div><hr>
<div id="details"></div><p>红外点、平面和法向由动捕姿态转换到 W 系，避免混入视觉姿态误差。模型简化仅用于显示。</p>
<p><a href="分析报告.md">分析报告</a> · <a href="per_group_errors.csv">逐组数据</a> · <a href="error_dashboard.png">误差图</a></p></aside><section id="plot"></section></main>
<script>
const DATA=__DATA__;
const $=id=>document.getElementById(id);
const xyz=pts=>({x:pts.map(p=>p[0]),y:pts.map(p=>p[1]),z:pts.map(p=>p[2])});
const tx=(T,p)=>[0,1,2].map(i=>T[i][0]*p[0]+T[i][1]*p[1]+T[i][2]*p[2]+T[i][3]);
const norm=v=>Math.sqrt(v.reduce((s,a)=>s+a*a,0));
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
function patch(c,n){let u=cross(n,Math.abs(n[0])<.9?[1,0,0]:[0,1,0]);const l=norm(u);u=u.map(v=>v/l);const v=cross(n,u);return [[-1,-1],[1,-1],[1,1],[-1,1]].map(([a,b])=>c.map((x,i)=>x+36*(a*u[i]+b*v[i])));}
function mesh(name,m,color,opacity){return {type:'mesh3d',name,...xyz(m.points),i:m.triangles.map(t=>t[0]),j:m.triangles.map(t=>t[1]),k:m.triangles.map(t=>t[2]),color,opacity,flatshading:false,lighting:{ambient:.7,diffuse:.8},hoverinfo:'skip',showlegend:false};}
function scatter(name,pts,color,size=5){return {type:'scatter3d',name,...xyz(pts),mode:'markers',marker:{color,size},showlegend:false,hovertemplate:name+'<br>X=%{x:.2f}<br>Y=%{y:.2f}<br>Z=%{z:.2f}<extra></extra>'};}
function line(name,pts,color,width=4){return {type:'scatter3d',name,...xyz(pts),mode:'lines',line:{color,width},hoverinfo:'skip',showlegend:false};}
const quad={points:[[200,-10,-75],[450,-10,-75],[450,160,-75],[200,160,-75]],triangles:[[0,1,2],[0,2,3]]};
const allGt=DATA.poses.map(p=>p.T_W_C_gt_mm.map(r=>r[3]).slice(0,3));
const allVision=DATA.poses.filter(p=>p.T_W_C_vision_mm).map(p=>p.T_W_C_vision_mm.map(r=>r[3]).slice(0,3));
const connectors=[];DATA.poses.forEach((p,i)=>{if(p.T_W_C_vision_mm)connectors.push(allGt[i],p.T_W_C_vision_mm.map(r=>r[3]).slice(0,3),[null,null,null]);});
const traces=[mesh('机翼 STL',DATA.wing,'#aebaca',.10),mesh('参考平面 z=-75',quad,'#77a9c6',.20),
scatter('动捕相机',allGt,'#007f9d',3),scatter('视觉相机',allVision,'#e68a23',3),line('相机位置误差',connectors,'#b3bbc7',2),
mesh('动捕机器人',DATA.robot,'#007f9d',.62),mesh('视觉机器人',DATA.robot,'#e68a23',.35),
scatter('动捕标记点',[],'#1d3552',5),scatter('当前相机：动捕',[],'#007f9d',7),scatter('当前相机：视觉',[],'#e68a23',7),
line('当前相机位置误差',[],'#e68a23',7),mesh('红外估计平面',{points:[],triangles:[[0,1,2],[0,2,3]]},'#cf4556',.50),
scatter('三个红外测量点',[],'#cf4556',5),line('红外法向',[],'#cf4556',8),line('基准法向',[],'#007f9d',8),line('红外垂直距离',[],'#cf4556',3)];
let current=17, fullView=false;
const scene={xaxis:{title:'X / mm',range:[205,440],backgroundcolor:'#f5f7fb'},yaxis:{title:'Y / mm',range:[-10,165],backgroundcolor:'#f5f7fb'},zaxis:{title:'Z / mm',range:[-105,155],backgroundcolor:'#f5f7fb'},aspectmode:'manual',aspectratio:{x:.904,y:.673,z:1},camera:{eye:{x:1.25,y:-2.0,z:.9},up:{x:0,y:0,z:1},projection:{type:'orthographic'}}};
function changePoints(idx,pts,visible=true){Plotly.restyle('plot',{x:[pts.map(p=>p[0])],y:[pts.map(p=>p[1])],z:[pts.map(p=>p[2])],visible},[idx]);}
function update(group){current=Math.max(1,Math.min(30,Number(group)));$('group').value=current;$('groupLabel').textContent=current+' / 30';
const p=DATA.poses[current-1],m=p.metrics,hasVision=!!p.T_W_R_vision_mm,hasIR=!!p.ir_normal_W;
const fmt=v=>v===null?'无效':Number(v).toFixed(2);
$('pos').textContent=fmt(m.camera_position_error_mm);$('rot').textContent=fmt(m.camera_rotation_error_deg);$('dist').textContent=fmt(m.ir_distance_error_abs_mm);$('angle').textContent=fmt(m.ir_normal_error_deg);
$('status').textContent=(!hasVision?'视觉 PnP 失败。 ':'')+(!hasIR?'红外平面无效。 ':'');
$('details').innerHTML=`机器人唯一标记点：<b>${m.unique_robot_marker_count}</b> 个<br>标记拟合 RMS：<b>${m.robot_fit_rms_mm.toFixed(3)}</b> mm<br>红外真实距离：<b>${m.ir_distance_gt_mm.toFixed(2)}</b> mm<br>红外测量距离：<b>${fmt(m.ir_distance_measured_mm)}</b> mm<br>距离有符号误差：<b>${fmt(m.ir_distance_error_signed_mm)}</b> mm`;
changePoints(5,DATA.robot.points.map(q=>tx(p.T_W_R_gt_mm,q)),$('gtRobot').checked);
changePoints(6,hasVision?DATA.robot.points.map(q=>tx(p.T_W_R_vision_mm,q)):[],$('visionRobot').checked&&hasVision);
changePoints(7,p.robot_observed_W_mm,$('gtRobot').checked);
const cam=tx(p.T_W_C_gt_mm,[0,0,0]);changePoints(8,[cam]);
const cv=hasVision?tx(p.T_W_C_vision_mm,[0,0,0]):null;changePoints(9,cv?[cv]:[],hasVision);changePoints(10,cv?[cam,cv]:[],hasVision);
const irVisible=$('ir').checked&&hasIR;
changePoints(11,hasIR?patch(p.ir_projection_W_mm,p.ir_normal_W):[],irVisible);
changePoints(12,hasIR?p.ir_points_W_mm:[],irVisible);
changePoints(13,hasIR?[p.ir_projection_W_mm,p.ir_projection_W_mm.map((v,i)=>v+35*p.ir_normal_W[i])]:[],irVisible);
changePoints(14,hasIR?[p.ir_true_projection_W_mm,p.ir_true_projection_W_mm.map((v,i)=>v+(i===2?35:0))]:[],irVisible);
changePoints(15,hasIR?[tx(p.T_W_R_gt_mm,[0,0,0]),p.ir_projection_W_mm]:[],irVisible);
Plotly.restyle('plot',{visible:$('wing').checked},[0]);Plotly.restyle('plot',{visible:$('all').checked},[2,3,4]);}
function view(full){fullView=full;const ranges=full?[[-40,1510],[-30,280],[-140,150]]:[[205,440],[-10,165],[-105,155]];Plotly.relayout('plot',{'scene.xaxis.range':ranges[0],'scene.yaxis.range':ranges[1],'scene.zaxis.range':ranges[2],'scene.aspectratio':full?{x:3,y:.6,z:.561}:{x:.904,y:.673,z:1},'scene.camera':{eye:full?{x:1.15,y:-2,z:1}:{x:1.25,y:-2,z:.9},up:{x:0,y:0,z:1},projection:{type:'orthographic'}}});}
Plotly.newPlot('plot',traces,{scene,margin:{l:0,r:0,t:0,b:0},paper_bgcolor:'white',uirevision:'keep',font:{family:'Microsoft YaHei, sans-serif',color:'#405168'}},{responsive:true,displaylogo:false,toImageButtonOptions:{format:'png',filename:'localization_3d',width:1600,height:1200,scale:1}}).then(()=>update(17));
$('group').addEventListener('input',e=>update(e.target.value));$('prev').onclick=()=>update(current-1);$('next').onclick=()=>update(current+1);
['wing','gtRobot','visionRobot','ir','all'].forEach(id=>$(id).onchange=()=>update(current));$('local').onclick=()=>view(false);$('full').onclick=()=>view(true);
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT')return;if(e.key==='ArrowLeft'){e.preventDefault();update(current-1);}if(e.key==='ArrowRight'){e.preventDefault();update(current+1);}});
</script></body></html>'''
    html=html.replace('__PLOTLY_JS__',get_plotlyjs()).replace('__DATA__',payload)
    (output/'interactive_3d.html').write_text(html,encoding='utf-8')


def build_visualizations(root,output,poses):
    wing,robot=load_display_meshes(root)
    checks={}
    for name,filename,display in [('wing','wing_general_simplified.STL',wing),('robot','robot_general_simplified.stl',robot)]:
        original=pv.read(root/filename)
        relative_area_change=abs(display.area-original.area)/original.area
        assert relative_area_change<.05, f'{name} STL simplification lost geometry'
        checks[name]=dict(original_triangles=original.n_cells,display_triangles=display.n_cells,
            original_surface_area_mm2=original.area,display_surface_area_mm2=display.area,
            relative_surface_area_change=relative_area_change)
    (output/'display_geometry_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    for filename,close in [('scene_overview.png',False),('scene_group17.png',True)]:
        plotter=pv.Plotter(off_screen=True,window_size=(1600,1000))
        add_scene(plotter,wing,robot,poses,selected=17,close=close)
        plotter.show(screenshot=str(output/filename),auto_close=True)
    build_html(output,wing,robot,poses)
    print(f'Display meshes: wing {wing.n_cells} triangles; robot {robot.n_cells} triangles')
