"""Optional native PyVista viewer: python view_3d.py --group 17."""
import argparse
import json
from pathlib import Path

import numpy as np
import pyvista as pv

from visualization_3d import add_scene, load_display_meshes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group',type=int,default=17,choices=range(1,31))
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    poses=json.loads((root/'results'/'poses.json').read_text(encoding='utf-8'))
    for pose in poses:
        for key in list(pose):
            if isinstance(pose[key],list):
                pose[key]=np.asarray(pose[key],float)
    wing,robot=load_display_meshes(root)
    plotter=pv.Plotter(window_size=(1500,950),title='Localization accuracy / PyVista')
    active={'group':None}
    def select(value):
        group=int(round(value))
        if group==active['group']:
            return
        plotter.clear()
        add_scene(plotter,wing,robot,poses,selected=group,close=True)
        active['group']=group
        plotter.render()
    plotter.add_slider_widget(select,rng=(1,30),value=args.group,title='Group',fmt='%.0f',
        pointa=(.26,.93),pointb=(.72,.93),interaction_event='end')
    plotter.show()


if __name__=='__main__':
    main()
