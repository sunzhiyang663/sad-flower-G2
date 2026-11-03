"""Authorized offline cleanup of scanned voxels conflicting with manipulation.

Preserves explicit rack/table geometry and the original occupied 2D grid. This
simplification is recorded and must not be deployed as online obstacle deletion.
"""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mujoco
import numpy as np
import yaml
from simulation.full_task_config import load_task
from simulation.build_map9_scene import path
from simulation.map9_route_preview import Clearance


def main():
    from simulation.render_map9_full_task import make_sequence
    cfg=yaml.safe_load((ROOT/'simulation/map9.yaml').read_text());task=load_task();out=path(task['output'])
    meta=json.loads((path(cfg['snapshot'])/'map.json').read_text())
    model=mujoco.MjModel.from_binary_path(str(out/'scene.mjb'));saved=dict(np.load(out/'state.npz'))
    checker=Clearance(model,saved,cfg['base_height_m'],.1);distances={}
    for goal in task['goal_guide_ids']:
        sequence,_,_=make_sequence(goal,cfg,task,out,meta)
        for p,q,T,label,_ in sequence:
            if label.startswith(('1 ','2 ','6 ')):continue
            checker.data.qpos[:]=q;checker.saved['T_base_payload']=T
            checker.check(p,voxel_distances=distances)
    removed=[]
    for geom,distance in distances.items():
        if distance >= task['motion']['workzone_voxel_cleanup_distance_m']:continue
        removed.append(dict(name=model.geom(geom).name,centre=model.geom_pos[geom].tolist(),minimum_distance_m=distance))
        model.geom_group[geom]=5;model.geom_contype[geom]=0;model.geom_conaffinity[geom]=0
    mujoco.mj_saveModel(model,str(out/'scene.mjb'))
    (out/'workzone_cleanup.json').write_text(json.dumps(dict(removed=removed,
        policy='User-authorized offline task-first cleanup along checked manipulation sweeps; occupied 2D grid and explicit furniture retained',
        deployment_allowed=False),indent=2))
    print('Workzone cloud voxels removed:',len(removed),flush=True)


if __name__=='__main__':main()
