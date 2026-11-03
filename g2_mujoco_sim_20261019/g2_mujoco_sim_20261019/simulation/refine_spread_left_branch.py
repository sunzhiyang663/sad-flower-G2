"""Choose an equivalent shoulder solution with lower left-arm joint travel."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from simulation.refine_grasp_elbow_down import load
OUT=ROOT/'simulation/outputs/grasp_spread_bilateral'

def refine(c,report):
    start=report['right_grasp']['joints'];pair=report['paired_grasp'];base=np.array(pair['base']);B=np.array(pair['payload'])
    c.set_base(base);c.set_state(pair['joints'],B)
    target=c.c.get_transform_frame_to_world('arm_l_end_link','body').as_matrix().copy()
    names=c.names['left'];lo,hi=np.array([c.model.joint(n).range for n in names]).T;lo=lo+.02;hi=hi-.02
    ids=np.array([int(c.model.joint(n).qposadr[0]) for n in names]);ref=c.data.qpos.copy()
    old=np.array([pair['joints'][n] for n in names]);waiting=np.array([start[n] for n in names]);best=old.copy();cost=float(np.abs(old-waiting).sum())
    def error(x):
        q=ref.copy();q[ids]=x;c.c.update(q);T=c.c.get_transform_frame_to_world('arm_l_end_link','body').as_matrix()
        return np.r_[1000*(T[:3,3]-target[:3,3]),250*Rotation.from_matrix(target[:3,:3].T@T[:3,:3]).as_rotvec(),.001*(x-waiting)]
    for a in [-np.pi,np.pi]:
        for b in [-np.pi,np.pi]:
            seed=old.copy();seed[:3]=[old[0]+a,-old[1],old[2]+b]
            if np.any(seed<lo) or np.any(seed>hi):continue
            result=least_squares(error,seed,bounds=(lo,hi),max_nfev=80,gtol=1e-10,xtol=1e-10,ftol=1e-10)
            e=error(result.x);travel=float(np.abs(result.x-waiting).sum())
            if np.linalg.norm(e[:3])>.01 or np.linalg.norm(e[3:6])>.001 or travel>=cost:continue
            candidate={**pair['joints'],**dict(zip(names,result.x.tolist()))}
            try:c.set_state(candidate,B);collision=c.collisions(['right','left'])
            except ValueError:continue
            best=result.x.copy();cost=travel;pair['collision']=collision
    pair['joints'].update(dict(zip(names,best.tolist())))
    report['left_shoulder_branch_optimization']=dict(before_joint_travel_rad=float(np.abs(old-waiting).sum()),after_joint_travel_rad=cost,method='Equivalent Z-Y-Z shoulder seeds, exact URDF IK refinement and whole-body collision check')
    c.set_state(pair['joints'],B);actual=c.c.get_transform_frame_to_world('arm_l_end_link','body').as_matrix()
    report['left_grasp_error_m']=float(np.linalg.norm(actual[:3,3]-(B@np.array(report['left_grasp_target_in_payload']))[:3,3]))
    return report

if __name__=='__main__':
    _,c,_,_=load(ROOT/'simulation/grasp_wrist_parallel_front.yaml')
    path=OUT/'bilateral_final_ik.json';r=json.loads(path.read_text())
    if not r['passed']:raise ValueError('Endpoint checks required')
    r=refine(c,r);path.write_text(json.dumps(r,indent=2));print(r['left_shoulder_branch_optimization'])
