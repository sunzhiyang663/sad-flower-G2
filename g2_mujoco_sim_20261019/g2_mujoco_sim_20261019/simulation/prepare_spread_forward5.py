"""Shift only the accepted left wrist 5 cm in shell +X; offline FK/IK audit."""
import json, sys, shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import mujoco
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from simulation.refine_grasp_elbow_down import load

OUT=ROOT/'simulation/outputs/grasp_spread_forward5'
def main():
    OUT.mkdir(exist_ok=True)
    settings_path=ROOT/'simulation/grasp_spread_forward5.yaml'
    adjustments=yaml.safe_load(settings_path.read_text()) if settings_path.exists() else {}
    forward=float(adjustments.get('left_forward_shift_m',.05));vertical=float(adjustments.get('left_vertical_shift_m',-.003))
    source=ROOT/'simulation/reference_inputs/accepted_grasp'
    r=json.loads((source/'bilateral_final_ik.json').read_text())
    cfg,c,_,_=load(ROOT/'simulation/grasp_wrist_parallel_front.yaml')
    pair=r['paired_grasp'];base=np.array(pair['base']);B=np.array(pair['payload']);c.set_base(base);c.set_state(pair['joints'],B)
    before=c.c.get_transform_frame_to_world('arm_l_end_link','body').as_matrix().copy();target=before.copy();target[:3,3]+=forward*B[:3,0]+vertical*B[:3,2]
    names=c.names['left'];ids=np.array([int(c.model.joint(n).qposadr[0]) for n in names]);ref=c.data.qpos.copy();seed=ref[ids].copy()
    lo,hi=np.array([c.model.joint(n).range for n in names]).T;lo+=.02;hi-=.02
    moving={g for g in c.robot if c.model.body(int(c.model.geom_bodyid[g])).name.startswith(('arm_l_','hand_l_'))}
    pairs=np.array([(a,b) for a,b in c.collision_pairs if (a in moving or b in moving) and a!=c.payload and b!=c.payload]);aa,bb=pairs.T
    radii=c.model.geom_rbound[aa]+c.model.geom_rbound[bb]+.006
    def residual(x):
        q=ref.copy();q[ids]=x;c.c.update(q);T=c.c.get_transform_frame_to_world('arm_l_end_link','body').as_matrix()
        d=np.zeros(len(pairs));near=np.linalg.norm(c.data.geom_xpos[aa]-c.data.geom_xpos[bb],axis=1)<radii
        for i in np.flatnonzero(near):d[i]=max(0,.0032-mujoco.mj_geomDistance(c.model,c.data,int(aa[i]),int(bb[i]),.004,None))
        return np.r_[1000*(T[:3,3]-target[:3,3]),250*Rotation.from_matrix(target[:3,:3].T@T[:3,:3]).as_rotvec(),5000*d,.02*(x-seed)]
    fit=least_squares(residual,seed,bounds=(lo,hi),max_nfev=350,ftol=1e-10,xtol=1e-10,gtol=1e-10);e=residual(fit.x)
    if np.linalg.norm(e[:3])>.3 or np.linalg.norm(e[3:6])/250>np.deg2rad(.05):raise ValueError('Left forward shift IK failed')
    pair['joints']=c.current_joints();pair['collision']=c.collisions(['right','left']);r['left_grasp_target_in_payload']=(np.linalg.inv(B)@target).tolist()
    r['left_forward_shift_m']=forward;r['left_vertical_shift_m']=vertical;r['left_shift_position_error_m']=float(np.linalg.norm(e[:3])/1000);r['path_validated']=False
    from simulation.refine_spread_left_branch import refine
    r=refine(c,r)
    (OUT/'bilateral_final_ik.json').write_text(json.dumps(r,indent=2));shutil.copy2(source/'preview.json',OUT/'preview.json')
    settings=yaml.safe_load((ROOT/'simulation/grasp_spread_bilateral.yaml').read_text());settings['output']=str(OUT.relative_to(ROOT));settings['left_forward_shift_m']=forward;settings['left_vertical_shift_m']=vertical
    (ROOT/'simulation/grasp_spread_forward5.yaml').write_text(yaml.safe_dump(settings,sort_keys=False))
    print('LEFT shift PASSED',forward,vertical,r['left_shift_position_error_m'],flush=True)
if __name__=='__main__':main()
