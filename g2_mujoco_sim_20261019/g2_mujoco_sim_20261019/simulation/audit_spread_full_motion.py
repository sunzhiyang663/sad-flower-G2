"""Resample local pickup/placement transitions with independent FK/collision checks."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import yaml
from scipy.spatial.transform import Rotation,Slerp
from simulation.full_task_config import load_task
from simulation.pose_grasp_check import Checker
from simulation.build_map9_scene import path

def main():
    task=load_task();out=path(task['output']);source=path('simulation/reference_inputs/extraction')
    plan=json.loads((source/'source_plan.json').read_text());config=plan['config'];config['pose_grasp']['contact_faces']['right']=['-x','-y','+y','+z']
    rack=yaml.safe_load((source/'source_inputs.yaml').read_text())['inputs']['scene'];table=yaml.safe_load((out/'placement_retract_state.yaml').read_text())['scene']
    reports={}
    def blend(a,b,t):
        T=np.eye(4);T[:3,3]=(1-t)*a[:3,3]+t*b[:3,3];T[:3,:3]=Slerp([0,1],Rotation.from_matrix(np.array([a[:3,:3],b[:3,:3]])))(t).as_matrix();return T
    for filename,scene in [('pickup',rack),('placement',table)]:
        c=Checker(config,scene);c.scene=scene;r=np.load(out/(filename+'.npz'));checks=0
        for k in range(1,len(r['qpos'])):
            label=str(r['labels'][k]);q0,q1=r['qpos'][k-1:k+1];base0,base1=r['base_offsets'][k-1:k+1];B0=base0@r['payload'][k-1];B1=base1@r['payload'][k]
            count=max(1,int(np.ceil(np.max(np.abs(q1-q0))/.0075)),int(np.ceil(np.linalg.norm(base1[:3,3]-base0[:3,3])/.0025)))
            if filename=='pickup':
                contacts=['left','right'] if label in ('left_entry','left_grasp','paired_full_drag','lift','carry_backoff') else ['right'] if label in ('right_grasp','right_partial_drag','left_open') else []
            else:
                contacts=['left','right'] if label in ('paired_table_hover','paired_table_approach','front_land','left_release','left_drop') else ['right'] if label in ('left_withdraw','left_park','right_arm_push','right_push_on_table','right_release','right_withdraw') else []
            c.cfg['contact_penetration_tolerance_m']=0. if label=='left_entry' else .003
            for t in np.linspace(0,1,count+1)[1:]:
                c.set_base(blend(base0,base1,t));c.c.update((1-t)*q0+t*q1);j=c.current_joints();B=blend(B0,B1,t);c.set_state(j,B);c.collisions(contacts);checks+=1
            if k%200==0:print(filename,k,checks,flush=True)
        reports[filename]=dict(passed=True,checked_samples=checks,max_joint_step_rad=.0075,max_base_step_m=.0025,
            source_sha256=hashlib.sha256((out/(filename+'.npz')).read_bytes()).hexdigest())
    report=dict(passed=True,segments=reports,continuous_collision_detection=False,dynamics_validated=False,hardware_commands_sent=False)
    (out/'local_motion_audit.json').write_text(json.dumps(report,indent=2));print(report,flush=True)
if __name__=='__main__':main()
