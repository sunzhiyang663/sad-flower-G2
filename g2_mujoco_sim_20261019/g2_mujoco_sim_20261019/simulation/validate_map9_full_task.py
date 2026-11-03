"""Whole-body/payload geometry and simulated lidar audit of the full sequence."""
import json
import hashlib
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import yaml
from simulation.full_task_config import load_task
from simulation.build_map9_scene import path,lidar_scan,filter_payload
from simulation.map9_route_preview import Clearance


def main():
    cfg=yaml.safe_load((ROOT/'simulation/map9.yaml').read_text());task=load_task();out=path(task['output'])
    from simulation.render_map9_full_task import make_sequence
    meta=json.loads((path(cfg['snapshot'])/'map.json').read_text())
    model=mujoco.MjModel.from_binary_path(str(out/'scene.mjb'));saved=dict(np.load(out/'state.npz'))
    def collision_hash():
        return hashlib.sha256(b''.join(getattr(model,name).tobytes() for name in
            ('geom_type','geom_size','geom_pos','geom_quat','geom_bodyid','geom_group','geom_contype','geom_conaffinity'))).hexdigest()
    before=collision_hash()
    checker=Clearance(model,saved,cfg['base_height_m'],.1);report={};lidar=[]
    for goal in task['goal_guide_ids']:
        sequence,_,_=make_sequence(goal,cfg,task,out,meta);minima=[];failures=[];empty_shell_minimum=.1
        for i,(p,q,T,label,attached) in enumerate(sequence):
            checker.data.qpos[:]=q;checker.saved['T_base_payload']=T
            values,witness=checker.check(p)
            minima.append(values)
            # The shell is a stationary obstacle before grasping, even though
            # the vehicle's empty footprint does not contain an attached load.
            if label.startswith(('1 ','2 ')):
                robot=np.array(checker.robot);shell=checker.payload
                near=np.linalg.norm(checker.data.geom_xpos[robot]-checker.data.geom_xpos[shell],axis=1)<=model.geom_rbound[robot]+model.geom_rbound[shell]+.1
                for geom in robot[near]:
                    distance=float(mujoco.mj_geomDistance(model,checker.data,int(geom),shell,.1,None))
                    empty_shell_minimum=min(empty_shell_minimum,distance)
                    if distance<task['motion']['min_navigation_clearance_m']:
                        failures.append(dict(frame=i,label=label,category='empty_robot_vs_stationary_shell',distance_m=distance))
            # Placement/rack-top contact is legal for the shell only. The
            # extraction/table checker separately checks the actual contact face.
            threshold=np.array([task['motion']['min_navigation_clearance_m']]*3+[-task['motion']['contact_tolerance_m']])
            if np.any(values<threshold):failures.append(dict(frame=i,label=label,distances=values.tolist(),witness=witness))
            if attached and label.startswith('6 ') and (i%160==0
                    or not sequence[i-1][3].startswith('6 ')
                    or not sequence[min(i+1,len(sequence)-1)][3].startswith('6 ')):
                Tb=checker.set_pose(p)
                for sensor in ('front','back'):
                    points,ids=lidar_scan(model,checker.data,Tb,cfg['lidar'][sensor+'_extrinsic'],cfg['lidar'])
                    removed=filter_payload(points,T,np.array(cfg['payload']['size_m'])/2,cfg['payload']['filter_numeric_epsilon_m'])
                    true=ids==checker.payload
                    lidar.append(dict(goal=goal,frame=i,sensor=sensor,payload_points=int(true.sum()),
                        missed=int((true&~removed).sum()),environment_removed=int((~true&removed).sum())))
            if i%250==0:print(f'G{goal} geometry {i}/{len(sequence)}',flush=True)
        minima=np.asarray(minima)
        poses=np.array([s[0] for s in sequence]);delta=np.diff(poses[:,:2],axis=0)
        dyaw=np.arctan2(np.sin(np.diff(poses[:,2])),np.cos(np.diff(poses[:,2])))
        heading=poses[:-1,2]+dyaw/2
        lateral=np.sum(delta*np.c_[-np.sin(heading),np.cos(heading)],axis=1)
        max_base_step=float(np.linalg.norm(delta,axis=1).max());max_lateral=float(np.abs(lateral).max())
        if task.get('spread_source') and (max_base_step>.025 or max_lateral>1e-7):
            failures.append(dict(category='discontinuous_or_lateral_base_motion',max_step_m=max_base_step,max_lateral_m=max_lateral))
        np.savez_compressed(out/f'G{goal}_full_clearances.npz',minimum_distances=minima)
        report[str(goal)]=dict(samples=len(sequence),minimum_by_category_m=minima.min(axis=0).tolist(),
            empty_robot_shell_minimum_m=empty_shell_minimum,max_base_step_m=max_base_step,max_lateral_step_m=max_lateral,failures=failures,passed=not failures)
        print('G',goal,'failures',len(failures),failures[:2],flush=True)
    vehicle=json.loads((out/'vehicle_routes.json').read_text())
    kinematic=all(v['max_lateral_step_m']<1e-7 and v['max_curvature_per_m']<=1.001/task['vehicle']['turning_radius_m'] for v in vehicle.values())
    unchanged=before==collision_hash()
    preposition_audit=json.loads((out/'preposition_audit.json').read_text()) if task['motion'].get('preposition') else None
    passed=(preposition_audit is None or preposition_audit['passed']) and all(v['passed'] for v in report.values()) and kinematic and unchanged and all(s['missed']==0 and s['environment_removed']==0 for s in lidar)
    local_audit=json.loads((out/'local_motion_audit.json').read_text()) if task.get('spread_source') else None
    if local_audit is not None:
        fresh=all(record['source_sha256']==hashlib.sha256((out/(name+'.npz')).read_bytes()).hexdigest() for name,record in local_audit['segments'].items())
        passed=passed and local_audit['passed'] and fresh and sum(s['payload_points'] for s in lidar)>0
    (out/'validation.json').write_text(json.dumps(dict(passed=passed,routes=report,vehicle=vehicle,
        kinematic_constraints_passed=kinematic,lidar=lidar,preposition_audit=preposition_audit,
        local_motion_audit=local_audit,
        perception_filter_does_not_modify_collision_geometry=unchanged,collision_geometry_sha256=before,
        contact='Shell may touch source shelf / destination table; body has 3 mm sampled clearance',
        continuous_collision_detection=False,closed_loop_navigation=False,dynamics_validated=False,
        hardware_commands_sent=False),ensure_ascii=False,indent=2))
    print('Full sequence passed:',passed,flush=True)


if __name__=='__main__':main()
