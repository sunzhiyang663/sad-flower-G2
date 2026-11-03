"""Checked local grasp sequence, starting with prepositioned arms. No hardware."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import mujoco
import yaml
from simulation.refine_grasp_elbow_down import load
from simulation.prepare_map9_manipulation import joint_route
OUT=ROOT/'simulation/outputs/grasp_spread_bilateral'

def main(config_path=None):
    settings=yaml.safe_load(Path(config_path or ROOT/'simulation/grasp_spread_bilateral.yaml').read_text())
    OUT=ROOT/settings['output']
    r=json.loads((OUT/'bilateral_final_ik.json').read_text());hand=json.loads((OUT/'preview.json').read_text())
    if not r['passed']:raise ValueError('Bilateral endpoint checks required')
    cfg,c,_,_=load(ROOT/'simulation/grasp_wrist_parallel_front.yaml')
    states=[];checks=0;left_rotations=[];wrist_axes=[]
    (OUT/'motion.json').write_text(json.dumps(dict(passed=False,status='VALIDATING')))
    def save(label,j,base,B,contacts):
        nonlocal checks
        c.set_base(base);c.set_state(j,B);collision=c.collisions(contacts);checks+=1
        assert all(abs(j[n]-v)<1e-9 for n,v in r['waist_fixed'].items())
        elbow=float(c.data.body('arm_r_link3').xpos[2]-c.data.body('arm_r_link4').xpos[2])
        if elbow<.001:raise ValueError('Right elbow crossed above shoulder')
        axis=B[:3,:3].T@c.c.get_transform_frame_to_world('arm_r_end_link','body').as_matrix()[:3,2]
        yaw=float(np.degrees(np.arctan2(axis[1],axis[0])))
        if not -10.0001<=yaw<=.0001 or abs(axis[2])>1e-5:raise ValueError('Wrist axis left the allowed horizontal interval')
        wrist_axes.append([yaw,float(axis[2])]);left_rotations.append(c.c.get_transform_frame_to_world('arm_l_end_link','body').as_matrix()[:3,:3].copy())
        states.append(dict(label=label,joints=j.copy(),base=base.tolist(),payload=B.tolist(),contacts=contacts,collision=collision,right_elbow_below_shoulder_m=elbow))
    grasp=r['right_grasp'];pair=r['paired_grasp'];base=np.array(grasp['base']);B=np.array(grasp['payload']);j=grasp['joints'].copy()
    opened={**j,**hand['stages'][0]['right_hand_motor_positions_rad']}
    for back in np.linspace(settings['approach_distance_m'],0,41):
        station=base.copy();station[0,3]-=back
        save('双臂已预摆 → 底盘直进，右手保持张开',opened,station,B,[])
    print('approach passed',flush=True)
    for t in np.linspace(0,1,41)[1:]:
        q={n:opened[n]+t*(j[n]-opened[n]) for n in j};save('靠近壳体后，右手手指闭合',q,base,B,['right'])
    print('right close passed',flush=True)
    for back in np.linspace(0,settings['partial_pull_m'],51)[1:]:
        station=base.copy();station[0,3]-=back;payload=B.copy();payload[0,3]-=back
        save('右手先拽出，底盘直退 19 cm',j,station,payload,['right'])
    base=np.array(pair['base']);B=np.array(pair['payload']);goal=pair['joints'].copy()
    print('pull passed',flush=True)
    # Fit a genuinely open left hand at its final wrist before moving the arm.
    names=c.hand_names['left'];lo,hi=np.array([c.model.joint(n).range for n in names]).T
    c.set_base(base);c.set_state(goal,B);ref=c.data.qpos.copy();ids=np.array([int(c.model.joint(n).qposadr[0]) for n in names])
    geoms=[g for g in c.robot if c.model.body(int(c.model.geom_bodyid[g])).name.startswith('hand_l_')]
    pairs=[(a,b) for a,b in c.collision_pairs if (a in geoms or b in geoms)]
    aa,bb=np.array(pairs).T;radii=c.model.geom_rbound[aa]+c.model.geom_rbound[bb]+.008
    seed=np.array([j[n] for n in names])
    def opening(x):
        q=ref.copy();q[ids]=x
        for n,m in c.mimics.items():
            if m.get('joint') in names:q[int(c.model.joint(n).qposadr[0])]=q[int(c.model.joint(m.get('joint')).qposadr[0])]*float(m.get('multiplier',1))+float(m.get('offset',0))
        c.c.update(q);d=np.full(len(pairs),.02);near=np.linalg.norm(c.data.geom_xpos[aa]-c.data.geom_xpos[bb],axis=1)<radii
        for k in np.flatnonzero(near):d[k]=mujoco.mj_geomDistance(c.model,c.data,int(aa[k]),int(bb[k]),.02,None)
        return np.r_[5000*np.maximum(.004-d,0),.02*(x-seed)]
    fit=least_squares(opening,np.clip(seed,lo+1e-8,hi-1e-8),bounds=(lo,hi),max_nfev=250,diff_step=1e-4)
    left_open=dict(zip(names,fit.x.tolist()))
    for t in np.linspace(0,1,31)[1:]:
        q=j.copy();q.update({n:j[n]+t*(left_open[n]-j[n]) for n in names});save('左手在下方打开，准备接入',q,base,B,['right'])
    # At the terminal wrist, the open palm may be within the generic 3 mm
    # clearance. Permit intended face proximity, while rejecting penetration.
    penetration=c.cfg['contact_penetration_tolerance_m'];c.cfg['contact_penetration_tolerance_m']=0.
    start={**j,**left_open};end={**goal,**left_open};c.set_state(end,B);c.collisions(['right','left'])
    route_cfg={**cfg,'joint_planning_reserve_m':0.,'joint_step_rad':.015}
    samples,optimization=joint_route(c,start,end,c.names['left'],B,route_cfg,contacts=['right','left'])
    for q in samples:save('右手保持，左手从下方接入',q,base,B,['right','left'])
    c.cfg['contact_penetration_tolerance_m']=penetration
    print('left approach passed',flush=True)
    for t in np.linspace(0,1,41)[1:]:
        q={n:end[n]+t*(goal[n]-end[n]) for n in goal};save('左手闭合，双手形成托抓构型',q,base,B,['right','left'])
    # Recheck all interpolated states at finer joint/base sampling.
    extra=0
    for a,b in zip(states,states[1:]):
        count=max(2,int(np.ceil(max(abs(a['joints'][n]-b['joints'][n]) for n in j)/.0075)))
        for t in np.linspace(0,1,count+1)[1:-1]:
            c.cfg['contact_penetration_tolerance_m']=0. if b['label']=='右手保持，左手从下方接入' else penetration
            q={n:a['joints'][n]+t*(b['joints'][n]-a['joints'][n]) for n in j}
            c.set_base((1-t)*np.array(a['base'])+t*np.array(b['base']))
            c.set_state(q,(1-t)*np.array(a['payload'])+t*np.array(b['payload']));c.collisions(b['contacts']);extra+=1
    result=dict(passed=True,states=states,samples_checked=checks,extra_interpolation_checks=extra,left_route_optimization=optimization,
        scope='prepositioned arms, straight base approach, close right, pull 19cm, open/approach/close left',
        default_to_preposition_planned=False,lift_planned=False,navigation_planned=False,dynamics_validated=False,hardware_commands_sent=False,
        base_shift_xy_m=r['base_shift_xy_m'],waist_fixed=r['waist_fixed'],collision_model=c.metadata)
    (OUT/'motion.json').write_text(json.dumps(result,indent=2));print('PASSED',len(states),extra,flush=True)
    audit=dict(passed=True,samples_checked=checks,additional_interpolation_checks=extra,
        minimum_right_elbow_below_shoulder_m=min(s['right_elbow_below_shoulder_m'] for s in states),
        wrist_yaw_deg_range=[min(a[0] for a in wrist_axes),max(a[0] for a in wrist_axes)],
        maximum_wrist_axis_vertical_component=max(abs(a[1]) for a in wrist_axes),
        left_wrist_accumulated_rotation_deg=float(sum(np.degrees(Rotation.from_matrix(a.T@b).magnitude()) for a,b in zip(left_rotations,left_rotations[1:]))),
        left_joint_travel_rad=optimization['after']['total_joint_travel_rad'],
        left_each_joint_travel_deg={n:float(np.degrees(v)) for n,v in optimization['after']['per_joint_travel_rad'].items()},
        base_shift_xy_m=r['base_shift_xy_m'],left_wait_shift_world_m=r['left_wait_shift_world_m'],
        default_to_preposition_planned=False,lift_planned=False,navigation_planned=False,return_to_default_planned=False,
        full_intra_hand_self_collision_checked=False,dynamics_validated=False)
    (OUT/'motion_audit.json').write_text(json.dumps(audit,indent=2))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--config');args=parser.parse_args();main(args.config)
