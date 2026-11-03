"""Measure arm travel, fixed-joint chassis moves and near-contact COM support."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.spatial.transform import Rotation
import yaml
from simulation.full_task_config import load_task
from simulation.build_map9_scene import path
from simulation.pose_grasp_check import Checker
from simulation.prepare_map9_preposition import support_patch


def movement(checker,record):
    k=np.flatnonzero(record['labels']=='arm_plan');indices=np.r_[k[0]-1,k];result={}
    for side,letter in [('left','l'),('right','r')]:
        rotations=[];xyz=[]
        for q in record['qpos'][indices]:
            checker.c.update(q);T=checker.c.get_transform_frame_to_world(f'arm_{letter}_end_link','body').as_matrix()
            rotations.append(T[:3,:3].copy());xyz.append(T[:3,3].copy())
        angles=[Rotation.from_matrix(a.T@b).magnitude() for a,b in zip(rotations[:-1],rotations[1:])]
        addresses=[checker.model.joint(n).qposadr[0] for n in checker.names[side]]
        result[side]=dict(wrist_rotation_deg=float(np.rad2deg(sum(angles))),
            shortest_wrist_rotation_deg=float(np.rad2deg(Rotation.from_matrix(rotations[0].T@rotations[-1]).magnitude())),
            wrist_path_length_m=float(np.linalg.norm(np.diff(xyz,axis=0),axis=1).sum()),
            joint_travel_rad=float(np.abs(np.diff(record['qpos'][indices][:,addresses],axis=0)).sum()))
    return result


def main():
    cfg=yaml.safe_load((ROOT/'simulation/map9.yaml').read_text());task=load_task()
    if 'preposition' not in task['motion']:
        print('No preposition profile; historical mode unchanged');return
    out=path(task['output']);source=path(cfg['carry_reference']);plan=json.loads((source/'source_plan.json').read_text())
    inputs=yaml.safe_load((source/'source_inputs.yaml').read_text())['inputs'];checker=Checker(plan['config'],inputs['scene']);checker.scene=inputs['scene']
    if task.get('spread_source'):checker.cfg['contact_faces']['right']=['-x','-y','+y','+z']
    pickup=np.load(out/'pickup.npz');place=np.load(out/'placement.npz');new=movement(checker,pickup)
    baseline=path(task['motion']['preposition']['comparison_baseline']);old=movement(checker,np.load(baseline/'pickup.npz'))
    arm_savings=1-sum(x['joint_travel_rad'] for x in new.values())/sum(x['joint_travel_rad'] for x in old.values())
    base_checks={}
    cross=task['motion']['preposition'].get('mode')=='cross_wrap'
    for label in (['base_approach','right_partial_drag','paired_full_drag'] if cross else ['base_approach','base_pull']):
        k=np.flatnonzero(pickup['labels']==label);k=np.r_[k[0]-1,k];bases=pickup['base_offsets'][k]
        base_checks[label]=dict(distance_m=float(np.linalg.norm(np.diff(bases[:,:3,3],axis=0),axis=1).sum()),
            max_joint_change_rad=float(np.ptp(pickup['qpos'][k],axis=0).max()),
            lateral_change_m=float(np.ptp(bases[:,1,3])),orientation_change=float(np.abs(bases[:,:3,:3]-bases[0,:3,:3]).max()))
    waist=[checker.model.joint(n).qposadr[0] for n in checker.names['waist']]
    waist_delta=np.rad2deg(np.max(np.abs(pickup['qpos'][:,waist]-pickup['qpos'][0,waist]),axis=0)).tolist()
    limits=[]
    for n in checker.names['left']+checker.names['right']:
        address=checker.model.joint(n).qposadr[0];lo,hi=checker.model.joint(n).range
        values=np.r_[pickup['qpos'][:,address],place['qpos'][:,address]]
        limits.append(dict(joint=n,minimum_limit_margin_deg=float(np.rad2deg(min(values.min()-lo,hi-values.max())))))
    shoulders=[]
    for k in [0,int(np.flatnonzero(pickup['labels']=='arm_plan')[-1])]:
        checker.c.update(pickup['qpos'][k]);shoulders.append(float(checker.c.get_transform_frame_to_world('arm_l_link1','body').translation()[2]))
    checker.set_base(pickup['base_offsets'][-1]);checker.c.update(pickup['qpos'][-1]);checker.set_battery(pickup['base_offsets'][-1]@pickup['payload'][-1])
    support=support_patch(checker,task['motion']['preposition']['support'])
    right=checker.c.get_transform_frame_to_world('arm_r_end_link','body').as_matrix()
    attachment=np.linalg.inv(right)@checker.battery
    profile_file=out/'pickup_profile.json';profile=json.loads(profile_file.read_text())
    profile['T_right_end_battery']=attachment.tolist();profile_file.write_text(json.dumps(profile,indent=2))
    # Geometric centre must project inside the table, including the configured
    # front overhang needed to clear the right-hand camera below the grasp.
    placement_base=place['base_offset'] if 'base_offset' in place else np.eye(4)
    placement_bases=place['base_offsets'] if 'base_offsets' in place else np.repeat(placement_base[None],len(place['qpos']),axis=0)
    com=(placement_bases[-1]@place['payload'][-1])[:3,3];table_xy=np.array([cfg['table']['centre_forward_m'],0.]);half=np.array([cfg['table']['size_m'][1],cfg['table']['size_m'][0]])/2
    table_margin=float((half-np.abs(com[:2]-table_xy)).min())
    wrap=json.loads((out/'wrap_contact_audit.json').read_text()) if cross else None
    carry_wrap=None
    if cross:
        from simulation.prepare_map9_cross_grip import wrap_contacts
        carry_wrap=wrap_contacts(checker,task['motion']['preposition']['wrap_contact_gap_m'],task['motion']['preposition']['wrap_depth_m'])
        k=np.flatnonzero(place['labels']=='right_push_on_table');k=np.r_[k[0]-1,k];bases=placement_bases[k]
        base_checks['right_push_on_table']=dict(distance_m=float(np.linalg.norm(np.diff(bases[:,:3,3],axis=0),axis=1).sum()),
            max_joint_change_rad=float(np.ptp(place['qpos'][k],axis=0).max()),lateral_change_m=float(np.ptp(bases[:,1,3])),
            orientation_change=float(np.abs(bases[:,:3,:3]-bases[0,:3,:3]).max()))
    valid=((wrap['passed'] and carry_wrap['passed']) if cross else support['com_inside_support_polygon']) and table_margin>0 and all(v['max_joint_change_rad']<1e-12 and v['lateral_change_m']<1e-12 and v['orientation_change']<1e-12 for v in base_checks.values())
    grasp_sequence=None
    if cross:
        def span(record,label):
            indices=np.flatnonzero(record['labels']==label)
            if not len(indices):raise ValueError('Missing required grasp stage: '+label)
            return [int(indices[0]),int(indices[-1])]
        stages={label:span(pickup,label) for label in ['right_grasp','right_partial_drag','left_grasp','paired_full_drag','lift']}
        placement_stages={label:span(place,label) for label in ['front_land','left_release','left_withdraw','right_push_on_table','right_release']}
        ordered=all(a[1]<b[0] for a,b in zip(stages.values(),list(stages.values())[1:])) and all(a[1]<b[0] for a,b in zip(placement_stages.values(),list(placement_stages.values())[1:]))
        reference=np.load(source/'replay.npz');reference_index=int(np.flatnonzero(reference['stages']=='right_grasp')[-1])
        checker.set_base(np.eye(4));checker.c.update(reference['qpos'][reference_index])
        original_R=checker.c.get_transform_frame_to_world('arm_r_end_link','body').as_matrix()[:3,:3].copy()
        if task.get('spread_source'):
            accepted=json.loads((path(task['spread_source'])/'bilateral_final_ik.json').read_text())
            original_R=np.array(accepted['right_target_world'])[:3,:3]
        checker.c.update(pickup['qpos'][stages['right_grasp'][1]])
        current_R=checker.c.get_transform_frame_to_world('arm_r_end_link','body').as_matrix()[:3,:3]
        right_error=float(Rotation.from_matrix(original_R.T@current_R).magnitude())
        checker.c.update(pickup['qpos'][stages['left_grasp'][1]])
        palm_R=checker.c.get_transform_frame_to_world('hand_l_base_link','body').as_matrix()[:3,:3]
        upward=float(palm_R[2,0]);finger_angle=float(np.rad2deg(np.arccos(np.clip(abs(palm_R[0,2]),0,1))))
        landed=placement_bases[placement_stages['front_land'][1]]@place['payload'][placement_stages['front_land'][1]]
        release=placement_bases[placement_stages['left_release'][1]]@place['payload'][placement_stages['left_release'][1]]
        landing_gap=float(landed[2,3]-cfg['payload']['size_m'][2]/2-(cfg['table']['size_m'][2]-cfg['base_height_m']))
        front=cfg['table']['centre_forward_m']-cfg['table']['size_m'][1]/2
        overlap=float(landed[0,3]+cfg['payload']['size_m'][0]/2-front)
        release_motion=float(np.abs(landed-release).max())
        partial=task['motion']['preposition']['right_partial_drag_m']
        initial=np.array(inputs['camera_observation']['T_base_camera'])@np.array(inputs['camera_observation']['T_camera_battery'])
        shelf_front=inputs['scene']['support']['T_base'][0][3]-inputs['scene']['support']['size_m'][0]/2
        remaining=float(initial[0,3]-partial+cfg['payload']['size_m'][0]/2-shelf_front)
        before_lift=[pickup['base_offsets'][k]@pickup['payload'][k] for k in range(stages['right_grasp'][1],stages['left_grasp'][1]+1)]
        early_vertical_motion=float(max(abs(T[2,3]-initial[2,3]) for T in before_lift))
        push_indices=np.flatnonzero(place['labels']=='right_push_on_table')
        push_bottom_heights=[(placement_bases[k]@place['payload'][k])[2,3]-cfg['payload']['size_m'][2]/2 for k in push_indices]
        push_height_error=float(max(abs(z-(cfg['table']['size_m'][2]-cfg['base_height_m'])) for z in push_bottom_heights))
        ok=ordered and early_vertical_motion<.001 and push_height_error<.001 and right_error<.002 and upward>.9 and finger_angle>=45 and abs(landing_gap)<.001 and overlap>0 and release_motion<1e-9 and remaining>0
        grasp_sequence=dict(passed=ok,pickup_stages=stages,placement_stages=placement_stages,right_orientation_error_rad=right_error,
            left_palm_up_dot=upward,left_fingers_to_long_edge_deg=finger_angle,front_landing_gap_m=landing_gap,
            front_landing_overlap_m=overlap,payload_motion_during_left_release=release_motion,shelf_overlap_after_right_partial_drag_m=remaining,
            maximum_vertical_motion_before_left_grip_m=early_vertical_motion,push_bottom_height_error_m=push_height_error)
        valid=valid and ok
    report=dict(passed=valid,current=new,baseline=old,comparison_baseline=str(baseline),
        total_arm_joint_travel_reduction_fraction=arm_savings,waist_max_joint_change_deg=waist_delta,
        shoulder_height_change_m=shoulders[1]-shoulders[0],arm_limit_margins=limits,base_segments=base_checks,carry_support=support,
        table_com_margin_m=table_margin,table_front_overhang_m=max(0.,-task['motion']['placement_edge_gap_m']),
        grasp_sequence=grasp_sequence,left_wrap=wrap,carry_wrap=carry_wrap,dynamics_validated=False,hardware_commands_sent=False)
    if task.get('spread_source'):
        k=int(np.flatnonzero(pickup['labels']=='left_grasp')[-1]);checker.set_base(pickup['base_offsets'][k]);checker.c.update(pickup['qpos'][k])
        actual=np.linalg.inv(pickup['base_offsets'][k]@pickup['payload'][k])@checker.c.get_transform_frame_to_world('arm_l_end_link','body').as_matrix()
        expected=np.array(accepted['left_grasp_target_in_payload'])
        reset_error=float(np.abs(place['qpos'][-1]-pickup['default_qpos']).max())
        shift_error=float(np.linalg.norm(actual[:3,3]-expected[:3,3]))
        report['left_forward_shift_m']=accepted['left_forward_shift_m'];report['left_target_error_m']=shift_error
        report['default_reset_max_error_rad']=reset_error
        report['passed']=valid and shift_error<.0003 and reset_error<1e-8
        valid=report['passed']
    (out/'preposition_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    if not valid:raise ValueError('Preposition / support audit failed')


if __name__=='__main__':main()
