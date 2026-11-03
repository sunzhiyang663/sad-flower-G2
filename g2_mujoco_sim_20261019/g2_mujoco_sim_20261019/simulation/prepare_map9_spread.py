"""Use the accepted spread grip for the complete offline pickup sequence."""
import json
import numpy as np
import yaml
from simulation.build_map9_scene import path
from simulation.prepare_map9_cross_grip import wrap_contacts
from simulation.prepare_map9_preposition import support_patch
from manipulation.wrist_pose import pose7

def prepare_pickup(c,plan,inputs,replay,task,out,default_joints,cached_joint_route):
    local=json.loads((path(task['spread_source'])/'motion.json').read_text())
    if not local['passed']:raise ValueError('Validate spread grasp first')
    states=local['states'];first=states[0];station=np.array(first['base']);B=np.array(first['payload'])
    prebase=station.copy();prebase[0,3]-=task['motion']['preposition']['preapproach_backoff_m']-.1
    base=prebase.copy();home=default_joints(plan['config'],first['joints']);j=home.copy()
    qpos=[];payload=[];labels=[];bases=[]
    def save(label,contacts):
        c.set_base(base);c.set_state(j,B);c.collisions(contacts)
        qpos.append(c.data.qpos.copy());payload.append(np.linalg.inv(base)@B);bases.append(base.copy());labels.append(label)
    save('default',[]);default_q=qpos[-1].copy()
    # Reconstruct the assumed optical observation at the actual default head pose.
    c.set_base(np.eye(4));c.set_state(inputs['initial_joints'],B)
    mount=np.linalg.inv(c.c.get_transform_frame_to_world('head_link3','body').as_matrix())@np.array(inputs['camera_observation']['T_base_camera'])
    c.set_base(base);c.set_state(j,B)
    camera=np.linalg.inv(base)@c.c.get_transform_frame_to_world('head_link3','body').as_matrix()@mount
    (out/'assumed_camera_observation.yaml').write_text(yaml.safe_dump(dict(camera_frame='assumed_head_color',T_base_camera=camera.tolist(),T_camera_battery=(np.linalg.inv(camera)@np.linalg.inv(base)@B).tolist(),real_observation=False)))
    target={**j,**{n:first['joints'][n] for n in c.names['waist']+c.names['neck']}}
    samples=cached_joint_route(out/'lower_body.yaml',c,j,target,c.names['waist']+c.names['neck'],B,task['motion'])
    for sample in samples:j=sample;save('lower_waist',[])
    target={**j,**{n:first['joints'][n] for group in c.hand_names.values() for n in group}}
    initial=j.copy()
    for t in np.linspace(0,1,41)[1:]:j={n:initial[n]+t*(target[n]-initial[n]) for n in j};save('prepare_hands',[])
    for side in ('right','left'):
        target={**j,**{n:first['joints'][n] for n in c.names[side]}}
        samples=cached_joint_route(out/f'default_to_preposition_{side}.yaml',c,j,target,c.names[side],B,task['motion'])
        for sample in samples:j=sample;save('arm_plan',[])
    for t in np.linspace(0,1,121)[1:]:base=(1-t)*prebase+t*station;save('base_approach',[])
    label_map=['base_approach','right_grasp','right_partial_drag','left_open','left_entry','left_grasp'];seen=[]
    for s in states:
        if s['label'] not in seen:seen.append(s['label'])
        j=s['joints'];base=np.array(s['base']);B=np.array(s['payload']);save(label_map[seen.index(s['label'])],s['contacts'])
    wrap=wrap_contacts(c,.004,.01);(out/'wrap_contact_audit.json').write_text(json.dumps(wrap,indent=2))
    if not wrap['passed']:raise ValueError('Shifted left hand no longer wraps shell')
    support=support_patch(c,task['motion']['preposition']['support']);(out/'support_audit.json').write_text(json.dumps(support,indent=2))
    initial_base=base.copy();initial_B=B.copy()
    for d in np.linspace(0,.06,31)[1:]:
        base=initial_base.copy();base[0,3]-=d;B=initial_B.copy();B[0,3]-=d;save('paired_full_drag',['left','right'])
    right=c.c.get_transform_frame_to_world('arm_r_end_link','body').as_matrix();attachment=np.linalg.inv(right)@B
    ends={s:c.c.get_transform_frame_to_world(f'arm_{letter}_end_link','body').as_matrix().copy() for s,letter in [('left','l'),('right','r')]}
    goals={s:T.copy() for s,T in ends.items()}
    for T in goals.values():T[2,3]+=task['motion']['preposition']['lift_m']
    print('Spread paired lift',flush=True)
    result=c.request(dict(operation='motion',T_world_base=base.tolist(),joints=j,battery=B.tolist(),contacts=['left','right'],ends={s:pose7(T) for s,T in ends.items()},goals={s:T.tolist() for s,T in goals.items()},allow_waist=False,payload_from_right=attachment.tolist()))
    for sample in result['path']:
        j=sample['joints'];c.set_state(j,B);B=c.c.get_transform_frame_to_world('arm_r_end_link','body').as_matrix()@attachment;save('lift',['left','right'])
    initial_base=base.copy();initial_B=B.copy()
    for d in np.linspace(0,task['motion'].get('carry_backoff_m',.35),101)[1:]:
        base=initial_base.copy();base[0,3]-=d;B=initial_B.copy();B[0,3]-=d;save('carry_backoff',['left','right'])
    np.savez_compressed(out/'pickup.npz',qpos=qpos,payload=payload,base_offsets=bases,labels=labels,default_qpos=default_q)
    (out/'pickup_profile.json').write_text(json.dumps(dict(mode='spread_forward5',station=states[0]['base'],carry_base=base.tolist(),T_right_end_battery=attachment.tolist(),left_forward_shift_m=.05,stationary_waist_during_extraction=True,hardware_commands_sent=False),indent=2))
    print('Spread pickup saved',len(labels),flush=True)
    return qpos,payload,labels,default_q
