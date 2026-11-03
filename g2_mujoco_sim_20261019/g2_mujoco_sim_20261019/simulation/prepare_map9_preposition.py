"""Offline dual-arm preposition followed by longitudinal chassis approach.

Mink chooses wrist IK; OMPL plans the free-space arm motion. The payload remains
in the collision model, including after perception filtering. No hardware I/O.
"""
import json
import mujoco
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import yaml
from manipulation.wrist_pose import palm_mounts, pose7


def support_patch(checker, settings):
    """Projected support polygon from near-contact, upward-facing hand geometry."""
    groups=checker.model.geom_group.copy();checker.model.geom_group[:]=5
    for g in checker.robot:
        name=checker.model.body(int(checker.model.geom_bodyid[g])).name
        if name.startswith('hand_l_') and not name.endswith('camera_link'):checker.model.geom_group[g]=3
    half=checker.model.geom_size[checker.payload];T=checker.battery
    points=[];gaps=[];gid=np.zeros(1,dtype=np.int32);normal=np.zeros(3)
    step=settings['ray_step_m'];tolerance=settings['contact_gap_tolerance_m']
    try:
        for x in np.arange(-half[0],half[0]+step/2,step):
            for y in np.arange(-half[1],half[1]+step/2,step):
                world=T[:3,:3]@np.array([x,y,-half[2]])+T[:3,3];world[2]+=.01
                distance=mujoco.mj_ray(checker.model,checker.data,world,np.array([0.,0.,-1.]),
                    np.array([0,0,0,1,0,0],dtype=np.uint8),True,-1,gid,normal)
                if distance>=0 and abs(distance-.01)<=tolerance and normal[2]>=settings['minimum_upward_normal']:
                    points.append([float(x),float(y)]);gaps.append(float(distance-.01))
    finally:checker.model.geom_group[:]=groups
    com3=np.array(settings['com_in_payload_m']);gravity_local=T[:3,:3].T@np.array([0.,0.,-1.])
    com=(com3+((-half[2]-com3[2])/gravity_local[2])*gravity_local)[:2]
    result=dict(com_in_payload_m=settings['com_in_payload_m'],
        mass_kg=settings['mass_kg'],points=len(points),gap_tolerance_m=tolerance,
        minimum_upward_normal=settings['minimum_upward_normal'],geometry_only=True,dynamics_validated=False)
    if len(points)<3:
        return {**result,'com_inside_support_polygon':False,'support_polygon_m':[],
                'margin_m':None,'minimum_overturning_torque_Nm_per_kg':None}
    points=np.array(points);hull=ConvexHull(points);polygon=points[hull.vertices]
    distances=hull.equations[:,:2]@com+hull.equations[:,2]
    margin=float(-distances.max());lever=0.
    if margin<0:
        lever=min(np.linalg.norm(com-(a+np.clip((com-a)@(b-a)/((b-a)@(b-a)),0,1)*(b-a)))
                  for a,b in zip(polygon,np.roll(polygon,-1,axis=0)))
    return {**result,'com_inside_support_polygon':margin>=0,'margin_m':margin,
        'support_polygon_m':polygon.tolist(),'gap_range_m':[min(gaps),max(gaps)],
        'minimum_overturning_torque_Nm_per_kg':float(lever*settings['gravity_m_s2'])}


def prepare_pickup(checker,plan,inputs,replay,task,out,default_joints,cached_joint_route):
    profile=task['motion']['preposition'];battery=np.array(inputs['camera_observation']['T_base_camera'])@np.array(inputs['camera_observation']['T_camera_battery'])
    mount=palm_mounts(checker.cfg['urdf_path'])['left'];palm=np.array(profile['left_palm_in_payload'])
    # Reuse the previously fitted front grasp, choosing a geometric symmetry
    # before solving new arm joints. These are Cartesian goals, not taught joints.
    index=int(np.flatnonzero(replay['stages']=='right_grasp')[-1]);checker.c.update(replay['qpos'][index])
    right=checker.c.get_transform_frame_to_world('arm_r_end_link','body').as_matrix().copy()
    if profile['mirror_right_grasp']:right=battery@np.diag([1.,-1.,-1.,1.])@np.linalg.inv(battery)@right
    future=battery.copy();future[0,3]-=profile['arm_pull_m']
    left=future@palm@np.linalg.inv(mount);left[:3,3]+=profile['left_wait_offset_m']
    station=np.eye(4);station[:3,3]=profile['station_offset_m']
    prebase=station.copy();prebase[0,3]-=profile['preapproach_backoff_m']
    home=default_joints(plan['config'],inputs['initial_joints']);waist={n:home[n] for n in checker.names['waist']}
    waist[checker.names['waist'][1]]+=profile.get('waist_joint2_delta_rad',0.)
    if profile['level_torso']:waist[checker.names['waist'][2]]=-waist[checker.names['waist'][0]]-waist[checker.names['waist'][1]]
    open_right=next(h for h in plan['checks']['history'] if h['id']=='right_open')['targets']['right']
    numerical={**home,**waist,**profile['left_motor_targets'],**open_right}
    for n,offset in profile.get('ik_seed_offsets_rad',{}).items():numerical[n]+=offset
    checker.set_base(station);checker.set_state(numerical,battery)
    checker.solve({'left':left,'right':right},['right'],constructing=True);seed=checker.current_joints()
    (out/'preposition_ik.yaml').write_text(yaml.safe_dump(dict(joints=seed,T_station=station.tolist(),
        goals={'left':left.tolist(),'right':right.tolist()},method='Mink / DAQP numerical endpoint; OMPL transition follows')))
    qpos=[];payload=[];bases=[];labels=[];B=prebase.copy();joints=home.copy();attachment=None
    checker.set_base(B);checker.set_state(joints,battery);checker.collisions([])
    default_q=checker.data.qpos.copy()
    def save(label,contacts):
        checker.collisions(contacts);qpos.append(checker.data.qpos.copy())
        payload.append(np.linalg.inv(B)@checker.battery);bases.append(B.copy());labels.append(label)
    save('default',[])
    # Observation transforms are explicitly local to the capture-time base.
    checker.set_base(np.eye(4));checker.set_state(inputs['initial_joints'],battery)
    head_mount=np.linalg.inv(checker.c.get_transform_frame_to_world('head_link3','body').as_matrix())@np.array(inputs['camera_observation']['T_base_camera'])
    checker.set_base(B);checker.set_state(joints,battery)
    camera=np.linalg.inv(B)@checker.c.get_transform_frame_to_world('head_link3','body').as_matrix()@head_mount
    (out/'assumed_camera_observation.yaml').write_text(yaml.safe_dump(dict(camera_frame='assumed_head_color',
        capture_timestamp_ns=None,T_base_camera=camera.tolist(),T_camera_battery=(np.linalg.inv(B@camera)@battery).tolist()),sort_keys=False))
    result=checker.request(dict(operation='waist_motion',T_world_base=B.tolist(),targets=waist,joints=joints,battery=battery.tolist()))
    for sample in result['path']:
        checker.set_state(sample['joints'],battery);save('lower_waist',[])
    joints=checker.current_joints();before=joints.copy();targets={**profile['left_motor_targets'],**open_right}
    for alpha in np.linspace(0,1,31)[1:]:
        joints={**before,**{n:before[n]+alpha*(v-before[n]) for n,v in targets.items()}}
        checker.set_state(joints,battery);save('prepare_hands',[])
    names=checker.names['left']+checker.names['right']
    # A collision-checked Mink Cartesian candidate gives OMPL a short spatial
    # route before its joint-travel optimization. Both wrists move on lines;
    # the near-half-turn sign follows the feasible numerical joint branch.
    starts={s:checker.c.get_transform_frame_to_world(f'arm_{l}_end_link','body').as_matrix().copy()
            for s,l in [('left','l'),('right','r')]}
    ends={'left':left.copy(),'right':right.copy()}
    for T in ends.values():T[:3,3]+=prebase[:3,3]-station[:3,3]
    rotations={s:Rotation.from_matrix(starts[s][:3,:3].T@ends[s][:3,:3]).as_rotvec() for s in ends}
    r=rotations['right'];angle=np.linalg.norm(r)
    if angle>np.pi-profile['arc']['near_half_turn_tolerance_rad']:
        probe={n:joints[n]+.01*(seed[n]-joints[n]) for n in joints};checker.set_state(probe,battery)
        Rprobe=checker.c.get_transform_frame_to_world('arm_r_end_link','body').as_matrix()[:3,:3]
        if profile['arc'].get('near_half_turn_direction')=='opposite' or Rotation.from_matrix(starts['right'][:3,:3].T@Rprobe).as_rotvec()@r<0:
            rotations['right']=-r/angle*(2*np.pi-angle)
    checker.set_state(joints,battery);candidate=[]
    original_minimum=checker.collision_limit.minimum_distance_from_collisions
    checker.collision_limit.minimum_distance_from_collisions+=profile['preposition_ik_extra_clearance_m']
    count=int(np.ceil(max(np.linalg.norm(r) for r in rotations.values())/task['motion']['joint_step_rad']))
    for fraction in np.linspace(0,1,count+1)[1:]:
        goals={s:np.eye(4) for s in ends}
        for s,T in goals.items():
            T[:3,3]=(1-fraction)*starts[s][:3,3]+fraction*ends[s][:3,3]
            T[:3,:3]=starts[s][:3,:3]@Rotation.from_rotvec(fraction*rotations[s]).as_matrix()
        checker.solve(goals,[]);candidate.append(checker.current_joints())
    checker.collision_limit.minimum_distance_from_collisions=original_minimum
    goal=checker.current_joints()
    (out/'cartesian_preposition_candidate.yaml').write_text(yaml.safe_dump(candidate))
    samples=cached_joint_route(out/'default_to_preposition_joints.yaml',checker,joints,goal,names,battery,task['motion'],
        wrist_arc=profile.get('arc'),candidate=candidate)
    for sample in samples:
        checker.set_state(sample,battery);save('arm_plan',[])
    joints=checker.current_joints()
    for alpha in np.linspace(0,1,int(np.ceil(profile['preapproach_backoff_m']/checker.cfg['path_step_m']))+1)[1:]:
        B=prebase.copy();B[:3,3]=(1-alpha)*prebase[:3,3]+alpha*station[:3,3]
        checker.set_base(B);checker.set_state(joints,battery);save('base_approach',['right'])
    before=joints.copy();targets=next(h for h in plan['checks']['history'] if h['id']=='right_grasp')['targets']['right']
    for alpha in np.linspace(0,1,31)[1:]:
        joints={**before,**{n:before[n]+alpha*(v-before[n]) for n,v in targets.items()}}
        checker.set_state(joints,battery);save('right_grasp',['right'])
    attachment=np.linalg.inv(checker.c.get_transform_frame_to_world('arm_r_end_link','body').as_matrix())@battery
    def move(label,goals,contacts):
        nonlocal joints
        ends={s:checker.c.get_transform_frame_to_world(f'arm_{"l" if s=="left" else "r"}_end_link','body').as_matrix() for s in goals}
        result=checker.request(dict(operation='motion',T_world_base=B.tolist(),joints=joints,battery=checker.battery.tolist(),
            contacts=contacts,ends={s:pose7(T) for s,T in ends.items()},goals={s:T.tolist() for s,T in goals.items()},payload_from_right=attachment.tolist()))
        for sample in result['path']:
            checker.set_state(sample['joints'],checker.battery);checker.set_battery(checker.c.get_transform_frame_to_world('arm_r_end_link','body').as_matrix()@attachment);save(label,contacts)
        joints=checker.current_joints();print(label,len(result['path']),flush=True)
    if profile['arm_pull_m']:
        right=checker.c.get_transform_frame_to_world('arm_r_end_link','body').as_matrix().copy();right[0,3]-=profile['arm_pull_m'];move('partial_pull',{'right':right},['right'])
    target_base=B.copy();target_base[0,3]-=profile['base_pull_m']
    result=checker.request(dict(operation='base_motion',T_world_base=B.tolist(),goal_base=target_base.tolist(),joints=joints,
        battery=checker.battery.tolist(),contacts=['right'],payload_from_right=attachment.tolist()))
    for sample in result['path']:
        B=np.array(sample['T_world_base']);checker.set_base(B);checker.set_state(sample['joints'],sample['battery']);save('base_pull',['right'])
    left=checker.battery@palm@np.linalg.inv(mount);left[2,3]-=profile['left_entry_gap_m']
    down=checker.c.get_transform_frame_to_world('arm_l_end_link','body').as_matrix().copy();down[2,3]=left[2,3]
    move('left_lower_entry',{'left':down},['right']);move('left_entry',{'left':left},['right'])
    left[2,3]+=profile['left_entry_gap_m']+profile['left_seat_raise_m'];move('left_seat',{'left':left},['left','right'])
    audit=support_patch(checker,profile['support'])
    (out/'support_audit.json').write_text(json.dumps(audit,indent=2))
    if not audit['com_inside_support_polygon']:raise ValueError('COM projection is outside near-contact support polygon')
    goals={s:checker.c.get_transform_frame_to_world(f'arm_{l}_end_link','body').as_matrix().copy() for s,l in [('left','l'),('right','r')]}
    for T in goals.values():T[2,3]+=profile['lift_m']
    move('lift',goals,['left','right'])
    np.savez_compressed(out/'pickup.npz',qpos=qpos,payload=payload,base_offsets=bases,labels=labels,default_qpos=default_q)
    (out/'pickup_profile.json').write_text(json.dumps(dict(preposition_before_base_approach=True,
        stationary_waist_during_extraction=True,station=station.tolist(),carry_base=B.tolist(),
        support=audit,T_right_end_battery=attachment.tolist(),arm_pull_m=profile['arm_pull_m'],base_pull_m=profile['base_pull_m'],
        hardware_commands_sent=False),indent=2))
    return qpos,payload,labels,default_q
