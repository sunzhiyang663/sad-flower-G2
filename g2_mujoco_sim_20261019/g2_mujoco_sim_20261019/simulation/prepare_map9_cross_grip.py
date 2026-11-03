"""Offline right partial drag -> left transverse wrap -> paired extraction/lift.

The original right Cartesian grasp is fixed. Arm transitions are sequential;
payload attachment is only a kinematic reference, not a load-bearing test.
"""
import json
import mujoco
import mink
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import yaml
from manipulation.wrist_pose import palm_mounts, pose7
from simulation.prepare_map9_preposition import support_patch


def reserve_joint_limits(checker, margin):
    for name in checker.names['left']+checker.names['right']+checker.names['waist'][:3]:
        low,high=checker.model.joint(name).range
        checker.narrow_limit(name,[low+margin,high-margin])


def wrap_contacts(checker, gap, depth):
    """Require a thumb and at least two fingers on opposing vertical sides."""
    half=checker.model.geom_size[checker.payload]
    best={s:None for s in ('thumb','index','middle','ring','pinky')}
    for g in checker.robot:
        name=checker.model.body(int(checker.model.geom_bodyid[g])).name
        if not name.startswith('hand_l_'):continue
        stem=next((s for s in best if '_'+s+'_' in name),None)
        if stem is None:continue
        segment=np.zeros(6)
        distance=float(mujoco.mj_geomDistance(checker.model,checker.data,checker.payload,g,.2,segment))
        point=checker.battery[:3,:3].T@(segment[3:]-checker.battery[:3,3])
        face=half[1] if stem=='thumb' else -half[1]
        # The thumb may oppose at the near bottom corner; fingers must curl
        # above the far bottom edge. Bottom-only finger contact is rejected.
        required_depth=0. if stem=='thumb' else depth
        error=max(max(0.,distance),abs(point[1]-face),max(0.,-half[2]+required_depth-point[2]))
        if best[stem] is None or error<best[stem]['error_m']:
            best[stem]=dict(geom=name,distance_m=distance,point_in_payload_m=point.tolist(),error_m=float(error))
    contacts=[s for s,v in best.items() if v and v['error_m']<=gap]
    return dict(passed='thumb' in contacts and len(set(contacts)-{'thumb'})>=2,
        contacts=contacts,geometry=best,contact_gap_m=gap,minimum_wrap_depth_m=depth,dynamics_validated=False)


def free_hand_park_goal(checker, position, contacts):
    """Numerical endpoint for a released hand; its orientation is free."""
    goal=checker.c.get_transform_frame_to_world('arm_l_end_link','body').as_matrix().copy()
    goal[:3,3]=checker.base_transform[:3,:3]@np.asarray(position)+checker.base_transform[:3,3]
    frame=mink.FrameTask('arm_l_end_link','body',position_cost=1.,orientation_cost=0.,lm_damping=.01)
    frame.set_target(mink.SE3.from_matrix(goal))
    names=checker.names['left'];moving={int(checker.model.joint(n).dofadr[0]) for n in names}
    freeze=mink.DofFreezingTask(checker.model,[n for n in range(checker.model.nv) if n not in moving])
    posture=mink.PostureTask(checker.model,cost=.001);posture.set_target_from_configuration(checker.c)
    limits=[mink.ConfigurationLimit(checker.model),mink.VelocityLimit(checker.model,{n:.3 for n in names}),checker.collision_limit]
    battery=checker.battery.copy();candidate=[];candidate_valid=True
    for step in range(1200):
        velocity=mink.solve_ik(checker.c,[frame,posture],.01,'daqp',damping=1e-5,constraints=[freeze],limits=limits)
        checker.c.integrate_inplace(velocity,.01);checker.set_battery(battery)
        try:checker.collisions(contacts)
        except ValueError:candidate_valid=False
        converged=np.linalg.norm(frame.compute_error(checker.c)[:3])<.0005
        if step%5==0 or converged:candidate.append(checker.current_joints())
        if converged:
            joints=checker.current_joints();checker.set_state(joints,battery);checker.collisions(contacts)
            return joints,candidate if candidate_valid else None
    raise ValueError('Released left-hand parking IK did not converge')


def prepare_pickup(checker, plan, inputs, replay, task, out, default_joints, cached_joint_route):
    profile = task['motion']['preposition']
    if profile.get('mirror_right_grasp'):raise ValueError('Cross-wrap task must preserve the original right grasp orientation')
    battery = np.array(inputs['camera_observation']['T_base_camera']) @ np.array(inputs['camera_observation']['T_camera_battery'])
    mount = palm_mounts(checker.cfg['urdf_path'])['left']
    palm = np.array(profile['left_palm_in_payload'])
    history = {h['id']: h for h in plan['checks']['history']}
    index = int(np.flatnonzero(replay['stages'] == 'right_grasp')[-1])
    checker.c.update(replay['qpos'][index])
    right = checker.c.get_transform_frame_to_world('arm_r_end_link', 'body').as_matrix().copy()
    # No mirrored goal: this is the original, accepted right grasp orientation.
    left = battery @ palm @ np.linalg.inv(mount)
    left[:3, 3] += profile['left_wait_offset_m']
    station = np.eye(4); station[:3, 3] = profile['station_offset_m']
    prebase = station.copy(); prebase[0, 3] -= profile['preapproach_backoff_m']
    home = default_joints(plan['config'], inputs['initial_joints'])
    waist = {n: home[n] for n in checker.names['waist']}
    waist.update(profile['waist_targets'])
    reserve_joint_limits(checker,profile['joint_limit_reserve_rad'])
    right_pre = profile.get('right_pre_motor_targets', history['right_open']['targets']['right'])
    seed = {**home, **waist, **profile['left_motor_targets'], **right_pre}
    seed.update(profile.get('endpoint_arm_seed', {}))
    checker.set_base(station)
    original_clearance=checker.cfg['collision_clearance_m']
    # The grasp station uses the contact clearance. Joint-path reserve applies
    # at the remote preposition station, where the hands do not touch the shell.
    failures=[];rng=np.random.default_rng(2512)
    checker.c.update(replay['qpos'][index]);source_seed=checker.current_joints()
    seeds=[seed.copy(),{**seed,**{n:source_seed[n] for n in checker.names['right']}},
        {**seed,**{n:home[n] for n in checker.names['right']}}]
    for attempt in range(16):
        candidate_seed=seed.copy()
        for n in checker.names['right']:
            lo,hi=checker.model.joint(n).range
            candidate_seed[n]=float(np.clip(source_seed[n]+rng.normal(0.,1.),lo+.01,hi-.01))
        seeds.append(candidate_seed)
    for candidate_seed in seeds:
        checker.set_base(station)
        checker.cfg['collision_clearance_m']=original_clearance
        for n in checker.names['left']+checker.names['right']:
            lo,hi=checker.model.joint(n).range;candidate_seed[n]=float(np.clip(candidate_seed[n],lo,hi))
        try:
            checker.set_state(candidate_seed,battery)
            checker.solve({'right': right, 'left': left}, ['right'], constructing=True)
            solved=checker.current_joints()
            for n in checker.names['left']+checker.names['right']:
                lo,hi=checker.model.joint(n).range
                if max(lo-solved[n],solved[n]-hi)>1e-5:raise ValueError('IK joint range violation: '+n)
                solved[n]=float(np.clip(solved[n],lo,hi))
            checker.set_state(solved,battery)
            checker.collisions(['right'])
            checker.set_base(prebase);checker.set_state(solved,battery)
            checker.cfg['collision_clearance_m']=original_clearance+task['motion']['joint_planning_reserve_m']
            checker.collisions([])
            checker.cfg['collision_clearance_m']=original_clearance
            checker.set_base(station);checker.set_state(solved,battery)
            break
        except ValueError as exc:
            failures.append(str(exc));print('Endpoint seed rejected:',len(failures),str(exc)[:200],flush=True)
    else:raise ValueError('No valid preposition endpoint: '+str(failures))
    checker.cfg['collision_clearance_m']=original_clearance
    endpoint = checker.current_joints()
    print('Preposition endpoint passed',flush=True)
    (out/'preposition_ik.yaml').write_text(yaml.safe_dump(dict(joints=endpoint, goals={'left': left.tolist(), 'right': right.tolist()}, T_station=station.tolist())))
    B = prebase.copy(); joints = home.copy(); attachment = None
    qpos = []; payload = []; bases = []; labels = []
    checker.set_base(B); checker.set_state(joints, battery)
    default_q = checker.data.qpos.copy()

    def save(label, contacts):
        checker.collisions(contacts)
        qpos.append(checker.data.qpos.copy()); payload.append(np.linalg.inv(B) @ checker.battery)
        bases.append(B.copy()); labels.append(label)

    save('default', [])
    checker.set_base(np.eye(4))
    # Arms do not affect the head extrinsic; the historical reference arms may
    # lie outside this candidate's additional planning-only joint reserve.
    reference={**inputs['initial_joints'],**{n:home[n] for n in checker.names['left']+checker.names['right']}}
    checker.set_state(reference, battery)
    head_mount = np.linalg.inv(checker.c.get_transform_frame_to_world('head_link3', 'body').as_matrix()) @ np.array(inputs['camera_observation']['T_base_camera'])
    checker.set_base(B); checker.set_state(joints, battery)
    camera = np.linalg.inv(B) @ checker.c.get_transform_frame_to_world('head_link3', 'body').as_matrix() @ head_mount
    (out/'assumed_camera_observation.yaml').write_text(yaml.safe_dump(dict(camera_frame='assumed_head_color', capture_timestamp_ns=None, T_base_camera=camera.tolist(), T_camera_battery=(np.linalg.inv(B @ camera) @ battery).tolist())))
    result = checker.request(dict(operation='waist_motion', T_world_base=B.tolist(), targets=waist, joints=joints, battery=battery.tolist()))
    for sample in result['path']:
        checker.set_state(sample['joints'], battery); save('lower_waist', [])
    joints = checker.current_joints()

    def fingers(label, targets, contacts):
        nonlocal joints
        before = joints.copy()
        for alpha in np.linspace(0, 1, 41)[1:]:
            joints = {**before, **{n: before[n] + alpha*(v-before[n]) for n, v in targets.items()}}
            checker.set_state(joints, checker.battery); save(label, contacts)

    fingers('prepare_hands', {**profile['left_motor_targets'], **right_pre}, [])
    # Plan each arm independently while freezing the other arm. A straight
    # Cartesian candidate is preferred; OMPL only shortens within collision limits.
    for side, frame, target in [('left', 'arm_l_end_link', left), ('right', 'arm_r_end_link', right)]:
        print('Planning preposition:',side,flush=True)
        target = target.copy(); target[:3, 3] += prebase[:3, 3] - station[:3, 3]
        start_joints = joints.copy()
        start = checker.c.get_transform_frame_to_world(frame, 'body').as_matrix().copy()
        rotation = Slerp([0, 1], Rotation.from_matrix(np.stack([start[:3, :3], target[:3, :3]])))
        count = max(1, int(np.ceil(max(np.linalg.norm(target[:3, 3]-start[:3, 3])/.005,
            Rotation.from_matrix(start[:3, :3].T @ target[:3, :3]).magnitude()/.025))))
        candidate = []
        original_minimum=checker.collision_limit.minimum_distance_from_collisions
        checker.collision_limit.minimum_distance_from_collisions+=profile['preposition_ik_extra_clearance_m']
        try:
            for alpha in np.linspace(0, 1, count+1)[1:]:
                goal = target.copy(); goal[:3, 3] = start[:3, 3] + alpha*(target[:3, 3]-start[:3, 3]); goal[:3, :3] = rotation(alpha).as_matrix()
                checker.solve({side: goal}, []); candidate.append(checker.current_joints())
            goal_joints = checker.current_joints()
        except ValueError as exc:
            print('Cartesian preposition candidate rejected:', side, exc, flush=True)
            checker.set_state({**start_joints, **{n: endpoint[n] for n in checker.names[side]}}, battery)
            checker.solve({side: target}, [], constructing=True)
            goal_joints = checker.current_joints(); candidate = None
        finally:
            checker.collision_limit.minimum_distance_from_collisions=original_minimum
        checker.set_state(start_joints, battery)
        arc = dict(frame=frame, tolerance_rad=.20, positions=[dict(frame=frame, tolerance_m=.05)])
        samples = cached_joint_route(out/f'default_to_preposition_{side}.yaml', checker, start_joints, goal_joints,
            checker.names[side], battery, task['motion'], wrist_arc=arc, candidate=candidate)
        for sample in samples:
            checker.set_state(sample, battery); save('arm_plan', [])
        joints = checker.current_joints()
    for alpha in np.linspace(0, 1, int(np.ceil(profile['preapproach_backoff_m']/checker.cfg['path_step_m']))+1)[1:]:
        B = prebase.copy(); B[:3, 3] = prebase[:3, 3] + alpha*(station[:3, 3]-prebase[:3, 3])
        checker.set_base(B); checker.set_state(joints, battery); save('base_approach', ['right'])
    fingers('right_grasp', history['right_grasp']['targets']['right'], ['right'])
    attachment = np.linalg.inv(checker.c.get_transform_frame_to_world('arm_r_end_link', 'body').as_matrix()) @ battery

    def retreat(label, distance, contacts):
        nonlocal B
        goal = B.copy(); goal[0, 3] -= distance
        result = checker.request(dict(operation='base_motion', T_world_base=B.tolist(), goal_base=goal.tolist(), joints=joints,
            battery=checker.battery.tolist(), contacts=contacts, payload_from_right=attachment.tolist()))
        for sample in result['path']:
            B = np.array(sample['T_world_base']); checker.set_base(B); checker.set_state(sample['joints'], sample['battery']); save(label, contacts)

    def move(label, goals, contacts):
        nonlocal joints
        allow_waist=label in profile.get('coordinated_waist_stages',[])
        if allow_waist and 'right' not in goals:
            goals={**goals,'right':checker.c.get_transform_frame_to_world('arm_r_end_link','body').as_matrix().copy()}
        ends = {s: checker.c.get_transform_frame_to_world(f'arm_{"l" if s == "left" else "r"}_end_link', 'body').as_matrix() for s in goals}
        result = checker.request(dict(operation='motion', T_world_base=B.tolist(), joints=joints, battery=checker.battery.tolist(),
            contacts=contacts, allow_waist=allow_waist, ends={s: pose7(T) for s, T in ends.items()}, goals={s: T.tolist() for s, T in goals.items()}, payload_from_right=attachment.tolist()))
        for sample in result['path']:
            checker.set_state(sample['joints'], checker.battery)
            checker.set_battery(checker.c.get_transform_frame_to_world('arm_r_end_link', 'body').as_matrix() @ attachment)
            save(label, contacts)
        joints = checker.current_joints(); print(label, len(result['path']), flush=True)

    retreat('right_partial_drag', profile['right_partial_drag_m'], ['right'])
    target = checker.battery @ palm @ np.linalg.inv(mount); target[2, 3] -= profile['left_entry_gap_m']
    down = checker.c.get_transform_frame_to_world('arm_l_end_link', 'body').as_matrix().copy(); down[2, 3] = target[2, 3]
    move('left_lower_entry', {'left': down}, ['right']); move('left_entry', {'left': target}, ['right'])
    target[2, 3] += profile['left_entry_gap_m']
    move('left_seat', {'left': target}, ['left', 'right'])
    fingers('left_grasp', profile['left_closed_motor_targets'], ['left', 'right'])
    wrap=wrap_contacts(checker,profile['wrap_contact_gap_m'],profile['wrap_depth_m'])
    (out/'wrap_contact_audit.json').write_text(json.dumps(wrap,indent=2))
    if not wrap['passed']:raise ValueError('Left fingers do not wrap opposing shell sides: '+str(wrap))
    audit = support_patch(checker, profile['support']); (out/'support_audit.json').write_text(json.dumps(audit, indent=2))
    # A wrapped grasp transmits side forces and moments. The old flat-palm COM
    # polygon is retained as a diagnostic, not used as a force-closure proof.
    retreat('paired_full_drag', profile['base_pull_m']-profile['right_partial_drag_m'], ['left', 'right'])
    goals = {s: checker.c.get_transform_frame_to_world(f'arm_{l}_end_link', 'body').as_matrix().copy() for s, l in [('left', 'l'), ('right', 'r')]}
    for T in goals.values(): T[2, 3] += profile['lift_m']
    move('lift', goals, ['left', 'right'])
    np.savez_compressed(out/'pickup.npz', qpos=qpos, payload=payload, base_offsets=bases, labels=labels, default_qpos=default_q)
    (out/'pickup_profile.json').write_text(json.dumps(dict(mode='right_partial_drag_left_wrap_paired_lift',
        preposition_before_base_approach=True, stationary_waist_during_extraction=not bool(profile.get('coordinated_waist_stages')), station=station.tolist(), carry_base=B.tolist(),
        support=audit, T_right_end_battery=attachment.tolist(), right_orientation_preserved=True,
        right_partial_drag_m=profile['right_partial_drag_m'], base_pull_m=profile['base_pull_m'], hardware_commands_sent=False), indent=2))
    return qpos, payload, labels, default_q
