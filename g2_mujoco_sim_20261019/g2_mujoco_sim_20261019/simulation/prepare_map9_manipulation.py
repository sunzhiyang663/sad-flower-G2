"""Default posture -> lowered waist -> validated extraction -> table placement.

Reuses the delivered URDF/SRDF checker, OMPL joint planning and Mink wrist IK.
The shell is kinematically supported; this is not a friction/dynamics test.
"""
import json
import hashlib
from pathlib import Path
import sys
import time
from copy import deepcopy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import mujoco
import numpy as np
from ompl import base as ob, geometric as og, util as ou
from scipy.spatial.transform import Rotation
import yaml
from simulation.full_task_config import load_task
from simulation.build_map9_scene import path,pose
from simulation.pose_grasp_check import Checker
from manipulation.wrist_pose import pose7


def default_joints(config, initial):
    joints = dict(initial)
    poses = yaml.safe_load((ROOT/'config/manipulation.yaml').read_text())['manipulation']['poses']
    for item in poses['default_pose'].values():
        if isinstance(item,dict) and 'joint_names' in item:joints.update(zip(item['joint_names'],item['joints']))
    for side in ('left','right'):
        hand=config['grasp']['hands'][side];joints.update(zip(hand['joint_names'],hand['home']))
    return joints


class JointMotionObjective(ob.OptimizationObjective):
    """Sum of absolute joint travel, in radians (not Cartesian distance)."""
    def __init__(self, si, dimension):
        super().__init__(si)
        self.dimension = dimension

    def stateCost(self, state):
        return ob.Cost(0.)

    def motionCost(self, a, b):
        return ob.Cost(sum(abs(a[k]-b[k]) for k in range(self.dimension)))


def joint_route(checker, start, goal, names, battery, cfg, contacts=(), seed=None, wrist_arc=None):
    """OMPL collision-constrained shortening with an explicit joint-travel cost."""
    checker.set_state(start,battery)
    original_clearance=checker.cfg['collision_clearance_m']
    checker.cfg['collision_clearance_m']=original_clearance+cfg['joint_planning_reserve_m']
    space=ob.RealVectorStateSpace(len(names));bounds=ob.RealVectorBounds(len(names))
    for k,n in enumerate(names):
        lo,hi=checker.model.joint(n).range;bounds.setLow(k,lo);bounds.setHigh(k,hi)
    space.setBounds(bounds);setup=og.SimpleSetup(space)
    indices=np.array([checker.model.joint(n).qposadr[0] for n in names]);q=checker.data.qpos.copy()
    if wrist_arc:
        frame=checker.model.body(wrist_arc['frame']).id
        R0=checker.data.xmat[frame].reshape(3,3).copy()
        checker.set_state(goal,battery);R1=checker.data.xmat[frame].reshape(3,3).copy()
        vector=Rotation.from_matrix(R0.T@R1).as_rotvec();angle=np.linalg.norm(vector)
        axis=vector/max(angle,1e-12)
        # Near 180 degrees, the opposite rotation differs by only a few degrees
        # and can avoid a distant joint branch. Choose its sign from the initial
        # direct joint direction; it still has an explicit bounded SO(3) corridor.
        half_turn=wrist_arc.get('near_half_turn_tolerance_rad',0.)
        if half_turn and angle>np.pi-half_turn:
            probe=q.copy();probe[indices]=[start[n]+.01*(goal[n]-start[n]) for n in names]
            checker.data.qpos[:]=probe;mujoco.mj_forward(checker.model,checker.data)
            direction=Rotation.from_matrix(R0.T@checker.data.xmat[frame].reshape(3,3)).as_rotvec()@axis
            if wrist_arc.get('near_half_turn_direction')=='opposite' or direction<0:
                axis=-axis;angle=2*np.pi-angle
    corridors=[]
    for item in wrist_arc.get('positions',[]) if wrist_arc else []:
        body=checker.model.body(item['frame']).id
        checker.set_state(start,battery);a=checker.data.xpos[body].copy()
        checker.set_state(goal,battery);b=checker.data.xpos[body].copy()
        corridors.append((body,a,b,item['tolerance_m']))
    calls=0
    def valid(s):
        nonlocal calls
        checker.data.qpos[:]=q;checker.data.qpos[indices]=[s[k] for k in range(len(names))]
        mujoco.mj_forward(checker.model,checker.data)
        calls+=1
        for body,a,b,tolerance in corridors:
            direction=b-a
            fraction=np.clip((checker.data.xpos[body]-a)@direction/max(direction@direction,1e-12),0,1)
            if np.linalg.norm(checker.data.xpos[body]-(a+fraction*direction))>tolerance:return False
        if wrist_arc:
            relative=R0.T@checker.data.xmat[frame].reshape(3,3)
            along=float(Rotation.from_matrix(relative).as_rotvec()@axis)
            if angle>np.pi and along < -np.pi/2:along+=2*np.pi
            if along < -.005 or along > angle+.005:return False
            expected=Rotation.from_rotvec(np.clip(along,0,angle)*axis).as_matrix()
            if Rotation.from_matrix(expected.T@relative).magnitude()>wrist_arc['tolerance_rad']:return False
        try:checker.collisions(contacts);return True
        except ValueError:return False
    setup.setStateValidityChecker(valid)
    setup.getSpaceInformation().setStateValidityCheckingResolution(cfg.get('joint_validity_step_rad',cfg['joint_step_rad'])/space.getMaximumExtent())
    states=[]
    for values in (start,goal):
        s=space.allocState()
        for k,n in enumerate(names):s[k]=values[n]
        states.append(s)
    for s in states:
        if not valid(s):
            checker.collisions(contacts)
            raise ValueError('Joint endpoint violates wrist corridor')
    si=setup.getSpaceInformation()
    objective=JointMotionObjective(si,len(names));setup.setOptimizationObjective(objective)
    setup.setStartAndGoalStates(*states)
    si.setup()
    def path_from_samples(samples):
        result=og.PathGeometric(si);result.append(states[0])
        for sample in samples:
            s=space.allocState()
            for k,n in enumerate(names):s[k]=sample[n]
            result.append(s)
        return result
    solution=path_from_samples(seed) if seed else None
    if solution is not None and not solution.check():solution=None
    if solution is None:
        planner=og.RRTConnect(si);planner.setRange(.4);setup.setPlanner(planner)
        setup.solve(cfg['joint_planning_time_s'])
        if not setup.haveExactSolutionPath():raise RuntimeError('No collision-free joint transition')
        solution=og.PathGeometric(setup.getSolutionPath())
    before=og.PathGeometric(solution)
    if si.checkMotion(*states):
        solution=og.PathGeometric(si,states[0],states[1])
    else:
        simplifier=og.PathSimplifier(si,None,objective)
        # Standard OMPL optimizers: shortcuts may only replace a segment when
        # joint travel decreases and every interpolated state remains valid.
        print('Shortening joint path with collision constraints',flush=True)
        simplifier.reduceVertices(solution,60,10,1.)
        deadline=time.monotonic()+cfg['joint_optimization_time_s']
        for attempt in range(0,cfg['joint_shortcut_attempts'],20):
            simplifier.partialShortcutPath(solution,20,10,1.,.005)
            if time.monotonic()>=deadline:break
    optimization_rejected=not solution.check()
    if optimization_rejected:
        if not before.check():raise RuntimeError('Both candidate and optimized joint paths failed recheck')
        print('Shortcut failed recheck; retaining the validated candidate',flush=True)
        solution=before
    if solution.cost(objective).value()>before.cost(objective).value()+1e-8:
        solution=before
    def metrics(p):
        values=np.array([[p.getState(k)[j] for j in range(len(names))] for k in range(p.getStateCount())])
        travel=np.abs(np.diff(values,axis=0)).sum(axis=0)
        return dict(total_joint_travel_rad=float(travel.sum()),
            joint_space_length_rad=float(p.length()),
            per_joint_travel_rad=dict(zip(names,travel.tolist())),
            per_joint_excursion_rad=dict(zip(names,np.ptp(values,axis=0).tolist())))
    report=dict(objective='sum(abs(delta_q))',method='OMPL reduceVertices + partialShortcutPath',
        before=metrics(before),after=metrics(solution),
        fixed_endpoint_lower_bound_rad=float(sum(abs(goal[n]-start[n]) for n in names)),
        state_check_step_rad=cfg['joint_step_rad'],planning_clearance_m=checker.cfg['collision_clearance_m'],
        global_optimum_proven=False)
    report['optimization_rejected_on_recheck']=optimization_rejected
    report['wrist_short_arc']=wrist_arc
    if wrist_arc:report['chosen_wrist_corridor_sweep_rad']=float(angle)
    samples=[]
    for k in range(solution.getStateCount()-1):
        a=np.array([solution.getState(k)[j] for j in range(len(names))])
        b=np.array([solution.getState(k+1)[j] for j in range(len(names))])
        for t in np.linspace(0,1,max(2,int(np.ceil(np.max(np.abs(b-a))/cfg['joint_step_rad']))+1))[1:]:
            sample={n:float(v) for n,v in {**start,**dict(zip(names,a+t*(b-a)))}.items()}
            checker.set_state(sample,battery);checker.collisions(contacts);samples.append(sample)
    checker.cfg['collision_clearance_m']=original_clearance
    report['collision_checks']=calls
    print('Joint optimization:',report['before']['total_joint_travel_rad'],'->',report['after']['total_joint_travel_rad'],
        'rad;',len(samples),'samples,',calls,'checks',flush=True)
    return samples,report


def cached_joint_route(cache,checker,start,goal,names,battery,cfg,contacts=(),wrist_arc=None,candidate=None):
    samples=yaml.safe_load(cache.read_text()) if cache.exists() else candidate
    if samples and (max(abs(samples[0][n]-v) for n,v in start.items())>cfg['joint_step_rad']+1e-7
                    or max(abs(samples[-1][n]-v) for n,v in goal.items())>1e-7):
        samples=None
    if samples is None:samples=candidate
    report_file=cache.with_suffix('.optimization.json')
    policy=dict(version=4,base=checker.base_transform.tolist(),
        start=start,goal=goal,names=names,battery=np.asarray(battery).tolist(),
        joint_ranges={n:checker.model.joint(n).range.tolist() for n in names},
        step=cfg['joint_step_rad'],reserve=cfg['joint_planning_reserve_m'],
        seconds=cfg['joint_optimization_time_s'],attempts=cfg['joint_shortcut_attempts'],wrist_arc=wrist_arc,
        scene_sha256=hashlib.sha256(json.dumps(checker.scene,sort_keys=True).encode()).hexdigest())
    if 'joint_validity_step_rad' in cfg:policy['joint_validity_step_rad']=cfg['joint_validity_step_rad']
    report=json.loads(report_file.read_text()) if report_file.exists() else {}
    prefer_candidate=False
    if candidate and samples is not candidate:
        def travel(records):
            values=np.array([[r[n] for n in names] for r in [start]+list(records)])
            return float(np.abs(np.diff(values,axis=0)).sum())
        if not samples or travel(candidate)<travel(samples)-1e-8:
            samples=candidate;prefer_candidate=True
    if not samples or report.get('policy')!=policy or prefer_candidate:
        original_clearance=checker.cfg['collision_clearance_m']
        try:
            samples,report=joint_route(checker,start,goal,names,battery,cfg,contacts,seed=samples,wrist_arc=wrist_arc)
        finally:
            checker.cfg['collision_clearance_m']=original_clearance
        report['policy']=policy
        cache.write_text(yaml.safe_dump(samples))
        report_file.write_text(json.dumps(report,indent=2))
    return samples


def retarget_extraction(checker, plan, inputs, replay, branch):
    """Keep accepted wrist goals; choose another arm/waist IK branch offline."""
    reference=Checker(plan['config'],inputs['scene'])
    first=int(np.flatnonzero(replay['stages']=='left_open')[0]);frames=[]
    for k in range(first,len(replay['qpos'])):
        reference.c.update(replay['qpos'][k])
        goals={s:reference.c.get_transform_frame_to_world(f'arm_{l}_end_link','body').as_matrix().copy()
               for s,l in [('left','l'),('right','r')]}
        T=np.eye(4);T[:3,3]=replay['mocap_pos'][k,0]
        T[:3,:3]=Rotation.from_quat(replay['mocap_quat'][k,0],scalar_first=True).as_matrix()
        frames.append((str(replay['stages'][k]),goals,T,reference.current_joints()))
    joints={**inputs['initial_joints'],**inputs['waist_target_joints']}
    joints[checker.names['waist'][3]]=branch['waist_roll_seed_rad']
    joints[checker.names['right'][4]]=branch['right_joint5_seed_rad']
    checker.set_state(joints,frames[0][2]);checker.solve(frames[0][1],[],constructing=True)
    seed=checker.current_joints();joints=seed.copy();samples=[]
    attachment=np.array(next(h for h in plan['checks']['history'] if h['id']=='right_grasp')['T_right_end_battery'])
    for label,goals,T,source_joints in frames:
        joints.update({n:source_joints[n] for names in checker.hand_names.values() for n in names})
        contacts=['right'] if label in ('right_approach','right_grasp','partial_pull','verify_pose','left_seat','left_grasp','full_pull','lift') else []
        if label in ('left_seat','left_grasp','full_pull','lift'):contacts.append('left')
        attached=label in ('partial_pull','verify_pose','left_seat','left_grasp','full_pull','lift')
        checker.set_state(joints,checker.battery if attached else T)
        ends={s:checker.c.get_transform_frame_to_world(f'arm_{l}_end_link','body').as_matrix().copy()
              for s,l in [('left','l'),('right','r')]}
        result=checker.request(dict(operation='motion',joints=joints,battery=checker.battery.tolist(),
            contacts=contacts,ends={s:pose7(v) for s,v in ends.items()},goals={s:v.tolist() for s,v in goals.items()},
            payload_from_right=attachment.tolist() if attached else None,allow_waist=True))
        for sample in result['path']:
            checker.set_state(sample['joints'],T)
            if attached:checker.set_battery(checker.c.get_transform_frame_to_world('arm_r_end_link','body').as_matrix()@attachment)
            checker.collisions(contacts)
            samples.append((checker.data.qpos.copy(),checker.battery.copy(),label))
        joints=checker.current_joints()
    return seed,samples


def main():
    cfg=yaml.safe_load((ROOT/'simulation/map9.yaml').read_text());task=load_task()
    out=path(task['output']);out.mkdir(parents=True,exist_ok=True)
    (out/'validation.json').write_text(json.dumps(dict(passed=False,reason='Manipulation changed; full sequence must be revalidated')))
    source=path(cfg['carry_reference']);plan=json.loads((source/'source_plan.json').read_text())
    inputs=yaml.safe_load((source/'source_inputs.yaml').read_text())['inputs'];replay=np.load(source/'replay.npz')
    branch=task['motion'].get('wrist_branch')
    checker_config=deepcopy(plan['config'])
    if branch:checker_config['pose_grasp'].update(branch['ik'])
    preposition=task['motion'].get('preposition')
    if preposition:checker_config['pose_grasp'].update(preposition['ik'])
    if task.get('spread_source'):
        checker_config['pose_grasp']['contact_faces']['right']=['-x','-y','+y','+z']
    checker=Checker(checker_config,inputs['scene']);checker.scene=inputs['scene']
    if preposition:
        if task.get('spread_source'):
            from simulation.prepare_map9_spread import prepare_pickup
        elif preposition.get('mode') == 'cross_wrap':
            from simulation.prepare_map9_cross_grip import prepare_pickup
        else:
            from simulation.prepare_map9_preposition import prepare_pickup
        qpos,payload,labels,default_q=prepare_pickup(checker,plan,inputs,replay,task,out,default_joints,cached_joint_route)
    else:
        seed,extraction=retarget_extraction(checker,plan,inputs,replay,branch) if branch else (inputs['initial_joints'],None)
        battery=np.array(inputs['camera_observation']['T_base_camera'])@np.array(inputs['camera_observation']['T_camera_battery'])
        joints=default_joints(plan['config'],inputs['initial_joints'])
        qpos=[];payload=[];labels=[]
        def save(label):
            qpos.append(checker.data.qpos.copy());payload.append(checker.battery.copy());labels.append(label)
        checker.set_state(joints,battery);checker.collisions([]);save('default')
        default_q=checker.data.qpos.copy()
        # Capture-time head pose for the assumed offline observation. Camera mount
        # relative to head_link3 is inherited from the original assumed observation.
        checker.set_state(inputs['initial_joints'],battery)
        T_head_camera=np.linalg.inv(checker.c.get_transform_frame_to_world('head_link3','body').as_matrix())@np.array(inputs['camera_observation']['T_base_camera'])
        checker.set_state(joints,battery)
        T_base_camera=checker.c.get_transform_frame_to_world('head_link3','body').as_matrix()@T_head_camera
        offline_observation=dict(camera_frame='assumed_head_color',capture_timestamp_ns=None,
            T_base_camera=T_base_camera.tolist(),T_camera_battery=(np.linalg.inv(T_base_camera)@battery).tolist())
        (out/'assumed_camera_observation.yaml').write_text(yaml.safe_dump(offline_observation,sort_keys=False))
        print('Waist adjustment',flush=True)
        waist_targets={n:seed[n] for n in checker.names['waist']} if branch else inputs['waist_target_joints']
        result=checker.request(dict(operation='waist_motion',targets=waist_targets,joints=joints,battery=battery.tolist()))
        for sample in result['path']:
            checker.set_state(sample['joints'],battery);save('lower_waist')
        joints=result['path'][-1]['joints']
        # Align only the fingers with the accepted seed before planning both arms.
        target={n:inputs['initial_joints'][n] for names in checker.hand_names.values() for n in names}
        before=dict(joints)
        for t in np.linspace(0,1,31)[1:]:
            joints={**before,**{n:before[n]+t*(v-before[n]) for n,v in target.items()}}
            checker.set_state(joints,battery);checker.collisions([]);save('prepare_hands')
        groups=[('right',checker.names['right']),('left',checker.names['left'])] if branch else [('both',checker.names['left']+checker.names['right'])]
        for side,names in groups:
            target={**joints,**{n:seed[n] for n in names}}
            cache=out/(f'default_to_grasp_{side}_joints.yaml' if branch else 'default_to_grasp_joints.yaml')
            samples=cached_joint_route(cache,checker,joints,target,names,battery,task['motion'],
                wrist_arc=branch['arc'] if branch and side=='right' else None)
            for sample in samples:
                checker.set_state(sample,battery);checker.collisions([]);save('arm_plan')
            joints=checker.current_joints()
        if extraction:
            for q,T,label in extraction:checker.c.update(q);checker.set_battery(T);save(label)
        else:
            start=int(np.flatnonzero(replay['stages']=='left_open')[0])
            for k in range(start,len(replay['qpos'])):
                checker.c.update(replay['qpos'][k]);T=np.eye(4);T[:3,3]=replay['mocap_pos'][k,0]
                T[:3,:3]=Rotation.from_quat(replay['mocap_quat'][k,0],scalar_first=True).as_matrix()
                checker.set_battery(T);save(str(replay['stages'][k]))
        np.savez_compressed(out/'pickup.npz',qpos=qpos,payload=payload,labels=labels,default_qpos=default_q)
        print('Pickup saved',len(labels),flush=True)

    # Table coordinates are relative to the stationary goal base, z minus the
    # 5 cm base origin height. Same geometry at G1 and G5.
    scene=dict(support=dict(T_base=np.eye(4).tolist(),size_m=[cfg['table']['size_m'][1],cfg['table']['size_m'][0],cfg['table']['top_thickness_m']]),opening=None,obstacles=[])
    scene['support']['T_base'][0][3]=cfg['table']['centre_forward_m']
    top=cfg['table']['size_m'][2]-cfg['base_height_m']
    scene['support']['T_base'][2][3]=top-cfg['table']['top_thickness_m']/2
    # Use the same legs as the full scene, so the left-arm joint planner cannot
    # shortcut through a leg hidden beneath the tabletop.
    full_model=mujoco.MjModel.from_binary_path(str(path(cfg['output'])/'scene.mjb'))
    full_data=mujoco.MjData(full_model);mujoco.mj_forward(full_model,full_data)
    meta=json.loads((path(cfg['snapshot'])/'map.json').read_text())
    table_guide=cfg['table']['guide_ids'][0]
    table_base=pose(next(p for p in meta['guide_points'] if p['id']==table_guide),cfg['base_height_m'])
    for k in range(4):
        geom=full_model.geom(f'table_leg_G{table_guide}_{k}').id
        T=np.eye(4);T[:3,3]=full_data.geom_xpos[geom];T[:3,:3]=full_data.geom_xmat[geom].reshape(3,3)
        scene['obstacles'].append(dict(name=f'table_leg_{k}',T_base=(np.linalg.inv(table_base)@T).tolist(),
            size_m=(2*full_model.geom_size[geom]).tolist()))
    carry_q=qpos[-1].copy()
    checker=Checker(checker_config,scene);checker.scene=scene
    if preposition and preposition.get('mode')=='cross_wrap':
        from simulation.prepare_map9_cross_grip import reserve_joint_limits
        reserve_joint_limits(checker,preposition['joint_limit_reserve_rad'])
    checker.cfg['ik_solve_position_tolerance_m']=task['motion']['placement_solve_position_tolerance_m']
    # Leave the joint planner's reserve around the torso while lowering.
    checker.collision_limit.minimum_distance_from_collisions += task['motion']['joint_planning_reserve_m']
    placement_base=np.eye(4)
    if preposition and preposition.get('mode')=='cross_wrap':
        placement_base[0,3]=-task['motion']['table_station_backoff_m']
    initial_placement_base=placement_base.copy()
    checker.set_base(placement_base)
    checker.c.update(carry_q);checker.set_battery(placement_base@payload[-1]);joints=checker.current_joints()
    checker.set_state(joints,placement_base@payload[-1]);checker.collisions(['left','right'])
    qpos=[];payload=[];labels=[];checks=[];placement_bases=[]
    def save(label):
        qpos.append(checker.data.qpos.copy());payload.append(np.linalg.inv(placement_base)@checker.battery);labels.append(label);placement_bases.append(placement_base.copy())
    right=checker.c.get_transform_frame_to_world('arm_r_end_link','body').as_matrix()
    attachment=np.linalg.inv(right)@checker.battery
    def move(label,delta,sides,contacts,attached=True):
        nonlocal joints
        ends={s:checker.c.get_transform_frame_to_world(f'arm_{"l" if s=="left" else "r"}_end_link','body').as_matrix() for s in sides}
        goals={s:T.copy() for s,T in ends.items()}
        for s,T in goals.items():T[:3,3]+=delta[s] if isinstance(delta,dict) else delta
        print(label,flush=True)
        try:
            if task.get('spread_source'):
                from simulation.spread_cartesian import translate
                signature=hashlib.sha256(json.dumps(dict(joints=joints,battery=checker.battery.tolist(),base=placement_base.tolist(),
                    goals={s:T.tolist() for s,T in goals.items()},contacts=contacts,attachment=attachment.tolist() if attached else None,
                    scene=scene,config=checker_config,solver_sha256=hashlib.sha256((ROOT/'simulation/spread_cartesian.py').read_bytes()).hexdigest()),sort_keys=True).encode()).hexdigest()
                cache=out/(label+'_cartesian.json');cached=json.loads(cache.read_text()) if cache.exists() else {}
                if cached.get('signature')==signature:result=cached['result']
                else:
                    result=translate(checker,joints,checker.battery.copy(),placement_base,goals,contacts,attachment if attached else None)
                    cache.write_text(json.dumps(dict(signature=signature,result=result),indent=2))
            else:
                result=checker.request(dict(operation='motion',T_world_base=placement_base.tolist(),joints=joints,battery=checker.battery.tolist(),
                contacts=contacts,ends={s:pose7(T) for s,T in ends.items()},goals={s:T.tolist() for s,T in goals.items()},
                allow_waist=task['motion']['placement_allow_waist'] and label in ('table_hover','place_on_table','right_withdraw','paired_table_hover','front_land'),
                payload_from_right=attachment.tolist() if attached else None))
        except ValueError:
            near=[]
            for a,b in checker.collision_pairs:
                distance=mujoco.mj_geomDistance(checker.model,checker.data,a,b,.02,None)
                if distance<.02:
                    near.append(dict(a=checker.model.body(checker.model.geom_bodyid[a]).name,
                        b=checker.model.body(checker.model.geom_bodyid[b]).name,distance=float(distance)))
            near.sort(key=lambda p:p['distance'])
            (out/'placement_failure.json').write_text(json.dumps(dict(stage=label,near=near,
                joints=checker.current_joints(),battery=checker.battery.tolist()),indent=2))
            print('Nearest pairs',near[:20],flush=True)
            raise
        fixed_battery=checker.battery.copy()
        for sample in result['path']:
            checker.set_state(sample['joints'],fixed_battery)
            if attached:
                end=checker.c.get_transform_frame_to_world('arm_r_end_link','body').as_matrix()
                checker.set_battery(end@attachment)
            checker.collisions(contacts)
            save(label)
        joints=checker.current_joints();checks.append(dict(stage=label,samples=len(result['path'])))
    def fingers(label,side,contacts):
        nonlocal joints
        stage=next(s for s in plan['checks']['history'] if s['id']==side+'_open')
        targets=preposition['left_motor_targets'] if preposition and side=='left' else stage['targets'][side]
        if preposition and preposition.get('mode')=='cross_wrap' and label=='right_release':
            targets=preposition['right_pre_motor_targets']
        before=dict(joints)
        for t in np.linspace(0,1,31)[1:]:
            joints={**before,**{n:before[n]+t*(v-before[n]) for n,v in targets.items()}}
            checker.set_state(joints,checker.battery);checker.collisions(contacts);save(label)
    half=cfg['payload']['size_m'][0]/2
    x=cfg['table']['centre_forward_m']-cfg['table']['size_m'][1]/2+half+task['motion']['placement_edge_gap_m']
    hover=top+cfg['payload']['size_m'][2]/2+task['motion']['placement_hover_m']
    cross_wrap = preposition and preposition.get('mode') == 'cross_wrap'
    if cross_wrap:
        # Both hands keep the same rigid shell transform until its free front
        # end contacts the table. Only then may the support fingers open.
        front=cfg['table']['centre_forward_m']-cfg['table']['size_m'][1]/2
        landing_x=front-half+task['motion']['landing_table_overlap_m']
        delta=np.array([task['motion'].get('lowering_forward_m',0.) if task.get('spread_source') else landing_x-checker.battery[0,3],task['motion'].get('lowering_left_m',0.),hover-checker.battery[2,3]])
        move('paired_table_hover',delta,['left','right'],['left','right'])
        if task.get('spread_source'):
            goal_base=placement_base.copy();goal_base[0,3]+=landing_x-checker.battery[0,3]
            result=checker.request(dict(operation='base_push',T_world_base=placement_base.tolist(),goal_base=goal_base.tolist(),
                joints=joints,battery=checker.battery.tolist(),contacts=['left','right'],payload_from_right=attachment.tolist()))
            for sample in result['path']:
                placement_base=np.array(sample['T_world_base']);checker.set_base(placement_base)
                checker.set_state(sample['joints'],sample['battery']);save('paired_table_approach')
            joints=checker.current_joints()
        move('front_land',np.array([0.,0.,top+cfg['payload']['size_m'][2]/2-checker.battery[2,3]]),['left','right'],['left','right'])
        landing=checker.battery.copy()
        (out/'front_landing.json').write_text(json.dumps(dict(T_base_payload=landing.tolist(),
            overlap_m=task['motion']['landing_table_overlap_m'],both_hands_before_contact=True,
            left_released_after_contact=True,dynamics_validated=False),indent=2))
    fingers('left_release','left',['left','right'])
    move('left_drop',np.array([0,0,-task['motion']['support_hand_drop_m']]),['left'],['left','right'])
    move('left_withdraw',np.array([-task['motion']['hand_withdraw_m'],task['motion']['support_hand_lateral_withdraw_m'],0]),['left'],['right'])
    if cross_wrap:
        (out/'placement_retract_state.yaml').write_text(yaml.safe_dump(dict(joints=joints,battery=checker.battery.tolist(),base=placement_base.tolist(),scene=scene)))
        # The lowered torso makes the recorded standing arm posture unsuitable.
        # Retract the free hand beside the torso and above the chassis instead.
        from simulation.prepare_map9_cross_grip import free_hand_park_goal
        park,parking_candidate=free_hand_park_goal(checker,task['motion']['support_hand_park_position_in_base_m'],['right'])
        checker.set_state(joints,checker.battery)
        samples=cached_joint_route(out/'left_park_joints.yaml',checker,joints,park,checker.names['left'],checker.battery.copy(),task['motion'],['right'],candidate=parking_candidate)
        for sample in samples:
            checker.set_state(sample,checker.battery);checker.collisions(['right']);save('left_park')
        joints=checker.current_joints()
    else:
        park={**joints,**{n:v for n,v in default_joints(plan['config'],joints).items() if n in checker.names['left']}}
        cache=out/'left_park_joints.yaml'
        samples=cached_joint_route(cache,checker,joints,park,checker.names['left'],checker.battery.copy(),task['motion'],['right'])
        for sample in samples:
            checker.set_state(sample,checker.battery);checker.collisions(['right']);save('left_park')
        joints=checker.current_joints()
    if not cross_wrap:
        move('table_hover',np.array([x-checker.battery[0,3],0,hover-checker.battery[2,3]]),['right'],['right'])
    final_z=top+cfg['payload']['size_m'][2]/2
    if cross_wrap and task['motion'].get('placement_push_with_base'):
        if task.get('spread_source'):
            move('right_arm_push',np.array(task['motion']['right_arm_push_delta_m']),['right'],['right'])
        goal_base=placement_base.copy();goal_base[0,3]+=x-checker.battery[0,3]
        result=checker.request(dict(operation='base_push',T_world_base=placement_base.tolist(),goal_base=goal_base.tolist(),
            joints=joints,battery=checker.battery.tolist(),contacts=['right'],payload_from_right=attachment.tolist()))
        for sample in result['path']:
            placement_base=np.array(sample['T_world_base']);checker.set_base(placement_base)
            checker.set_state(sample['joints'],sample['battery']);save('right_push_on_table')
        joints=checker.current_joints();checks.append(dict(stage='right_push_on_table',samples=len(result['path']),base_translation_m=float(goal_base[0,3]-initial_placement_base[0,3])))
    else:
        move('right_push_on_table' if cross_wrap else 'place_on_table',
            np.array([x-checker.battery[0,3] if cross_wrap else 0,0,final_z-checker.battery[2,3]]),['right'],['right'])
    bottom=float(checker.battery[2,3]-cfg['payload']['size_m'][2]/2+cfg['base_height_m'])
    resting=checker.battery.copy()
    fingers('right_release','right',['right'])
    move('right_withdraw',np.array([-task['motion']['grasp_hand_withdraw_m'],0,0]),['right'],['right'],attached=False)
    if cross_wrap and not task.get('spread_source'):fingers('right_open_clear','right',[])
    if not np.allclose(checker.battery,resting):raise RuntimeError('Released payload moved with wrist')
    checker.collisions([])
    if task.get('spread_source'):
        # Payload stays on the table during retreat and whole-body reset.
        start_base=placement_base.copy()
        for distance in np.linspace(0,task['motion']['post_place_retreat_m'],121)[1:]:
            placement_base=start_base.copy();placement_base[0,3]-=distance
            checker.set_base(placement_base);checker.set_state(joints,resting);checker.collisions([]);save('base_retreat')
        home=default_joints(plan['config'],joints)
        for side in ('left','right'):
            target={**joints,**{n:home[n] for n in checker.hand_names[side]}}
            for t in np.linspace(0,1,41)[1:]:
                sample={n:joints[n]+t*(target[n]-joints[n]) for n in joints}
                checker.set_state(sample,resting);checker.collisions([]);save('reset_hands')
            joints=target
        for side in ('left','right'):
            target={**joints,**{n:home[n] for n in checker.names[side]}}
            samples=cached_joint_route(out/f'reset_{side}.yaml',checker,joints,target,checker.names[side],resting,task['motion'])
            for sample in samples:
                checker.set_state(sample,resting);checker.collisions([]);save('reset_arms')
            joints=checker.current_joints()
        target={**joints,**{n:home[n] for n in checker.names['waist']+checker.names['neck']}}
        samples=cached_joint_route(out/'reset_body.yaml',checker,joints,target,checker.names['waist']+checker.names['neck'],resting,task['motion'])
        for sample in samples:
            checker.set_state(sample,resting);checker.collisions([]);save('reset_body')
        if max(abs(checker.current_joints()[n]-home[n]) for n in home)>1e-8:raise RuntimeError('Default reset incomplete')
    np.savez_compressed(out/'placement.npz',qpos=qpos,payload=payload,labels=labels,base_offset=initial_placement_base,base_offsets=placement_bases)
    (out/'manipulation.json').write_text(json.dumps(dict(pickup_source=str(source),
        default_pose='recorded manipulation.poses.default_pose + grasp.hands.home',
        detection='assumed observation at default head pose; real CV is a separate adapter',
        waist_adjusted_after_detection=True,wrist_branch=branch,placement_checks=checks,
        final_bottom_height_m=bottom,table_height_m=cfg['table']['size_m'][2],
        payload_released=True,hands_clear=True,dynamics_validated=False,hardware_commands_sent=False),indent=2))
    print('Placement passed',bottom,flush=True)


if __name__=='__main__':main()
