"""Recheck parameterized, prepositioned grasp stages with the existing Checker."""
import math
import numpy as np


def check_stage(c, request):
    step=request['step'];settings=request['settings'];sid=step['id']
    start=c.current_joints();base=c.base_transform.copy();battery=c.battery.copy()
    goal=start.copy();end_base=base.copy()
    joint_step=settings['joint_step_rad'];base_step=settings['base_step_m']
    for value in (joint_step,base_step,settings['minimum_right_elbow_drop_m']):
        if not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:raise ValueError('采样步长和肘部高度余量须为有限正数')
    if joint_step>.0075 or base_step>.0025:raise ValueError('不得降低已验证抓取段的碰撞采样精度')
    if step['type']=='direct_drive':
        distance=step['distance_m']
        if not math.isfinite(distance) or (distance<=0 if sid=='right_approach' else distance>=0):raise ValueError('接近须前进，拽出须后退')
        end_base[:3,3]+=distance*base[:3,0]
        count=max(1,math.ceil(abs(distance)/base_step))
    else:
        names=c.names['left'] if step['type']=='arm_joint' else c.hand_names[step['side']]
        if set(step['targets'])!=set(names):raise ValueError('本阶段须完整指定所选手/臂的关节，不能改动其他组')
        goal.update(step['targets']);c.set_state(goal,battery);c.set_state(start,battery)
        count=max(1,math.ceil(max(abs(goal[n]-start[n]) for n in names)/joint_step))
    contacts=[] if sid=='right_approach' else ['right','left'] if sid in ('left_entry','left_grasp') else ['right']
    right=c.c.get_transform_frame_to_world('arm_r_end_link','body').as_matrix().copy()
    attachment=np.linalg.inv(right)@battery
    old_penetration=c.cfg['contact_penetration_tolerance_m']
    if sid=='left_entry':c.cfg['contact_penetration_tolerance_m']=0.
    minimum=float('inf');elbow_min=float('inf');yaw_min=float('inf');yaw_max=-float('inf')
    try:
        for alpha in np.linspace(0,1,count+1):
            q={n:float(start[n]+alpha*(goal[n]-start[n])) for n in start}
            station=base.copy();station[:3,3]=base[:3,3]+alpha*(end_base[:3,3]-base[:3,3])
            c.set_base(station);c.set_state(q,battery)
            wrist=c.c.get_transform_frame_to_world('arm_r_end_link','body').as_matrix()
            if sid=='partial_pull':c.set_battery(wrist@attachment)
            col=c.collisions(contacts)['minimum_near_distance_m']
            if col is not None:minimum=min(minimum,col)
            drop=float(c.data.body('arm_r_link3').xpos[2]-c.data.body('arm_r_link4').xpos[2]);elbow_min=min(elbow_min,drop)
            axis=c.battery[:3,:3].T@wrist[:3,2];yaw=float(np.degrees(np.arctan2(axis[1],axis[0])))
            yaw_min=min(yaw_min,yaw);yaw_max=max(yaw_max,yaw)
            low,high=settings['wrist_yaw_range_deg']
            if drop<settings['minimum_right_elbow_drop_m'] or not low-1e-4<=yaw<=high+1e-4 or abs(axis[2])>1e-5:
                raise ValueError('右肘高度或腕部水平朝向超出本动作要求')
        grasp_geometry=None
        if sid=='right_grasp':
            from simulation.refine_right_parallel import right_contacts
            grasp_geometry=right_contacts(c,.002,.001)
        elif sid=='left_grasp':
            from simulation.prepare_map9_cross_grip import wrap_contacts
            grasp_geometry=wrap_contacts(c,.004,.01)
        if grasp_geometry is not None and not grasp_geometry['passed']:
            raise ValueError('闭合目标没有形成拇指与至少两指的对向几何接触；不能把无碰撞当作抓取成功')
        attachments={s:(np.linalg.inv(c.c.get_transform_frame_to_world(f'arm_{letter}_end_link','body').as_matrix())@c.battery).tolist() for s,letter in [('right','r'),('left','l')]}
        return dict(final_state=dict(joints=c.current_joints(),base=c.base_transform.tolist(),payload=c.battery.tolist()),
            samples_checked=count+1,minimum_near_distance_m=None if minimum==float('inf') else minimum,
            minimum_right_elbow_drop_m=elbow_min,wrist_yaw_deg_range=[yaw_min,yaw_max],
            payload_collision_included=True,contacts=contacts,
            T_wrist_payload=attachments,grasp_geometry=grasp_geometry,hardware_commands_sent=False)
    finally:c.cfg['contact_penetration_tolerance_m']=old_penetration
