"""Render/replay a saved pose_grasp report in MuJoCo. No GDK, no mj_step."""
import argparse
import hashlib
import html
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

STAGE_NAMES = {
    'lower_waist': '调整腰部降低肩部', 'left_support_front': '左手在货架前方调整托举朝向',
    'left_support_wait': '左手先到托举等待位',
    'initial': '初始站位', 'right_open': '右手张开', 'right_pregrasp': '右手预抓取',
    'right_approach': '右手接近端面', 'right_grasp': '右手位置抓握',
    'partial_pull': '右手部分拽出', 'verify_pose': '核对壳体位姿',
    'left_clear': '左手侧向让位', 'left_entry': '左手进入托底区',
    'left_orient': '左手调整托底朝向', 'left_lower': '左手下降至托底高度',
    'left_open': '左手张开', 'left_seat': '左手托底就位', 'left_grasp': '左手包握',
    'left_retract': '左手退至货架正前方', 'left_wait': '左手下方预落位',
    'right_front': '右手移动至货架正前方', 'right_orient': '右手调整端面抓握朝向',
    'right_center': '右手在货架前方对准端面',
    'full_pull': '双手完全取出', 'lift': '双手抬升',
}


def configure_graphics(viewer):
    if viewer:
        if not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
            raise RuntimeError('当前终端没有图形桌面；请先观看输出的index.html／replay.mp4，或在有桌面的终端运行--viewer')
        os.environ['MUJOCO_GL'] = 'glfw'
        return
    os.environ.setdefault('MUJOCO_GL', 'osmesa')
    os.environ.setdefault('LP_NUM_THREADS', '4')
    vendor = str(HERE/'vendor_libs/usr/lib/aarch64-linux-gnu')
    paths = os.environ.get('LD_LIBRARY_PATH', '').split(':')
    if platform.machine() in ('aarch64', 'arm64') and os.environ['MUJOCO_GL'] == 'osmesa' and Path(vendor, 'libOSMesa.so.8').is_file() and vendor not in paths:
        env = dict(os.environ, LD_LIBRARY_PATH=':'.join([vendor, *filter(None, paths)]))
        os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


def build(plan_path, output, fps, sample_stride=1):
    import mujoco
    import numpy as np
    import yaml
    from scipy.spatial.transform import Rotation
    from simulation.pose_grasp_check import Checker
    from manipulation.wrist_pose import transform, from_pose7

    plan_bytes = plan_path.read_bytes()
    input_bytes = (plan_path.parent/'inputs.yaml').read_bytes()
    plan = json.loads(plan_bytes)
    inputs = yaml.safe_load(input_bytes)['inputs']
    cfg = plan['config']
    scene = inputs['scene']
    if scene is None or inputs['initial_joints'] is None:
        raise ValueError('回放需要含scene及冻结initial_joints的完整离线方案，请先运行scripts/prepare_pose_grasp.py')
    checker = Checker(cfg, scene, include_visuals=True)
    checker.scene = scene
    source_meta = plan['checks']['collision_backend']
    for key in ('urdf_sha256', 'srdf_sha256', 'robot_collision_geoms', 'self_collision_pairs'):
        if checker.metadata[key] != source_meta[key]:
            raise ValueError(f'回放模型与原检查不一致：{key}；请重新生成方案')
    joints = dict(inputs['initial_joints'])
    from perception.pose_input import resolve_battery_pose
    battery, _ = resolve_battery_pose(inputs, np.array([cfg['battery']['size'][a] for a in 'xyz']))
    attached = None
    frames, chapters = [], []
    failure = None

    def begin(stage, kind):
        chapters.append({'stage': stage, 'name': STAGE_NAMES.get(stage, stage),
                         'kind': kind, 'frame': len(frames), 'time_s': len(frames)/fps})
        print(f'{kind}: {stage}', flush=True)

    def capture(stage, kind, targets=None, ghost=None, repeat=1, raw=False):
        if not raw:
            checker.set_state(joints, battery)
            if attached is not None:
                end = checker.c.get_transform_frame_to_world('arm_r_end_link', 'body').as_matrix()
                checker.set_battery(end @ attached)
        markers = np.full((2, 7), np.nan)
        for side, target in (targets or {}).items():
            markers[0 if side == 'left' else 1] = target
        frame = (checker.data.qpos.copy(), checker.data.mocap_pos.copy(), checker.data.mocap_quat.copy(),
                 stage, kind, markers, np.full((4, 4), np.nan) if ghost is None else transform(ghost),
                 checker.base_transform.copy())
        frames.extend([frame]*repeat)

    begin('initial', 'checked')
    capture('initial', 'checked', repeat=fps)
    for stage in plan['checks']['history']:
        name = stage['id']
        targets = stage.get('requested_end_targets', {})
        if stage['status'] != 'completed':
            begin(name, 'rejected')
            failure = {'stage': name, 'saved_error': stage.get('error'), 'candidate_is_executable': False}
            safe_joints, safe_battery = dict(joints), battery.copy()
            safe_base = checker.base_transform.copy()
            if targets:
                checker.set_state(joints, battery)
                ends = {s: checker.c.get_transform_frame_to_world(f'arm_{letter}_end_link', 'body').as_matrix()
                        for s, letter in (('left', 'l'), ('right', 'r'))}
                from manipulation.wrist_pose import pose7
                try:
                    if stage.get('requested_joint_targets'):
                        checker.request({'operation': 'arm_joint_motion', 'side': 'left',
                            'targets': stage['requested_joint_targets'], 'joints': joints,
                            'battery': battery.tolist(), 'T_world_base': safe_base.tolist()})
                    else:
                        checker.request({'operation': 'motion', 'joints': joints, 'battery': battery.tolist(),
                        'contacts': ['right'] if attached is not None else [],
                        'ends': {s: pose7(T) for s, T in ends.items()},
                        'goals': {s: from_pose7(p).tolist() for s, p in targets.items()},
                        'T_world_base': safe_base.tolist(),
                        'allow_waist': plan['checks'].get('coordination', {}).get('allow_waist', False),
                        'passive_wait_sides': stage.get('passive_wait_sides', []),
                            'payload_from_right': None if attached is None else attached.tolist()})
                    failure['reproduction'] = 'saved rejection did not reproduce; candidate remains unvalidated'
                except Exception as error:
                    failure['reproduction'] = f'{type(error).__name__}: {error}'
                failure['candidate_joints'] = checker.current_joints()
                capture(name, 'rejected', targets, repeat=fps*3, raw=True)
            else:
                capture(name, 'rejected', targets, repeat=fps*3)
            joints, battery = safe_joints, safe_battery
            checker.set_base(safe_base)
            break
        begin(name, 'checked')
        path = stage.get('checked_path', {}).get('path', [])
        if path:
            indices = sorted(set([*range(0, len(path), sample_stride), len(path)-1]))
            for sample_index in indices:
                sample = path[sample_index]
                joints = dict(sample['joints'])
                if 'T_world_base' in sample:
                    checker.set_base(sample['T_world_base'])
                capture(name, 'checked', targets)
            battery = transform(stage['final_T_base_battery'])
        elif stage.get('targets'):
            goals = {n: v for values in stage['targets'].values() for n, v in values.items()}
            start = dict(joints)
            count = stage['offline_check']['samples']
            # Use the same discrete finger interpolation fractions as the checker.
            for index in np.linspace(1, count, fps*2).round().astype(int):
                joints.update({n: start[n]+index/count*(v-start[n]) for n, v in goals.items()})
                capture(name, 'checked')
            joints.update(goals)
        else:
            capture(name, 'checked', repeat=fps)
        if 'T_right_end_battery' in stage:
            attached = transform(stage['T_right_end_battery'])
        capture(name, 'checked', targets, repeat=max(1, fps//3))

    # Show the complete intended targets after stopping, without moving robot joints.
    if failure:
        remaining = plan['nominal_stages']
        first = next((i for i, s in enumerate(remaining) if s['id'] == failure['stage']), 0)
        selected = [s for s in remaining[first:] if s.get('end_targets_xyzw') or s.get('T_base_battery')]
        for stage in selected:
            begin(stage['id'], 'reference')
            capture(stage['id'], 'reference', stage.get('end_targets_xyzw'),
                    stage.get('T_base_battery'), repeat=fps*2)

    spec = checker.spec
    floor_from_base = transform(inputs['height_reference']['T_floor_base'])
    base_from_floor = np.linalg.inv(floor_from_base)
    spec.worldbody.add_geom(name='preview_floor', type=mujoco.mjtGeom.mjGEOM_PLANE,
        pos=base_from_floor[:3, 3], quat=Rotation.from_matrix(base_from_floor[:3, :3]).as_quat(scalar_first=True),
        size=[3, 3, .05], rgba=[.18, .22, .27, 1], contype=0, conaffinity=0, group=2)
    spec.worldbody.add_light(pos=[1, -2, 3], dir=[-.2, .2, -1], diffuse=[.8, .8, .8])
    model = spec.compile()
    if model.nq != checker.model.nq or [model.joint(i).name for i in range(model.njnt)] != [checker.model.joint(i).name for i in range(checker.model.njnt)]:
        raise ValueError('渲染模型关节顺序变化，禁止按原样本回放')
    # Exact visual meshes are group 1; group 0 is the unchanged collision model.
    for g in range(model.ngeom):
        name = model.body(int(model.geom_bodyid[g])).name
        if name in checker.robot_bodies:
            model.geom_rgba[g] = [.70, .76, .80, 1]
            if name.startswith('arm_'):
                model.geom_rgba[g] = [.16, .46, .51, 1]
            elif name.startswith('hand_'):
                model.geom_rgba[g] = [.87, .89, .91, 1]
        elif name == 'payload':
            model.geom_rgba[g] = [.77, .81, .84, 1]
            model.geom_group[g] = 3
        elif name == 'scene_support':
            model.geom_rgba[g] = [.46, .33, .22, 1]
            model.geom_group[g] = 2
        elif name.startswith('scene_'):
            model.geom_group[g] = 2
            # Keep the measured obstructions visible without hiding both hands.
            # Only rendering alpha changes; collision geometry stays intact.
            if name.endswith('_panel'):
                model.geom_rgba[g] = [.50, .63, .72, .18]
            elif name in ('scene_rack_shelf_3', 'scene_opening_roof'):
                model.geom_rgba[g] = [.55, .64, .68, .32]
    model.vis.global_.offwidth = 960
    model.vis.global_.offheight = 540
    model.vis.quality.offsamples = 0
    output.mkdir(parents=True, exist_ok=True)
    (output/'source_plan.json').write_bytes(plan_bytes)
    (output/'source_inputs.yaml').write_bytes(input_bytes)
    mujoco.mj_saveModel(model, str(output/'scene.mjb'))
    np.savez_compressed(output/'replay.npz', qpos=np.array([f[0] for f in frames]),
        mocap_pos=np.array([f[1] for f in frames]), mocap_quat=np.array([f[2] for f in frames]),
        stages=np.array([f[3] for f in frames]), kinds=np.array([f[4] for f in frames]),
        targets=np.array([f[5] for f in frames]), ghost=np.array([f[6] for f in frames]),
        base_poses=np.array([f[7] for f in frames]), fps=fps)
    report = {'scope': 'MuJoCo kinematic replay + rejected candidate diagnosis + reference targets',
        'hardware_commands_sent': False, 'dynamics_validated': False,
        'source_plan': str(plan_path), 'source_plan_sha256': hashlib.sha256(plan_bytes).hexdigest(),
        'source_status': plan['status'], 'validated_through': plan.get('validated_through'),
        'full_extraction_trajectory_validated': plan.get('full_extraction_trajectory_validated', False),
        'model': 'G2_t2_crsB_o12_t2', 'model_metadata': checker.metadata,
        'approach_strategy': plan['checks'].get('approach_strategy', 'legacy'),
        'coordination': plan['checks'].get('coordination', {}),
        'base_preset': plan.get('base_preset', {}),
        'pipeline': plan['checks'].get('pipeline', []),
        'upper_body_prepared': plan['checks'].get('upper_body_prepared', False),
        'phase_state': plan['checks'].get('phase_state'),
        'phase_history': plan['checks'].get('phase_history', []),
        'waiting_position_tolerance_m': cfg['pose_grasp'].get('waiting_position_tolerance_m'),
        'height_reference': inputs['height_reference'], 'frames': len(frames), 'fps': fps,
        'path_sample_stride': sample_stride,
        'duration_s': len(frames)/fps, 'chapters': chapters, 'failure': failure,
        'base_retreat_m': float(np.linalg.norm(frames[-1][7][:2, 3])),
        'waist_joint_range_rad': {n: float(np.ptp([f[0][int(model.joint(n).qposadr[0])] for f in frames if f[4] == 'checked']))
                                 for n in cfg['manipulation']['joint_names']['waist']},
        'note': 'Playback time is illustrative. Payload follows a rigid no-slip assumption. Orange targets have no accepted robot trajectory.'}
    (output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return report


def apply_frame(model, data, recording, index):
    import mujoco
    if 'base_poses' in recording:
        from scipy.spatial.transform import Rotation
        base = recording['base_poses'][index]
        root = model.body('base_link').id
        model.body_pos[root] = base[:3, 3]
        model.body_quat[root] = Rotation.from_matrix(base[:3, :3]).as_quat(scalar_first=True)
    data.qpos[:] = recording['qpos'][index]
    data.mocap_pos[:] = recording['mocap_pos'][index]
    data.mocap_quat[:] = recording['mocap_quat'][index]
    mujoco.mj_kinematics(model, data)
    mujoco.mj_camlight(model, data)


def rejected_sides(stage):
    return ('r',) if stage.startswith('right_') else ('l',) if stage.startswith('left_') else ('l', 'r')


def decorate(scene, recording, index, model):
    import mujoco
    import numpy as np
    from manipulation.wrist_pose import from_pose7
    kind = str(recording['kinds'][index])
    color = [.2, .9, .65, 1] if kind == 'checked' else [1, .25, .2, 1] if kind == 'rejected' else [1, .65, .15, 1]
    for side, target in zip(('L', 'R'), recording['targets'][index]):
        if not np.isfinite(target).all():
            continue
        T = from_pose7(target)
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, [.014]*3, T[:3, 3], np.eye(3).ravel(), color)
        scene.ngeom += 1
        for axis, rgba in enumerate(([1,.15,.15,1], [.15,1,.15,1], [.15,.5,1,1])):
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.eye(3).ravel(), rgba)
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, .003, T[:3,3], T[:3,3]+T[:3,axis]*.065)
            scene.ngeom += 1
    ghost = recording['ghost'][index]
    if np.isfinite(ghost).all():
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_BOX, model.geom('payload').size,
                          ghost[:3,3], ghost[:3,:3].ravel(), [1,.65,.15,.35])
        scene.ngeom += 1


def render(output, report, still=False):
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from mujoco.rendering.classic.renderer import Renderer
    model = mujoco.MjModel.from_binary_path(str(output/'scene.mjb'))
    data = mujoco.MjData(model)
    recording = np.load(output/'replay.npz')
    renderer = Renderer(model, 540, 960)
    detail_renderer = Renderer(model, 220, 320)
    option = mujoco.MjvOption()
    option.geomgroup[0] = 0
    option.geomgroup[3] = 1  # Payload uses its own visible group.
    detail_option = mujoco.MjvOption()
    detail_option.geomgroup[0] = 0
    detail_option.geomgroup[2] = 0  # Hide rack only in inset; collision checks are unchanged.
    detail_option.geomgroup[3] = 1
    model.geom_group[model.geom('payload').id] = 3  # Also supports older frozen replay models.
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [.35, -.05, .85]
    camera.distance, camera.azimuth, camera.elevation = 2.9, 120, -22
    detail_camera = mujoco.MjvCamera()
    detail_camera.distance, detail_camera.azimuth, detail_camera.elevation = .85, 225, -12
    colors = model.geom_rgba.copy()
    font_path = str(ROOT/'assets/fonts/NotoSansCJK-Regular.ttc')
    font = ImageFont.truetype(font_path, 22)
    small = ImageFont.truetype(font_path, 16)
    chapters = {c['frame']: c for c in report['chapters']}
    frames = sorted(chapters) if still else range(report['frames'])
    process = None
    if not still:
        process = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
            '-pix_fmt', 'rgb24', '-s', '960x540', '-r', str(report['fps']), '-i', '-', '-an',
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', str(output/'replay.mp4')], stdin=subprocess.PIPE)
    previous, pixels = None, None
    try:
        for index in frames:
            same_pose = (previous is not None and recording['stages'][index] == recording['stages'][previous]
                         and recording['kinds'][index] == recording['kinds'][previous]
                         and all(np.array_equal(recording[key][index], recording[key][previous], equal_nan=True)
                                 for key in ('qpos', 'mocap_pos', 'mocap_quat', 'targets', 'ghost', 'base_poses')
                                 if key in recording))
            if same_pose and index not in chapters:
                if process:
                    process.stdin.write(pixels)
                continue
            apply_frame(model, data, recording, index)
            kind, stage = str(recording['kinds'][index]), str(recording['stages'][index])
            model.geom_rgba[:] = colors
            if kind == 'rejected':
                for g in range(model.ngeom):
                    if model.body(int(model.geom_bodyid[g])).name.startswith(tuple(prefix+side+'_' for side in rejected_sides(stage) for prefix in ('arm_', 'hand_'))):
                        model.geom_rgba[g] = [1, .25, .18, 1]
            renderer.update_scene(data, camera, scene_option=option)
            decorate(renderer.scene, recording, index, model)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
            image = Image.fromarray(renderer.render())
            if kind == 'rejected':
                detail_camera.lookat[:] = data.body('hand_'+rejected_sides(stage)[0]+'_base_link').xpos
                detail_camera.distance = .65
            else:
                detail_camera.lookat[:] = data.body('payload').xpos+[0,0,-.03]
                detail_camera.distance = .85
            detail_renderer.update_scene(data, detail_camera, scene_option=detail_option)
            decorate(detail_renderer.scene, recording, index, model)
            detail_renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            detail_renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
            image.paste(Image.fromarray(detail_renderer.render()), (630, 270))
            draw = ImageDraw.Draw(image)
            color = (64,220,170) if kind == 'checked' else (255,100,85) if kind == 'rejected' else (255,191,76)
            title = '通过样本回放' if kind == 'checked' else '被拒绝的候选姿态' if kind == 'rejected' else '后续参考目标 · 机器人停止'
            draw.rectangle((0,0,960,76), fill=(15,23,35))
            draw.text((18,8), title+' / '+STAGE_NAMES.get(stage, stage), font=font, fill=color)
            subtitle = ('底盘固定 · 左手等待 → 右手拽出 → 托底 → 双手抬升' if report.get('full_extraction_trajectory_validated')
                        else '底盘不动 · 降腰 → 左手托举等待 → 右手接近' if report.get('approach_strategy') == 'fixed_left_first'
                        else '固定腰颈和底盘 · 先识别再规划腕部' if report.get('approach_strategy', '').startswith('fixed_')
                        else '正面货架 · 左手先落位 · 第二层净高0.28m' if report.get('approach_strategy') == 'frontal_left_first'
                        else '台面离地0.83m')
            draw.text((18,43), 'G2_t2_crsB + O12 t2  |  '+subtitle+'  |  绿/红/橙=通过/拒绝/参考', font=small, fill=(211,223,233))
            draw.rectangle((629,269,951,491), outline=color, width=2)
            draw.text((640,274), '手部近景（隐藏货架）', font=small, fill=color)
            draw.rectangle((0,500,960,540), fill=(15,23,35))
            draw.text((18,510), 'MuJoCo 运动学回放 · 无实机连接 · 刚性无滑移假设 · 未验证动力学/抓稳', font=small, fill=(211,223,233))
            if index in chapters:
                filename = f'{len([c for c in chapters if c < index]):02d}_{kind}_{stage}.png'
                image.save(output/filename)
                chapters[index]['image'] = filename
                print(f'render: {index}/{report["frames"]} {stage} ({kind})', flush=True)
            if process:
                pixels = np.asarray(image).tobytes()
                process.stdin.write(pixels)
            previous = index
    finally:
        renderer.close()
        detail_renderer.close()
        if process:
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError('FFmpeg视频编码失败')
    report['video_rendered'] = not still
    (output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    write_page(output, report)


def write_page(output, report):
    fixed_body = report.get('approach_strategy', '').startswith('fixed_')
    full_task = report.get('approach_strategy') in ('fixed_pregrasp_extraction', 'fixed_left_first')
    page_title = 'G2 · 固定上半身 → 识别壳体 → 抓取目标 → 腕部规划' if fixed_body else 'G2 · 83cm台面：拽出与下方托起'
    buttons = ''.join(f'<button data-time="{c["time_s"]}" class="{c["kind"]}">{html.escape(c["name"])} · {c["kind"]}</button>' for c in report['chapters'])
    failure = report.get('failure') or {}
    rejection = html.escape(failure.get('reproduction') or failure.get('saved_error') or '无拒绝记录')
    images = ''.join(f'<figure><img loading="lazy" src="{c["image"]}"><figcaption>{html.escape(c["name"])} · {c["kind"]}</figcaption></figure>' for c in report['chapters'] if c.get('image'))
    coordination = ''
    if fixed_body:
        coordination = ('<p>顺序：上半身到合适位置并固定 → 检测壳体位姿 → 生成抓取位置 → 规划腕部。'
                        '本回放以冻结的离线初始构型假定已到位，未验证现场起点到该构型的运动；相机输入为假定识别结果，尚未连接现场识别程序。'
                        '检测后底盘、腰部和颈部固定，只规划双臂。'
                        '若需要改变观察姿态或底盘位置，应回到第一步，然后重新识别。</p>')
        coordination += ('<p>本方案继续检查抓握、拽出、左手托底及双手取出抬升；完整动作是否通过以报告标志为准。</p>' if full_task
                         else '<p>本方案止于腕部接近，不执行抓握、拖拽或抬升。</p>')
        coordination += f'<p>已接受样本的腰部最大变化{max(report["waist_joint_range_rad"].values()):.6f}rad；底盘位移{report["base_retreat_m"]:.6f}m。含静态碰撞检查，未接入实时点云。</p>'
        station = report.get('base_preset', {})
        if station.get('base_offset_xy_m') is not None:
            coordination += (f'<p>相对原场景的底盘[x,y]偏移：{html.escape(str(station["base_offset_xy_m"]))}m；'
                             f'壳体单独向外移动{station.get("shell_outward_shift_m",0):.3f}m。几何重叠不证明支撑稳定或抓稳。</p>')
        if station.get('base_backoff_before_detection_m'):
            coordination += (f'<p>第一步站位比原方案后移{station["base_backoff_before_detection_m"]:.3f}m；'
                             f'货架前沿距base_link原点{station["shelf_front_distance_from_base_origin_m"]:.4f}m（不是车壳前缘间隙）。'
                             '此识别位姿是新站位的离线预测，现场移到位后须重新识别。</p>')
    if report.get('approach_strategy') == 'frontal_left_first':
        coordination = (f'<p>正面货架，左手先在前沿外侧下方等待。净宽1.10m、深度0.50m，层板厚0.02m，三层顶面离地0.53/0.83/1.13m；第二层净高0.28m，壳体前端内缩0.07m。'
                        f'本次已接受回放中的底盘后退{report["base_retreat_m"]:.3f}m；腰部最大关节变化{max(report["waist_joint_range_rad"].values()):.3f}rad。</p>'
                        '<p>检查静态货架、壳体及机器人碰撞；未接入实时点云或导航后端，未连接实机。</p>')
        coordination += f'<p>未接触负载的空闲手使用固定中心、{report["waiting_position_tolerance_m"]*100:.0f}cm位置及0.10rad朝向的等待区域；抓持手与主动目标仍使用2mm／0.002rad的解算要求。等待区域不累计漂移，碰撞净空要求保持3mm。</p>'
    if report.get('approach_strategy') == 'fixed_left_first':
        page_title = ('G2 · 左手等待 → 右手拽出 → 左手托底 → 双手取出抬升'
                      if report.get('full_extraction_trajectory_validated') else
                      'G2 · 底盘不动 → 降低肩部 → 左手托举等待 → 右手接近')
        coordination = ('<p>回放从指定底盘站位开始，动作中底盘固定。先检查降腰关节插值，再固定腰颈；假定识别输入对应降腰后的相机位姿。'
            '左手先到未来露出底面下方的等待位，右手随后接近；等待阶段左手尚未接触或承重，部分拽出后再托底包握。</p>'
            '<p>绿色只表示已检查样本，后续橙色目标不驱动机器人。初始双臂构型为离线假设，实机到该构型的过渡、实时识别和动力学均未验证。</p>')
        if report.get('upper_body_prepared'):
            coordination = coordination.replace('先检查降腰关节插值，再固定腰颈；', '上游已提供到位后的构型，本任务固定腰颈；')
        station = report.get('base_preset', {})
        if station.get('base_offset_xy_m') is not None:
            coordination += (f'<p>相对原始场景的底盘[x,y]偏移：{html.escape(str(station["base_offset_xy_m"]))}m；'
                f'壳体向出口外移{station.get("shell_outward_shift_m", 0):.3f}m。此站位的相机输入为离线预测；现场到位后须重新识别。</p>')
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{page_title}</title><style>
body{{margin:24px auto;padding:0 20px;max-width:1100px;background:#101722;color:#e0e9f0;font:16px/1.6 sans-serif}}h1{{font-size:27px}}video{{width:100%;border-radius:12px;background:#070b10}}button{{background:#243246;color:inherit;border:1px solid #42536b;padding:9px;margin:4px;border-radius:6px;cursor:pointer}}.checked{{border-color:#40dcaa}}.rejected{{border-color:#ff6455}}.reference{{border-color:#ffbf4c}}a{{color:#80d5df}}pre{{white-space:pre-wrap;background:#202b3b;padding:16px;border-radius:8px}}.shots{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}}figure{{margin:0}}img{{width:100%;border-radius:8px}}figcaption{{font-size:14px;color:#b8c8d5}}</style>
<h1>{page_title}</h1><p>G2_t2_crsB / O12 t2。绿色为已通过的离线样本；红色为被拒绝候选；橙色仅表示后续目标，机器人保持停止。台面／壳体底面离地0.83m，壳体中心0.9375m。</p>
{coordination}
<video id="replay" src="replay.mp4" poster="{report['chapters'][0].get('image','')}" controls preload="metadata"></video><div>{buttons}</div>
<p>运动学回放，未执行动力学积分。壳体随右腕运动使用刚性无滑移假设；手指位置响应与接触几何不证明抓稳。</p>
<p>规划状态：<strong>{html.escape(report['source_status'])}</strong>；通过至 <strong>{html.escape(str(report['validated_through']))}</strong>。底盘停靠、相机外参和场景尺寸仍含假设。</p><pre>{rejection}</pre>
<p><a href="replay.mp4" download>下载视频</a> · <a href="report.json">查看报告</a> · <a href="scene.mjb" download>MuJoCo模型</a> · <a href="replay.npz" download>回放数据</a></p><div class="shots">{images}</div>
<script>document.querySelectorAll('button[data-time]').forEach(b=>b.onclick=()=>{{document.querySelector('video').currentTime=Number(b.dataset.time);document.querySelector('video').pause();}});</script></html>'''
    (output/'index.html').write_text(page)


def view(output):
    import mujoco
    import mujoco.viewer
    import numpy as np
    model = mujoco.MjModel.from_binary_path(str(output/'scene.mjb'))
    data = mujoco.MjData(model)
    recording = np.load(output/'replay.npz')
    fps = int(recording['fps'])
    playback = {'playing': True, 'index': 0}
    colors = model.geom_rgba.copy()
    last_stage = None
    def key(code):
        if code == 32:
            playback['playing'] = not playback['playing']
        elif code == 82:
            playback['index'] = 0
        elif code in (262, 263):
            playback['playing'] = False
            playback['index'] = min(len(recording['qpos'])-1, max(0, playback['index']+(fps if code == 262 else -fps)))
    with mujoco.viewer.launch_passive(model, data, key_callback=key) as viewer:
        viewer.cam.lookat[:] = [.35, -.05, .85]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.9, 120, -22
        viewer.opt.geomgroup[0] = 0
        while viewer.is_running():
            started = time.monotonic()
            with viewer.lock():
                apply_frame(model, data, recording, playback['index'])
                kind, stage = str(recording['kinds'][playback['index']]), str(recording['stages'][playback['index']])
                model.geom_rgba[:] = colors
                if kind == 'rejected':
                    for g in range(model.ngeom):
                        if model.body(int(model.geom_bodyid[g])).name.startswith(tuple(prefix+side+'_' for side in rejected_sides(stage) for prefix in ('arm_', 'hand_'))):
                            model.geom_rgba[g] = [1, .25, .18, 1]
                viewer.user_scn.ngeom = 0
                decorate(viewer.user_scn, recording, playback['index'], model)
            viewer.sync()
            if (stage, kind) != last_stage:
                print(f'{kind}: {STAGE_NAMES.get(stage,stage)}', flush=True)
                last_stage = (stage, kind)
            if playback['playing']:
                playback['index'] = (playback['index']+1) % len(recording['qpos'])
            time.sleep(max(0., 1/fps-(time.monotonic()-started)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=ROOT/'outputs/debug/extraction_83cm/plan.json')
    parser.add_argument('--output', type=Path, default=HERE/'outputs/extraction_83cm')
    parser.add_argument('--fps', type=int, default=12)
    parser.add_argument('--sample-stride', type=int, default=1, help='每N个已检查路径样本显示1个；各段末样本始终保留')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--viewer', action='store_true', help='有图形桌面时打开可旋转缩放的MuJoCo回放')
    mode.add_argument('--still', action='store_true', help='只渲染阶段截图')
    mode.add_argument('--build-only', action='store_true', help='只生成独立模型和回放数据')
    mode.add_argument('--render-only', action='store_true', help='使用已冻结的模型/回放数据重渲染，不重新求解候选')
    args = parser.parse_args()
    if not 1 <= args.fps <= 60:
        parser.error('--fps须为1到60')
    if args.sample_stride < 1:
        parser.error('--sample-stride须为正整数')
    output = args.output.resolve()
    configure_graphics(args.viewer)
    if args.viewer:
        view(output)
        return
    report = json.loads((output/'report.json').read_text()) if args.render_only else build(args.plan.resolve(), output, args.fps, args.sample_stride)
    if not args.build_only:
        render(output, report, args.still)
    print(output/'index.html' if not args.build_only else output/'report.json', flush=True)


if __name__ == '__main__':
    main()
