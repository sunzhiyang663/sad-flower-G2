"""Offline right arm/hand optimization with an explicitly fixed body posture."""
import argparse
import json
from pathlib import Path
import sys
import time
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from simulation.refine_grasp_elbow_down import load


def right_contacts(c, gap, depth, thumb_face_sign=1):
    half = c.model.geom_size[c.payload]
    best = {}
    for stem in ('thumb', 'index', 'middle', 'ring', 'pinky'):
        candidates = []
        for g in c.robot:
            name = c.model.body(int(c.model.geom_bodyid[g])).name
            if not name.startswith('hand_r_' + stem + '_'):
                continue
            segment = np.zeros(6)
            distance = float(mujoco.mj_geomDistance(c.model, c.data, c.payload, g, .5, segment))
            point = c.battery[:3, :3].T @ (segment[3:] - c.battery[:3, 3])
            face = thumb_face_sign * (half[1] if stem == 'thumb' else -half[1])
            error = max(max(0., distance), abs(point[1]-face), max(0., -half[0]+depth-point[0]))
            candidates.append(dict(geom=name, distance_m=distance, point_in_payload_m=point.tolist(), error_m=float(error)))
        best[stem] = min(candidates, key=lambda a: a['error_m'])
    contacts = [s for s, item in best.items() if item['error_m'] <= gap]
    return dict(passed='thumb' in contacts and len(set(contacts)-{'thumb'}) >= 2,
                contacts=contacts, geometry=best)


def audit(c, B, baseline, target):
    palm = np.linalg.inv(B) @ c.c.get_transform_frame_to_world('hand_r_base_link', 'body').as_matrix()
    wrist = np.linalg.inv(B) @ c.c.get_transform_frame_to_world('arm_r_end_link', 'body').as_matrix()
    return dict(palm_in_payload=palm.tolist(),
        wrist_in_payload=wrist.tolist(),
        wrist_axis_long_axis_deg=float(np.rad2deg(np.arccos(np.clip(wrist[0, 2], -1, 1)))),
        torso_rpy_deg=Rotation.from_matrix(c.data.body('body_link5').xmat.reshape(3,3)).as_euler('xyz',degrees=True).tolist(),
        shoulder_world_m=c.data.body('arm_r_link3').xpos.tolist(),
        elbow_world_m=c.data.body('arm_r_link4').xpos.tolist(),
        palm_normal_long_axis_deg=float(np.rad2deg(np.arccos(np.clip(palm[0, 0], -1, 1)))),
        orientation_error_deg=float(np.rad2deg(Rotation.from_matrix(target.T@palm[:3, :3]).magnitude())),
        elbow_below_shoulder_m=float(c.data.body('arm_r_link3').xpos[2]-c.data.body('arm_r_link4').xpos[2]),
        right_joint2_deg=float(np.rad2deg(c.current_joints()['idx62_arm_r_joint2'])),
        fixed_joint_max_change_rad=max(abs(v-baseline[n]) for n, v in c.current_joints().items()
                                     if n not in c.names['right']+c.hand_names['right']))


def main(config):
    cfg, c, B, baseline = load(config)
    original_baseline = baseline.copy()
    baseline.update(cfg.get('fixed_joint_overrides', {}))
    c.set_state(baseline, B)
    out = ROOT/cfg['output']; out.mkdir(parents=True, exist_ok=True)
    (out/'motion.json').write_text(json.dumps(dict(passed=False,reason='Grasp endpoint changed; replan motion')))
    (out/'motion_audit.json').write_text(json.dumps(dict(passed=False,reason='Grasp endpoint changed; recheck motion')))
    names = c.names['right']+c.hand_names['right']
    ids = np.array([int(c.model.joint(n).qposadr[0]) for n in names])
    for n in c.names['right']:
        lo, hi = c.model.joint(n).range.copy()
        c.narrow_limit(n, [lo+cfg['joint_reserve_rad'], hi-cfg['joint_reserve_rad']])
    if cfg.get('right_joint2_range_deg'):
        c.narrow_limit('idx62_arm_r_joint2',np.deg2rad(cfg['right_joint2_range_deg']))
    for item_name, item in c.mimics.items():
        parent = item.get('joint')
        if parent in names:
            scale, offset = float(item.get('multiplier',1)), float(item.get('offset',0))
            if scale:
                limits = sorted((c.model.joint(item_name).range-offset)/scale)
                c.narrow_limit(parent, limits)
    low, high = np.array([c.model.joint(n).range for n in names]).T
    ref = c.data.qpos.copy(); initial = ref[ids].copy()
    target = np.asarray(cfg['palm_rotation_in_payload'], float)
    before = audit(c, B, baseline, target)
    original_position = np.asarray(before['palm_in_payload'])[:3, 3]
    original_position = np.asarray(cfg.get('palm_position_reference_m', original_position))
    axis = cfg.get('alignment_axis', 0)
    thumb_sign = cfg.get('thumb_face_sign', 1)
    right_geoms = {g for g in c.robot if c.model.body(int(c.model.geom_bodyid[g])).name.startswith(('arm_r_', 'hand_r_'))}
    pairs = np.array([(a,b) for a,b in c.collision_pairs if a in right_geoms or b in right_geoms])
    a,b = pairs.T
    allowed = np.array([aa == c.payload and c.model.body(int(c.model.geom_bodyid[bb])).name.startswith('hand_r_')
                        and not c.model.body(int(c.model.geom_bodyid[bb])).name.endswith('camera_link') for aa,bb in pairs])
    # Retain the original 3mm non-contact clearance, with an optimization reserve.
    thresholds = np.where(allowed, -.0005, c.cfg['collision_clearance_m']+.001)
    mimic = []
    for n, item in c.mimics.items():
        if item.get('joint') in names:
            mimic.append((int(c.model.joint(n).qposadr[0]), int(c.model.joint(item.get('joint')).qposadr[0]),
                          float(item.get('multiplier', 1)), float(item.get('offset', 0))))

    def update(x):
        q = ref.copy(); q[ids] = x
        for dst, src, scale, offset in mimic: q[dst] = q[src]*scale+offset
        c.c.update(q)

    def residual(x):
        update(x)
        palm = np.linalg.inv(B) @ c.c.get_transform_frame_to_world('hand_r_base_link','body').as_matrix()
        # Wrist mode aligns the actual end-link +Z (also hand-base +Z).
        # Rotation about the longitudinal axis remains a redundant coordinate.
        errors = list((1000 if axis == 2 else 400)*(palm[:3,axis]-target[:,axis]))
        if axis == 2:
            errors.append(100*max(0., .3-palm[1,0]))
        else:
            errors.append(100*max(0., .3+palm[1,2]))
        errors.extend(10*(palm[:3,3]-original_position))
        dz = c.data.body('arm_r_link4').xpos[2]-c.data.body('arm_r_link3').xpos[2]
        errors.append(1000*max(0., dz+cfg['minimum_elbow_below_shoulder_m']))
        for stem in cfg.get('fit_fingers', ['thumb','middle','ring']):
            g = next(g for g in c.robot if c.model.body(int(c.model.geom_bodyid[g])).name == f'hand_r_{stem}_dip_link')
            segment = np.zeros(6)
            distance = mujoco.mj_geomDistance(c.model,c.data,c.payload,g,.5,segment)
            point = B[:3,:3].T@(segment[3:]-B[:3,3])
            half = c.model.geom_size[c.payload]
            face = thumb_sign * (half[1] if stem=='thumb' else -half[1])
            errors.extend(2000*np.array([distance-.0005, point[1]-face,
                max(0., -half[0]+cfg['wrap_depth_m']-point[0]),
                max(0., abs(point[2])-half[2]+cfg.get('contact_vertical_edge_margin_m', .008))]))
        errors.append(.2*(x[1]-np.deg2rad(cfg['right_joint2_target_deg'])))
        errors.extend(.04*(x-initial))
        distances = np.full(len(pairs), .05)
        near = np.linalg.norm(c.data.geom_xpos[a]-c.data.geom_xpos[b],axis=1) < (c.model.geom_rbound[a]+c.model.geom_rbound[b]+.012)
        for i in np.flatnonzero(near):
            distances[i] = mujoco.mj_geomDistance(c.model,c.data,int(a[i]),int(b[i]),.02,None)
        errors.extend(7000*np.maximum(thresholds-distances,0))
        return np.array(errors)

    rng = np.random.default_rng(2512)
    seeds = []
    previous = out/'attempts_normal_third.json'
    if cfg.get('seed_joints'):
        seeds.append(np.clip([cfg['seed_joints'][n] for n in names],low+1e-6,high-1e-6))
    elif previous.exists():
        good = min(json.loads(previous.read_text()), key=lambda r:r['cost'])
        seeds.append(np.clip([good['joints'][n] for n in names],low+1e-6,high-1e-6))
    else:
        seeds.append(np.clip(initial.copy(),low+1e-6,high-1e-6))
    for k in range(cfg['attempts']-1):
        seed = seeds[0].copy()
        seed[:7] += rng.normal(0, .15 if k<4 else .4, 7)
        seeds.append(np.clip(seed, low+1e-6, high-1e-6))
    best = None; reports = []; started = time.monotonic()
    for attempt, seed in enumerate(seeds):
        solution = least_squares(residual, seed, bounds=(low,high), max_nfev=cfg['max_nfev'],
                                 ftol=1e-9, xtol=1e-9, gtol=1e-9, diff_step=1e-4, x_scale='jac')
        update(solution.x)
        joints = c.current_joints(); collision = None
        try:
            c.set_state(joints,B); collision = c.collisions(['right'])
        except ValueError as exc: collision = str(exc)
        after = audit(c,B,baseline,target)
        wrap = right_contacts(c,cfg['contact_gap_m'],cfg['wrap_depth_m'],thumb_sign)
        angle_key = 'wrist_axis_long_axis_deg' if axis == 2 else 'palm_normal_long_axis_deg'
        passed = (isinstance(collision,dict) and wrap['passed'] and after[angle_key] < 1.
                  and after['elbow_below_shoulder_m'] >= cfg['minimum_elbow_below_shoulder_m']-.001
                  and after['fixed_joint_max_change_rad'] == 0.)
        result = dict(passed=passed, attempt=attempt, cost=float(solution.cost), optimality=float(solution.optimality),
                      nfev=solution.nfev, message=solution.message, before=before, after=after,
                      wrap=wrap, collision=collision, joints=joints, baseline_joints=baseline,
                      original_baseline_joints=original_baseline, alignment_axis=axis,
                      base=c.base_transform.tolist(), payload=B.tolist(), config=str(config),
                      transition_validated=False, dynamics_validated=False, hardware_commands_sent=False)
        reports.append(result)
        (out/'attempts.json').write_text(json.dumps(reports,indent=2))
        print(json.dumps(dict(attempt=attempt,passed=passed,cost=float(solution.cost),after=after,
                              wrap=wrap['contacts'],collision=collision,elapsed_s=time.monotonic()-started)),flush=True)
        if best is None or (passed, -solution.cost) > (best['passed'], -best['cost']):
            best = result; (out/'grasp_pose.json').write_text(json.dumps(result,indent=2))
        if passed: break
    if best['passed']:
        (out/'grasp_pose.yaml').write_text(yaml.safe_dump(dict(executable=False,
            joints=best['joints'],T_world_base=best['base'],T_world_payload=best['payload'],
            T_payload_right_wrist=best['after']['wrist_in_payload'],
            T_payload_right_palm=best['after']['palm_in_payload']),sort_keys=False))
    return 0 if best['passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'simulation/grasp_right_parallel.yaml')
    raise SystemExit(main(parser.parse_args().config))
