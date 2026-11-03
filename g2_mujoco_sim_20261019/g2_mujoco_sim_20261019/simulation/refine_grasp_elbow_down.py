"""Offline grasp endpoint optimization; no trajectory or hardware commands.

Uses the recorded cross-wrap grip as the Cartesian reference, MuJoCo FK and
SciPy bounded least squares to select redundant waist/arm coordinates.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import yaml
from simulation.pose_grasp_check import Checker
from simulation.prepare_map9_cross_grip import reserve_joint_limits, wrap_contacts


def load(config_path, visuals=False):
    cfg = yaml.safe_load(config_path.read_text())
    source = ROOT / cfg['model_reference']
    plan = json.loads((source / 'source_plan.json').read_text())
    inputs = yaml.safe_load((source / 'source_inputs.yaml').read_text())['inputs']
    checker = Checker(plan['config'], inputs['scene'], include_visuals=visuals)
    for side, faces in cfg.get('contact_faces', {}).items():
        checker.cfg['contact_faces'][side] = list(faces)
    checker.scene = inputs['scene']
    record = np.load(ROOT / cfg['source'])
    index = int(np.flatnonzero(record['labels'] == cfg['stage'])[-1])
    base = record['base_offsets'][index]
    battery = base @ record['payload'][index]
    checker.set_base(base)
    checker.c.update(record['qpos'][index])
    checker.set_battery(battery)
    return cfg, checker, battery, checker.current_joints()


def metrics(checker):
    elbows = {}
    for side, letter in [('left', 'l'), ('right', 'r')]:
        shoulder = checker.data.body(f'arm_{letter}_link3').xpos.copy()
        elbow = checker.data.body(f'arm_{letter}_link4').xpos.copy()
        elbows[side] = dict(shoulder_world_m=shoulder.tolist(), elbow_world_m=elbow.tolist(),
                           below_shoulder_m=float(shoulder[2] - elbow[2]))
    palm = np.linalg.inv(checker.battery) @ checker.c.get_transform_frame_to_world('hand_l_base_link', 'body').as_matrix()
    wrap = wrap_contacts(checker, .004, .01)
    ring = wrap['geometry']['ring']
    return dict(elbows=elbows, wrap=wrap, left_palm_in_payload=palm.tolist(),
                left_palm_up_dot=float(palm[2, 0]),
                fingers_to_long_edge_deg=float(np.rad2deg(np.arccos(np.clip(abs(palm[0, 2]), 0, 1)))),
                ring_longitudinal_midpoint_error_m=abs(ring['point_in_payload_m'][0]),
                right_joint2_deg=float(np.rad2deg(checker.current_joints()['idx62_arm_r_joint2'])))


def main(config_path):
    cfg, checker, battery, baseline = load(config_path)
    before = metrics(checker)
    goals = {s: checker.c.get_transform_frame_to_world(f'arm_{s}_end_link', 'body').as_matrix().copy() for s in 'lr'}
    goals['l'][:3, 3] += battery[:3, 0] * cfg['left_forward_shift_m']
    original_base = checker.base_transform.copy()
    if 'T_world_base' in cfg:
        checker.set_base(np.asarray(cfg['T_world_base']))
    reserve_joint_limits(checker, cfg['joint_reserve_rad'])
    checker.narrow_limit('idx62_arm_r_joint2', np.deg2rad(cfg['right_joint2_range_deg']))
    for name, limits in cfg.get('waist_limits_rad', {}).items():
        checker.narrow_limit(name, limits)
    seed = {**cfg['seed_joints'], 'idx44_hand_l_ring_mcp_joint': cfg['left_ring_mcp_rad']}
    checker.set_state(seed, battery)
    names = checker.names['right'] + checker.names['left'] + checker.names['waist']
    ids = np.array([checker.model.joint(n).qposadr[0] for n in names])
    low, high = np.array([checker.model.joint(n).range for n in names]).T
    reference = checker.data.qpos.copy()
    initial = reference[ids].copy()
    # Explicit torso/elbow separation helps the optimizer avoid inward-folded
    # branches. The final independent check still covers every collision pair.
    torso_pairs = [(a, b) for a, b in checker.pairs
                   if any(checker.model.body(int(checker.model.geom_bodyid[g])).name.startswith('body_link') for g in (a, b))]

    def residual(values):
        state = reference.copy()
        state[ids] = values
        checker.c.update(state)
        errors = []
        for side in 'lr':
            actual = checker.c.get_transform_frame_to_world(f'arm_{side}_end_link', 'body').as_matrix()
            errors.extend(1000 * (actual[:3, 3] - goals[side][:3, 3]))
            errors.extend(300 * Rotation.from_matrix(goals[side][:3, :3].T @ actual[:3, :3]).as_rotvec())
            dz = checker.data.body(f'arm_{side}_link4').xpos[2] - checker.data.body(f'arm_{side}_link3').xpos[2]
            errors.append(1000 * max(0., dz + cfg['minimum_elbow_below_shoulder_m']))
        errors.extend(.01 * (values - initial))
        errors.append(values[1] - np.deg2rad(cfg['right_joint2_target_deg']))
        errors.extend(.12 * values[-2:])
        if 'waist_pitch_sum_limit_rad' in cfg:
            errors.append(1000 * max(0., abs(sum(values[14:17])) - cfg['waist_pitch_sum_limit_rad']))
        for a, b in torso_pairs:
            distance = mujoco.mj_geomDistance(checker.model, checker.data, a, b, .02, None)
            errors.append(2000 * max(0., .005 - distance))
        return np.asarray(errors)

    solution = least_squares(residual, initial, bounds=(low, high), max_nfev=400,
                             ftol=1e-9, xtol=1e-9, gtol=1e-9)
    residual(solution.x)
    joints = checker.current_joints()
    checker.set_state(joints, battery)  # Checks limits, mimics and independent URDF FK.
    collision = checker.collisions(['left', 'right'])
    after = metrics(checker)
    errors = {}
    for side in 'lr':
        actual = checker.c.get_transform_frame_to_world(f'arm_{side}_end_link', 'body').as_matrix()
        errors[side] = dict(position_m=float(np.linalg.norm(actual[:3, 3] - goals[side][:3, 3])),
                            rotation_rad=float(Rotation.from_matrix(goals[side][:3, :3].T @ actual[:3, :3]).magnitude()))
    if any(e['position_m'] > .0005 or e['rotation_rad'] > .0005 for e in errors.values()):
        raise ValueError(f'Wrist targets not reached: {errors}')
    if any(e['below_shoulder_m'] < cfg['minimum_elbow_below_shoulder_m'] for e in after['elbows'].values()):
        raise ValueError('Elbow height requirement failed')
    if not after['wrap']['passed'] or after['ring_longitudinal_midpoint_error_m'] > .001:
        raise ValueError('Wrap / ring midpoint requirement failed')
    ring = after['wrap']['geometry']['ring']
    if abs(ring['distance_m']) > .001:
        raise ValueError('Ring finger remains separated from shell')
    pairs = np.asarray(checker.pairs + [(g, e) for g in checker.robot for e in checker.env])
    a, b = pairs.T
    near = np.linalg.norm(checker.data.geom_xpos[a] - checker.data.geom_xpos[b], axis=1) < (
        checker.model.geom_rbound[a] + checker.model.geom_rbound[b] + .1)
    clearance, closest = .1, None
    for a, b in pairs[near]:
        distance = float(mujoco.mj_geomDistance(checker.model, checker.data, a, b, .1, None))
        if distance < clearance:
            clearance = distance
            closest = [checker.model.body(int(checker.model.geom_bodyid[g])).name for g in (a, b)]
    pitch_sum = sum(joints[n] for n in checker.names['waist'][:3])
    if 'waist_pitch_sum_limit_rad' in cfg and abs(pitch_sum) > cfg['waist_pitch_sum_limit_rad'] + 1e-6:
        raise ValueError('Torso pitch requirement failed')
    out = ROOT / cfg['output']
    out.mkdir(parents=True, exist_ok=True)
    result = dict(passed=True, scope='static grasp endpoint after right partial drag',
                  config=str(config_path.relative_to(ROOT)), joints=joints,
                  joints_deg={n: float(np.rad2deg(v)) for n, v in joints.items()},
                  baseline_joints=baseline, before=before, after=after,
                  right_joint2_range_deg=cfg['right_joint2_range_deg'],
                  left_forward_shift_m=cfg['left_forward_shift_m'],
                  right_wrist_target_unchanged=True, wrist_target_errors=errors,
                  base=checker.base_transform.tolist(), payload=battery.tolist(),
                  baseline_base=original_base.tolist(),
                  noncontact_robot_min_clearance_m=clearance, closest_noncontact_pair=closest,
                  waist_pitch_sum_deg=float(np.rad2deg(pitch_sum)),
                  station_change=dict(translation_world_m=(checker.base_transform[:3, 3]-original_base[:3, 3]).tolist(),
                      heading_deg=float(np.rad2deg(np.arctan2(checker.base_transform[1, 0], checker.base_transform[0, 0]))),
                      path_planned=False),
                  collision=collision, collision_metadata=checker.metadata,
                  transition_validated=False, navigation_updated=False,
                  dynamics_validated=False, hardware_commands_sent=False,
                  ring_contact_description='Longitudinal midpoint at lower long edge; ring is not counted as a deep side-wrap contact')
    (out / 'grasp_pose.json').write_text(json.dumps(result, indent=2))
    (out / 'grasp_pose.yaml').write_text(yaml.safe_dump(dict(joints=joints, executable=False,
                        T_world_base=result['base'], T_world_payload=result['payload']), sort_keys=False))
    print(json.dumps(dict(passed=True, before=before['elbows'], after=after['elbows'],
                         right_joint2_deg=after['right_joint2_deg'], ring=ring,
                         wrist_target_errors=errors), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'simulation/grasp_elbow_down.yaml')
    main(parser.parse_args().config.resolve())
