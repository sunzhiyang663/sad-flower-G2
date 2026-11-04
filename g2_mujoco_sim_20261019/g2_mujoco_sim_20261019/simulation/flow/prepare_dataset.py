"""Convert saved geometric routes to uniformly timed state/action samples.

Run from the project root: python simulation/flow/prepare_dataset.py
Requires NumPy and PyYAML. This does not certify resampled collision clearance.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]


def convert(poses, dt, forward_speed, reverse_speed, yaw_speed):
    if dt <= 0 or min(forward_speed, reverse_speed, yaw_speed) <= 0:
        raise ValueError('Time step and speed limits must be positive')
    q = np.asarray(poses, dtype=float).copy()
    if q.ndim != 2 or q.shape[1] != 3 or len(q) < 2 or not np.isfinite(q).all():
        raise ValueError('Expected finite poses of shape (N, 3), N >= 2')
    q[:, 2] = np.unwrap(q[:, 2])
    delta = np.diff(q, axis=0)
    heading = q[:-1, 2] + delta[:, 2] / 2
    longitudinal = delta[:, 0]*np.cos(heading) + delta[:, 1]*np.sin(heading)
    limits = np.where(longitudinal >= 0, forward_speed, reverse_speed)
    durations = np.maximum(np.linalg.norm(delta[:, :2], axis=1)/limits,
                           np.abs(delta[:, 2])/yaw_speed)
    keep = np.r_[True, durations > 1e-12]
    q = q[keep]
    durations = durations[durations > 1e-12]
    if len(q) == 1:
        states = np.repeat(q, 2, axis=0)
        times = np.array([0., dt])
    else:
        source_times = np.r_[0., np.cumsum(durations)]
        # Stretch the full duration to an integer number of physical steps.
        steps = max(1, int(np.ceil(source_times[-1]/dt)))
        source_times *= steps*dt/source_times[-1]
        times = np.arange(steps+1)*dt
        states = np.column_stack([np.interp(times, source_times, q[:, i]) for i in range(3)])
    velocity = np.diff(states, axis=0)/dt
    c, s = np.cos(states[:-1, 2]), np.sin(states[:-1, 2])
    actions = np.column_stack([c*velocity[:, 0]+s*velocity[:, 1],
                               -s*velocity[:, 0]+c*velocity[:, 1], velocity[:, 2]])
    reconstructed = np.column_stack([c*actions[:, 0]-s*actions[:, 1],
                                     s*actions[:, 0]+c*actions[:, 1], actions[:, 2]])
    residual = np.diff(states, axis=0)-dt*reconstructed
    max_speed = float(np.linalg.norm(actions[:, :2], axis=1).max())
    max_yaw = float(np.abs(actions[:, 2]).max())
    if max_speed > max(forward_speed, reverse_speed)+1e-8 or max_yaw > yaw_speed+1e-8:
        raise ValueError('Resampled trajectory exceeds configured speed bounds')
    return states, actions, times, dict(
        states=len(states), physical_dt_s=dt, duration_s=float(times[-1]),
        max_translation_speed_m_s=max_speed, max_yaw_speed_rad_s=max_yaw,
        max_euler_residual=float(np.abs(residual).max()),
        collision_checked=False,
        note='Euler residual is by construction; independent collision and execution checks remain required.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', default='simulation/map9_spread_forward5_full.yaml')
    parser.add_argument('--dt', type=float, default=0.1)
    parser.add_argument('--output', default='simulation/flow/data/converted')
    args = parser.parse_args()
    task = yaml.safe_load((ROOT/args.task).read_text(encoding='utf-8'))
    vehicle = task['vehicle']
    sources = sorted((ROOT/task['output']).glob('route_*.npz'))
    if not sources:
        raise FileNotFoundError('No route_*.npz files in task output')
    output = ROOT/args.output
    output.mkdir(parents=True, exist_ok=True)
    report = {}
    for source in sources:
        with np.load(source, allow_pickle=False) as record:
            states, actions, timestamps, metrics = convert(
                record['poses'], args.dt, vehicle['speed_m_s'],
                vehicle['reverse_speed_m_s'], vehicle['max_yaw_speed_rad_s'])
            np.savez_compressed(output/source.name, states=states, actions=actions,
                                timestamps=timestamps, dt=args.dt,
                                start=states[0], goal=states[-1], loaded=record['loaded'])
        report[source.name] = dict(source=str(source), **metrics)
    (output/'conversion_report.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
