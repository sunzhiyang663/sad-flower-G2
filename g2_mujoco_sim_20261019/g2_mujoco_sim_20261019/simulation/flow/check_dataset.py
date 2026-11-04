"""Sampled 3D clearance audit of converted routes; no hardware IO."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from simulation.map9_route_preview import Clearance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='simulation/flow/data/converted')
    parser.add_argument('--task', default='simulation/map9_spread_forward5_full.yaml')
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT/'simulation/map9.yaml').read_text(encoding='utf-8'))
    task = yaml.safe_load((ROOT/args.task).read_text(encoding='utf-8'))
    out, data_dir = ROOT/task['output'], ROOT/args.data
    # Portable editable scene is used instead of a version-specific MJB binary.
    scene = ROOT/'scenes/map9_current.xml'
    model = mujoco.MjModel.from_xml_path(str(scene))
    with np.load(out/'state.npz') as record:
        saved = {k: record[k].copy() for k in record.files}
    with np.load(out/'pickup.npz') as pickup:
        empty_q = pickup['default_qpos'].copy()
        loaded_q = pickup['qpos'][-1].copy()
        payload = pickup['payload'][-1].copy()
        initial_payload = pickup['payload'][0].copy()
    with np.load(out/f'route_G{task["start_guide_id"]}_G4.npz') as empty_route:
        station = empty_route['poses'][-1]
    station_transform = np.eye(4)
    station_transform[:3, :3] = Rotation.from_euler('z', station[2]).as_matrix()
    station_transform[:3, 3] = [*station[:2], cfg['base_height_m']]
    shell_world = station_transform @ initial_payload
    threshold = task['motion']['min_navigation_clearance_m']
    reports = {}
    sources = sorted(data_dir.glob('route_*.npz'))
    if not sources:
        raise FileNotFoundError('No converted routes found')
    for source in sources:
        with np.load(source) as record:
            states = record['states'].copy()
            loaded = bool(record['loaded'])
        state = dict(saved)
        state['qpos'] = loaded_q if loaded else empty_q
        if loaded:
            state['T_base_payload'] = payload
        checker = Clearance(model, state, cfg['base_height_m'], .1)
        minima = np.full(4, .1)
        shell_min = .1
        failures, count = [], 0
        for k in range(len(states)-1):
            delta = states[k+1]-states[k]
            subdivisions = max(1, int(np.ceil(np.linalg.norm(delta[:2])/.01)),
                               int(np.ceil(abs(delta[2])/.02)))
            # Include the final endpoint once, and all intermediate samples.
            fractions = np.arange(subdivisions)/subdivisions
            if k == len(states)-2:
                fractions = np.r_[fractions, 1.]
            for fraction in fractions:
                p = states[k]+fraction*delta
                values, witness = checker.check(p, include_payload=loaded)
                minima = np.minimum(minima, values)
                stationary_shell = .1
                if not loaded:
                    # Empty navigation must also avoid the ungrasped object.
                    T = shell_world
                    checker.data.mocap_pos[0] = T[:3, 3]
                    checker.data.mocap_quat[0] = Rotation.from_matrix(T[:3, :3]).as_quat(scalar_first=True)
                    mujoco.mj_forward(model, checker.data)
                    for geom in checker.robot:
                        stationary_shell = min(stationary_shell, float(mujoco.mj_geomDistance(
                            model, checker.data, geom, checker.payload, .1, None)))
                    shell_min = min(shell_min, stationary_shell)
                if values.min() < threshold or stationary_shell < threshold:
                    failures.append(dict(segment=k, fraction=float(fraction),
                                         distances_m=values.tolist(),
                                         stationary_shell_m=stationary_shell, witness=witness))
                count += 1
        reports[source.name] = dict(passed=not failures, loaded=loaded, samples=count,
            minimum_by_category_m=minima.tolist(), stationary_shell_minimum_m=shell_min if not loaded else None,
            threshold_m=threshold, failure_count=len(failures), failures=failures[:20],
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        print(source.name, 'PASS' if not failures else 'FAIL', count, minima, flush=True)
    report = dict(passed=all(r['passed'] for r in reports.values()), routes=reports,
                  scene=str(scene), scene_sha256=hashlib.sha256(scene.read_bytes()).hexdigest(),
                  mujoco_version=mujoco.__version__, translation_step_m=.01, angle_step_rad=.02,
                  continuous_collision_detection=False, dynamics_validated=False)
    target = data_dir/'collision_report.json'
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(target)
    if not report['passed']:
        raise SystemExit('Some converted routes failed collision clearance checks')


if __name__ == '__main__':
    main()
