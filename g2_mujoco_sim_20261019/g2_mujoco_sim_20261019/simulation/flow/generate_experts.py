"""Collect checked OMPL expert episodes on the bundled scene, with restart support."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import mujoco
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from simulation.flow.collection_scene import CollectionScene
from simulation.flow.prepare_dataset import convert


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path,value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2),encoding='utf-8')
    temp.replace(path)


def split_plan(count,seed):
    order = np.random.default_rng(seed).permutation(count)
    n = max(1,count//10)
    splits = ['train']*count
    for i in order[:n]:
        splits[i] = 'test'
    for i in order[n:2*n]:
        splits[i] = 'validation'
    return splits


def collect(scene,plan,settings,output,resume=False,max_attempts=1000):
    """Planner injection permits independent restart tests without OMPL."""
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    lock = output/'.collection.lock'
    try:
        descriptor = os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError('Collection lock exists. Verify no process is running before removing .collection.lock')
    os.write(descriptor,str(os.getpid()).encode())
    os.close(descriptor)
    try:
        path = output/'manifest.json'
        if resume:
            manifest = json.loads(path.read_text(encoding='utf-8'))
            if manifest['settings'] != settings:
                raise ValueError('Resume settings/input hashes differ; keep original settings or choose a new output')
            for entry in manifest['episodes']:
                if digest(output/entry['file']) != entry['sha256']:
                    raise ValueError(f'Corrupt/missing episode: {entry["file"]}')
        else:
            if any(p.name!='.collection.lock' for p in output.iterdir()):
                raise ValueError('Output not empty; use --resume or a new output')
            manifest = dict(schema_version=2,settings=settings,episodes=[],attempts=[],complete=False,
                continuous_collision_detection=False,dynamics_validated=False,
                scope='Fixed static scene; single payload mode per dataset; grid footprint and sampled whole-body/payload geometry')
        splits = split_plan(settings['count'],settings['seed'])

        def persist():
            episodes,attempts = manifest['episodes'],manifest['attempts']
            manifest.update(requested=settings['count'],accepted=len(episodes),complete=len(episodes)==settings['count'])
            manifest['statistics'] = dict(attempts=len(attempts),accepted=len(episodes),
                rejection_reasons=dict(Counter(a['reason'] for a in attempts if a['status']=='rejected')),
                split_counts=dict(Counter(e['split'] for e in episodes)),
                recorded_attempt_seconds=sum(a['elapsed_s'] for a in attempts),
                total_states=sum(e['states'] for e in episodes),
                total_physical_seconds=sum(e['duration_s'] for e in episodes))
            write_json(path,manifest)
        persist()
        for attempt in range(len(manifest['attempts']),max_attempts):
            if manifest['complete']:
                return manifest
            # Same attempted endpoint task after restart; OMPL internals need not replay bitwise.
            rng = np.random.default_rng(np.random.SeedSequence([settings['seed'],attempt]))
            entry = dict(attempt=attempt)
            clock = time.monotonic()
            try:
                start,goal = scene.sample_pose(rng),scene.sample_pose(rng)
                entry.update(start=start.tolist(),goal=goal.tolist())
                distance = np.linalg.norm(start[:2]-goal[:2])
                if not settings['min_distance']<=distance<=settings['max_distance']:
                    raise ValueError('endpoint_distance_range')
                poses = plan(start,goal,scene.valid)
                states,actions,times,metrics = convert(poses,settings['dt'],settings['speed'],
                    settings['reverse_speed'],settings['yaw_speed'])
                if len(states)<settings['min_states']:
                    raise ValueError('too_short_for_training_window')
                if times[-1]>settings['max_duration']:
                    raise ValueError('duration_limit')
                if np.linalg.norm(states[0,:2]-start[:2])>1e-5 or np.linalg.norm(states[-1,:2]-goal[:2])>1e-5:
                    raise ValueError('planner_endpoint_position_mismatch')
                angle_error = states[[0,-1],2]-np.array([start[2],goal[2]])
                if np.max(np.abs(np.arctan2(np.sin(angle_error),np.cos(angle_error))))>1e-5:
                    raise ValueError('planner_endpoint_angle_mismatch')
                audit = scene.audit(states,settings['translation_step'],settings['angle_step'])
                if not audit['passed']:
                    entry['audit'] = audit
                    raise ValueError('resampled_'+audit['reason'])
                index = len(manifest['episodes'])
                filename = f'episode_{index:06d}.npz'
                temporary = output/'episode_pending.npz'
                np.savez_compressed(temporary,states=states,actions=actions,timestamps=times,
                    dt=settings['dt'],start=states[0],goal=states[-1],loaded=settings['mode']=='loaded',
                    original_poses=poses,requested_start=start,requested_goal=goal)
                temporary.replace(output/filename)
                manifest['episodes'].append(dict(file=filename,split=splits[index],sha256=digest(output/filename),
                    attempt=attempt,loaded=settings['mode']=='loaded',**dict(metrics,collision_checked=True),audit=audit))
                entry.update(status='accepted',file=filename)
            except (ValueError,RuntimeError) as exc:
                entry.update(status='rejected',reason=str(exc))
            entry['elapsed_s'] = time.monotonic()-clock
            manifest['attempts'].append(entry)
            persist()
            print(f'Attempt {attempt}: {entry["status"]}; accepted {manifest["accepted"]}/{settings["count"]}',flush=True)
        return manifest
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['empty','loaded'],default='empty')
    parser.add_argument('--motion',choices=['forward','reeds-shepp'],default='forward')
    parser.add_argument('--count',type=int,default=10)
    parser.add_argument('--seed',type=int,default=20261103)
    parser.add_argument('--dt',type=float,default=.1)
    parser.add_argument('--max-attempts',type=int,default=1000,help='Total cumulative attempt budget, including resumed attempts')
    parser.add_argument('--planning-seconds',type=float,default=5.)
    parser.add_argument('--min-distance',type=float,default=1.)
    parser.add_argument('--max-distance',type=float,default=12.)
    parser.add_argument('--max-duration',type=float,default=120.)
    parser.add_argument('--min-states',type=int,default=64)
    parser.add_argument('--clearance',type=float,default=.003)
    parser.add_argument('--translation-step',type=float,default=.01)
    parser.add_argument('--angle-step',type=float,default=.02)
    parser.add_argument('--output',default='simulation/flow/data/experts_pilot')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--preflight',action='store_true',help='Inspect packaged scene/grid and endpoints without OMPL')
    args = parser.parse_args()
    if args.count<3 or args.min_states<2 or args.max_attempts<1 or min(args.dt,args.planning_seconds,
            args.min_distance,args.max_duration,args.translation_step,args.angle_step)<=0 or args.max_distance<args.min_distance:
        parser.error('Invalid collection counts, distances, duration or steps')
    scene = CollectionScene(args.mode,args.clearance)
    if args.preflight:
        p = scene.sample_pose(np.random.default_rng(args.seed))
        print(json.dumps(dict(mode=args.mode,grid_shape=list(scene.grid.shape),free_cells=len(scene.free_centers),
            valid_pose=p.tolist(),inspection=scene.inspect(p),mujoco=mujoco.__version__),indent=2))
        return
    try:
        from ompl import util as ou
        from simulation.plan_map9_vehicle import plan_base_path
    except ImportError as exc:
        raise SystemExit('OMPL bindings unavailable; use the Linux collection environment. No episodes generated.') from exc
    ou.RNG.setSeed(args.seed)
    try:
        ompl_version = importlib.metadata.version('ompl')
    except importlib.metadata.PackageNotFoundError:
        ompl_version = 'source-build-without-package-metadata'
    task = yaml.safe_load((ROOT/'simulation/map9_spread_forward5_full.yaml').read_text(encoding='utf-8'))
    vehicle = dict(task['vehicle'],planning_time_s=args.planning_seconds,simplify_time_s=.5)
    settings = {k:v for k,v in vars(args).items() if k not in ('output','resume','preflight','max_attempts')}
    settings.update(speed=vehicle['speed_m_s'],reverse_speed=vehicle['reverse_speed_m_s'],
        yaw_speed=vehicle['max_yaw_speed_rad_s'],turning_radius=vehicle['turning_radius_m'],
        sample_step=vehicle['sample_step_m'],mujoco_version=mujoco.__version__,
        ompl_version=ompl_version,
        input_hashes={p.relative_to(ROOT).as_posix():digest(p) for p in scene.input_paths},
        code_hashes={name:digest(Path(__file__).parent/name) for name in
                     ('generate_experts.py','collection_scene.py','prepare_dataset.py')},
        planner_code_hashes={name:digest(ROOT/'simulation'/name) for name in
                            ('plan_map9_vehicle.py','map9_route_preview.py')})
    def planner(start,goal,valid):
        return plan_base_path(start,goal,valid,vehicle,scene.bounds,forward_only=args.motion=='forward')
    manifest = collect(scene,planner,settings,ROOT/args.output,args.resume,args.max_attempts)
    print(json.dumps(manifest['statistics'],indent=2))
    if not manifest['complete']:
        raise SystemExit('Attempt budget exhausted: partial data retained. Resume with a larger --max-attempts.')


if __name__ == '__main__':
    main()
