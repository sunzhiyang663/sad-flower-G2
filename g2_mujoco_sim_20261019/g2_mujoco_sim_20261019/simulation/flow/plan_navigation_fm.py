"""Export timed G2 navigation tasks from plain FM, without modifying old outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks',required=True,help='JSON tasks with name, mode, start and goal')
    parser.add_argument('--empty-checkpoint')
    parser.add_argument('--loaded-checkpoint')
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='auto')
    parser.add_argument('--steps',type=int,default=100)
    parser.add_argument('--candidates',type=int,default=16)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--speed-limit',type=float,default=.45)
    parser.add_argument('--yaw-limit',type=float,default=.45)
    parser.add_argument('--position-tolerance',type=float,default=.02)
    parser.add_argument('--angle-tolerance',type=float,default=.02)
    args = parser.parse_args()
    from simulation.flow.collection_scene import CollectionScene
    from simulation.flow.planner import FMPlanner
    from simulation.flow.fm_inference import save_route
    tasks_path = ROOT/args.tasks
    tasks = json.loads(tasks_path.read_text(encoding='utf-8-sig'))['tasks']
    names = [t['name'] for t in tasks]
    if not tasks or len(names)!=len(set(names)) or any(Path(n).name!=n or n in ('.','..') for n in names):
        raise ValueError('Task names must be unique plain filenames')
    if min(args.steps,args.candidates,args.speed_limit,args.yaw_limit,args.position_tolerance,args.angle_tolerance)<=0:
        parser.error('Counts, limits and tolerances must be positive')
    output = ROOT/args.output
    if output.exists() and any(output.iterdir()):
        raise SystemExit('Choose an empty output directory')
    modes = set(t['mode'] for t in tasks)
    planners,scenes,hashes = {},{},{}
    for mode in modes:
        if mode not in ('empty','loaded'):
            raise ValueError(f'Unknown mode: {mode}')
        checkpoint = getattr(args,f'{mode}_checkpoint')
        if not checkpoint:
            parser.error(f'--{mode}-checkpoint is required for these tasks')
        checkpoint = ROOT/checkpoint
        hashes[mode] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        planners[mode] = FMPlanner(checkpoint,args.device,args.steps,args.candidates,args.seed)
        scenes[mode] = CollectionScene(mode)
    output.mkdir(parents=True,exist_ok=True)
    report = dict(settings=vars(args),checkpoint_hashes=hashes,tasks_sha256=hashlib.sha256(tasks_path.read_bytes()).hexdigest(),tasks=[])
    for task in tasks:
        try:
            route,diagnostics = planners[task['mode']].plan(task['start'],task['goal'],scenes[task['mode']],
                args.speed_limit,args.yaw_limit,args.position_tolerance,args.angle_tolerance)
            if route is not None:
                save_route(output/(task['name']+'.npz'),route['states'],route['actions'],route['dt'],task['mode']=='loaded')
        except (ValueError,RuntimeError) as exc:
            diagnostics = dict(passed=False,reason=str(exc))
        report['tasks'].append(dict(task=task,**diagnostics))
        report['passed'] = len(report['tasks'])==len(tasks) and all(t['passed'] for t in report['tasks'])
        (output/'navigation_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(task['name'],diagnostics['passed'],flush=True)
    if not report['passed']:
        raise SystemExit('Some FM tasks failed; no OMPL fallback was used. Inspect navigation_report.json.')


if __name__ == '__main__':
    main()
