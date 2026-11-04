"""Reproduce the forward-5-cm full MuJoCo transfer. No robot connection."""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    from simulation.full_task_config import load_task
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',default=os.environ.get('G2_SIM_TASK','simulation/map9_spread_forward5_full.yaml'))
    parser.add_argument('--skip-render',action='store_true')
    args=parser.parse_args()
    task=load_task(args.task)
    planner=task['vehicle'].get('planner','ompl')
    if planner not in ('ompl','flow'):raise ValueError(f'Unknown planner: {planner}')
    if planner=='flow':
        for mode in ('empty','loaded'):
            if not (ROOT/task['vehicle']['flow'][mode+'_checkpoint']).is_file():
                raise FileNotFoundError(f'Missing {mode} FM checkpoint; configure vehicle.flow')
    env={**os.environ,'OPENBLAS_NUM_THREADS':'1','G2_SIM_TASK':args.task}
    steps=[('prepare_spread_forward5.py',),('plan_spread_bilateral.py','--config','simulation/grasp_spread_forward5.yaml'),
        ('prepare_map9_manipulation.py',),('plan_map9_vehicle.py',),('audit_map9_preposition.py',),
        ('audit_spread_full_motion.py',),('prepare_map9_workzone_cleanup.py',),('validate_map9_full_task.py',)]
    for script,*script_args in steps:
        print('Running',script,flush=True)
        try:
            subprocess.run([sys.executable,str(ROOT/'simulation'/script),*script_args],cwd=ROOT,env=env,check=True)
        except subprocess.CalledProcessError as exc:
            failure_out=ROOT/task['output'];failure_out.mkdir(parents=True,exist_ok=True)
            (failure_out/'execution_failure.json').write_text(json.dumps(dict(stage=script,returncode=exc.returncode,
                planner=planner,task=args.task),indent=2),encoding='utf-8')
            raise
    out=ROOT/task['output']
    if not json.loads((out/'validation.json').read_text())['passed']:raise RuntimeError('Full validation failed; do not render success video')
    if args.skip_render:
        print(out,flush=True);return
    for goal in task['goal_guide_ids']:subprocess.run([sys.executable,str(ROOT/'simulation/render_map9_full_task.py'),'--goal',str(goal)],cwd=ROOT,env=env,check=True)
    subprocess.run([sys.executable,str(ROOT/'simulation/render_map9_grasp_closeup.py')],cwd=ROOT,env=env,check=True)
    subprocess.run([sys.executable,str(ROOT/'simulation/render_spread_bilateral.py'),'--output','simulation/outputs/grasp_spread_forward5'],cwd=ROOT,env=env,check=True)
    print(out,flush=True)
if __name__=='__main__':main()
