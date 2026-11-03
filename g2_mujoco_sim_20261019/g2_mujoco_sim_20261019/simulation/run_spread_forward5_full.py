"""Reproduce the forward-5-cm full MuJoCo transfer. No robot connection."""
import json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    env={**os.environ,'OPENBLAS_NUM_THREADS':'1','G2_SIM_TASK':'simulation/map9_spread_forward5_full.yaml'}
    steps=[('prepare_spread_forward5.py',),('plan_spread_bilateral.py','--config','simulation/grasp_spread_forward5.yaml'),
        ('prepare_map9_manipulation.py',),('plan_map9_vehicle.py',),('audit_map9_preposition.py',),
        ('audit_spread_full_motion.py',),('prepare_map9_workzone_cleanup.py',),('validate_map9_full_task.py',)]
    for script,*args in steps:
        print('Running',script,flush=True);subprocess.run([sys.executable,str(ROOT/'simulation'/script),*args],cwd=ROOT,env=env,check=True)
    out=ROOT/'simulation/outputs/map9_spread_forward5_full'
    if not json.loads((out/'validation.json').read_text())['passed']:raise RuntimeError('Full validation failed; do not render success video')
    for goal in (1,5):subprocess.run([sys.executable,str(ROOT/'simulation/render_map9_full_task.py'),'--goal',str(goal)],cwd=ROOT,env=env,check=True)
    subprocess.run([sys.executable,str(ROOT/'simulation/render_map9_grasp_closeup.py')],cwd=ROOT,env=env,check=True)
    subprocess.run([sys.executable,str(ROOT/'simulation/render_spread_bilateral.py'),'--output','simulation/outputs/grasp_spread_forward5'],cwd=ROOT,env=env,check=True)
    print(out,flush=True)
if __name__=='__main__':main()
