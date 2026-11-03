"""Replay saved offline kinematics in MuJoCo; never connects to hardware."""
import argparse,os,sys,time,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT);sys.path.insert(0,str(ROOT))
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--goal',type=int,choices=(1,5),default=1)
p.add_argument('--check-only',action='store_true')
p.add_argument('--speed',type=float,default=1.)
a=p.parse_args()
if a.speed<=0:p.error('--speed must be positive')
os.environ['MUJOCO_GL']='glfw'
import mujoco,numpy as np,yaml
from simulation.full_task_config import load_task
from simulation.render_map9_full_task import make_sequence
from simulation.map9_route_preview import Clearance
cfg=yaml.safe_load((ROOT/'simulation/map9.yaml').read_text());task=load_task();out=ROOT/task['output']
meta=json.loads((ROOT/cfg['snapshot']/'map.json').read_text())
sequence,timestamps,duration=make_sequence(a.goal,cfg,task,out,meta)
model=mujoco.MjModel.from_binary_path(str(out/'scene.mjb'));saved=dict(np.load(out/'state.npz'))
c=Clearance(model,saved,cfg['base_height_m'],.1)
def frame(k):
    pose,q,T,label,attached=sequence[k]
    c.data.qpos[:]=q;c.saved['T_base_payload']=T;c.set_pose(pose)
    return label
if a.check_only:
    for k in (0,len(sequence)//2,len(sequence)-1):
        frame(k);assert np.all(np.isfinite(c.data.geom_xpos))
    print(json.dumps({'passed':True,'goal':a.goal,'states':len(sequence),'duration_s':duration,'nq':model.nq,'ngeom':model.ngeom,'hardware_io':False}))
else:
    import mujoco.viewer
    with mujoco.viewer.launch_passive(model,c.data) as viewer:
        viewer.opt.geomgroup[:]=[0,1,1,1,1,0]
        viewer.cam.lookat[:]=[2,9,0.8];viewer.cam.distance=12;viewer.cam.azimuth=135;viewer.cam.elevation=-40
        start=time.monotonic()
        while viewer.is_running():
            t=((time.monotonic()-start)*a.speed)%duration
            k=min(len(sequence)-1,max(0,int(np.searchsorted(timestamps,t,side='right')-1)))
            with viewer.lock():frame(k)
            viewer.sync();time.sleep(1/60)
