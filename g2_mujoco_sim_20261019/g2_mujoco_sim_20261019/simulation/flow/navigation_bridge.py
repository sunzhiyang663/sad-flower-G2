"""Timed FM trajectory adapter for the existing whole-task planner."""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
_PLANNERS = {}


class CallbackScene:
    def __init__(self,valid,bounds):
        self.callback,self.bounds = valid,bounds

    def valid(self,p):
        x0,x1,y0,y1 = self.bounds
        return bool(x0<=p[0]<=x1 and y0<=p[1]<=y1 and self.callback(p))

    def audit(self,states,translation_step=.01,angle_step=.02):
        samples = 0
        for index,(a,b) in enumerate(zip(states[:-1],states[1:])):
            n = max(1,int(np.ceil(np.linalg.norm(b[:2]-a[:2])/translation_step)),int(np.ceil(abs(b[2]-a[2])/angle_step)))
            for t in np.linspace(0.,1.,n+1):
                samples += 1
                if not self.valid(a+(b-a)*t):
                    return dict(passed=False,segment=index,fraction=float(t),samples=samples)
        return dict(passed=True,samples=samples)


def plan_flow(start,goal,valid,cfg,bounds,loaded):
    from .planner import FMPlanner
    flow = cfg['flow']
    checkpoint = ROOT/flow['loaded_checkpoint' if loaded else 'empty_checkpoint']
    key = (str(checkpoint),flow.get('device','auto'),flow.get('ode_steps',100),
           flow.get('candidates',16),cfg.get('random_seed',42))
    if key not in _PLANNERS:
        _PLANNERS[key] = FMPlanner(*key)
    trajectory,diagnostics = _PLANNERS[key].plan(start,goal,CallbackScene(valid,bounds),
        cfg['speed_m_s'],cfg['max_yaw_speed_rad_s'],flow.get('position_tolerance',.02),flow.get('angle_tolerance',.02))
    if trajectory is None:
        raise RuntimeError('FM rejected all candidates: '+str(diagnostics))
    return dict(poses=trajectory['states'],actions=trajectory['actions'],
        timestamps=np.arange(len(trajectory['states']))*trajectory['dt'],dt=trajectory['dt'],
        planner='flow',diagnostics=diagnostics)


def join(first,second):
    poses = second['poses'].copy()
    if not np.allclose(first['poses'][-1,:2],poses[0,:2],atol=1e-5):
        raise ValueError('Trajectory join positions differ')
    angle = first['poses'][-1,2]-poses[0,2]
    if abs(np.arctan2(np.sin(angle),np.cos(angle)))>1e-5:
        raise ValueError('Trajectory join headings differ')
    poses[:,2] += 2*np.pi*round(angle/(2*np.pi))
    result = dict(poses=np.vstack([first['poses'][:-1],poses]),planner=first['planner'],
                  diagnostics=[first['diagnostics'],second['diagnostics']],actions=None,timestamps=None,dt=None)
    if first['planner']=='flow':
        if not np.isclose(first['dt'],second['dt']):
            raise ValueError('Physical dt differs across FM segments')
        result.update(actions=np.vstack([first['actions'],second['actions']]),dt=first['dt'],
            timestamps=np.r_[first['timestamps'][:-1],first['timestamps'][-1]+second['timestamps']])
    return result


def audit_saved_route(record,vehicle):
    """Recompute timed reference checks, independently of planning-time reports."""
    from .fm_inference import diagnose
    q,u,t = record['poses'],record['actions'],record['timestamps']
    dt = float(record['dt'])
    if q.ndim!=2 or q.shape[1]!=3 or u.shape!=(len(q)-1,3) or t.shape!=(len(q),):
        raise ValueError('Malformed timed trajectory')
    if dt<=0 or not all(np.isfinite(v).all() for v in (q,u,t)) or not np.allclose(np.diff(t),dt):
        raise ValueError('Invalid trajectory values or physical time')
    metrics = diagnose(q,u,dt,record['expected_start'],record['expected_goal'],
                       vehicle['speed_m_s'],vehicle['max_yaw_speed_rad_s'])
    flow = vehicle['flow']
    passed = (metrics['endpoint_position_error_m']<=1e-5 and metrics['endpoint_angle_error_rad']<=1e-5
        and metrics['speed_limit_excess_m_s']<=1e-6 and metrics['yaw_limit_excess_rad_s']<=1e-6
        and metrics['max_position_residual_m']<=flow.get('position_tolerance',.02)
        and metrics['max_angle_residual_rad']<=flow.get('angle_tolerance',.02))
    return dict(passed=passed,**metrics)
