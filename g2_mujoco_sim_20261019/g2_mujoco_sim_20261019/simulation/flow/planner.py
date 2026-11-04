"""Plain FM candidate planner; filtering is separate from the learned flow."""
import numpy as np

from .fm_inference import load_model, sample, diagnose


class FMPlanner:
    def __init__(self,checkpoint,device='auto',steps=100,candidates=16,seed=42):
        self.model,self.saved,self.device = load_model(checkpoint,device)
        self.steps,self.candidates,self.seed = steps,candidates,seed
        self.calls = 0

    def plan(self,start,goal,scene,speed_limit=.45,yaw_limit=.45,
             position_tolerance=.02,angle_tolerance=.02,translation_step=.01,angle_step=.02):
        start,goal = np.asarray(start,dtype=float).copy(),np.asarray(goal,dtype=float).copy()
        if not scene.valid(start) or not scene.valid(goal):
            raise ValueError('Invalid FM planning endpoints')
        # Use the shortest continuous heading branch consistently with training unwrap.
        goal[2] = start[2]+np.arctan2(np.sin(goal[2]-start[2]),np.cos(goal[2]-start[2]))
        duration = (self.model.horizon-1)*self.saved['dt']
        if np.linalg.norm(goal[:2]-start[:2]) > speed_limit*duration+1e-6:
            raise ValueError(f'Goal exceeds the fixed {duration:.3f} s horizon; retrain with a longer horizon or provide intermediate tasks')
        q,u = sample(self.model,self.saved,start,goal,self.steps,self.seed+self.calls,self.candidates)
        self.calls += 1
        diagnostics,accepted = [],[]
        for index,(states,actions) in enumerate(zip(q,u)):
            metrics = diagnose(states,actions,self.saved['dt'],start,goal,speed_limit,yaw_limit)
            reasons = []
            if metrics['speed_limit_excess_m_s']>1e-6 or metrics['yaw_limit_excess_rad_s']>1e-6:
                reasons.append('action_bounds')
            if metrics['max_position_residual_m']>position_tolerance or metrics['max_angle_residual_rad']>angle_tolerance:
                reasons.append('kinematic_residual')
            if metrics['endpoint_position_error_m']>1e-5 or metrics['endpoint_angle_error_rad']>1e-5:
                reasons.append('endpoint')
            # Audit even rejected candidates so failures remain interpretable.
            audit = scene.audit(states,translation_step,angle_step)
            if not audit['passed']:
                reasons.append('collision_or_map')
            metrics.update(candidate=index,collision_checked=True,geometry_audit=audit,
                           accepted_for_execution=False,offline_reference_passed=not reasons,reasons=reasons)
            diagnostics.append(metrics)
            if not reasons:
                accepted.append(index)
        if not accepted:
            return None,dict(passed=False,candidates=diagnostics,reason='No candidate passed offline reference checks')
        best = min(accepted,key=lambda i:diagnostics[i]['path_length_m'])
        return dict(states=q[best],actions=u[best],dt=self.saved['dt']),dict(
            passed=True,selected=best,candidates=diagnostics,duration_s=duration,
            note='Plain FM + independent rejection/shortest-path selection; not CBF/CLF or closed-loop execution')
