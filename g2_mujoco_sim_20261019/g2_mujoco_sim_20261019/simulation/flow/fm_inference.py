"""Plain FM inference and numerical diagnostics, without safety correction."""
from pathlib import Path

import numpy as np
import torch

from .model import G2Transformer


def load_model(checkpoint, device='cpu'):
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device)
    saved = torch.load(Path(checkpoint), map_location=device, weights_only=True)
    model = G2Transformer(**saved['model_config']).to(device)
    model.load_state_dict(saved['model'])
    model.eval()
    return model, saved, device


@torch.no_grad()
def sample(model, saved, start, goal, steps=100, seed=42, candidates=1):
    if steps < 1 or candidates < 1:
        raise ValueError('steps and candidates must be positive')
    device = next(model.parameters()).device
    mean = torch.tensor(saved['normalizer']['mean'], device=device, dtype=torch.float32)
    scale = torch.tensor(saved['normalizer']['scale'], device=device, dtype=torch.float32)
    endpoints = torch.as_tensor(np.array([start, goal]), device=device, dtype=torch.float32)
    if endpoints.shape != (2, 3) or not torch.isfinite(endpoints).all():
        raise ValueError('Endpoints must be finite [x, y, unwrapped_theta] poses')
    endpoints = (endpoints-mean[3:])/scale[3:]
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(candidates, model.horizon, 6, generator=generator, device=device)
    x[:, [0, -1], 3:] = endpoints
    x[:, -1, :3] = 0.
    for k in range(steps):
        velocity = model(x, torch.full((candidates,), k/steps, device=device))
        velocity[:, [0,-1], 3:] = 0.
        velocity[:, -1, :3] = 0.
        x = x+velocity/steps
    if not torch.isfinite(x).all():
        raise RuntimeError('FM generated nonfinite values')
    physical = (x.double()*scale.double()+mean.double()).cpu().numpy()
    # Padding is never returned as an executable action.
    return physical[:, :, 3:], physical[:, :-1, :3]


def diagnose(states, actions, dt, start, goal, speed_limit=.45, yaw_limit=.45):
    if dt <= 0 or speed_limit <= 0 or yaw_limit <= 0:
        raise ValueError('dt and limits must be positive')
    theta = states[:-1, 2]
    world = np.column_stack([np.cos(theta)*actions[:,0]-np.sin(theta)*actions[:,1],
        np.sin(theta)*actions[:,0]+np.cos(theta)*actions[:,1], actions[:,2]])
    error = np.diff(states, axis=0)-dt*world
    speed = np.linalg.norm(actions[:,:2], axis=1)
    angle_error = states[[0,-1],2]-np.asarray([start,goal])[:,2]
    return dict(endpoint_position_error_m=float(np.linalg.norm(
        states[[0,-1],:2]-np.asarray([start,goal])[:,:2], axis=1).max()),
        endpoint_angle_error_rad=float(np.abs(np.arctan2(np.sin(angle_error), np.cos(angle_error))).max()),
        max_position_residual_m=float(np.linalg.norm(error[:,:2],axis=1).max()),
        max_angle_residual_rad=float(np.abs(error[:,2]).max()),
        max_translation_speed_m_s=float(speed.max()), max_yaw_speed_rad_s=float(np.abs(actions[:,2]).max()),
        speed_limit_excess_m_s=float(max(0.,speed.max()-speed_limit)),
        yaw_limit_excess_rad_s=float(max(0., np.abs(actions[:,2]).max()-yaw_limit)),
        path_length_m=float(np.linalg.norm(np.diff(states[:,:2],axis=0),axis=1).sum()),
        collision_checked=False, accepted_for_execution=False)


def save_route(path, states, actions, dt, loaded):
    delta = np.diff(states, axis=0)
    dyaw = np.arctan2(np.sin(delta[:,2]),np.cos(delta[:,2]))
    heading = states[:-1,2]+dyaw/2
    np.savez_compressed(path, states=states, poses=states, actions=actions, dt=dt,
        timestamps=np.arange(len(states))*dt, start=states[0], goal=states[-1], loaded=loaded,
        ds=np.linalg.norm(delta[:,:2],axis=1), dyaw=dyaw,
        longitudinal=delta[:,0]*np.cos(heading)+delta[:,1]*np.sin(heading),
        lateral=-delta[:,0]*np.sin(heading)+delta[:,1]*np.cos(heading))
