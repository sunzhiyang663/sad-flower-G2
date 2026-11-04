"""Endpoint-preserving velocity QP with box limits and a soft scalar CLF."""
import numpy as np
import torch


def residual(physical, dt):
    states, actions = physical[:, 3:], physical[:-1, :3]
    theta = states[:-1, 2]
    world = torch.stack((theta.cos()*actions[:, 0]-theta.sin()*actions[:, 1],
                         theta.sin()*actions[:, 0]+theta.cos()*actions[:, 1], actions[:, 2]), dim=-1)
    return states[1:]-states[:-1]-dt*world


def solve_box_clf(nominal, gradient, rhs, lower, upper, penalty=1000.):
    """Solve min .5||v-f||² + .5*rho*s²; a.v <= b+s, s>=0, l<=v<=u.

    The one CLF multiplier gives v=clip(f-lambda*a,l,u), s=lambda/rho.
    Bisection solves the monotone scalar KKT equation; no external QP package.
    """
    if penalty <= 0 or np.any(lower > upper):
        raise ValueError('Invalid QP bounds/penalty')
    def value(lam):
        v = np.clip(nominal-lam*gradient, lower, upper)
        return v, float(np.sum(gradient*v)-rhs-lam/penalty)
    velocity, violation = value(0.)
    if violation <= 0:
        return velocity, 0.
    lo, hi = 0., 1.
    while value(hi)[1] > 0:
        hi *= 2
        if hi > 1e20:
            raise RuntimeError('CLF multiplier failed to bracket')
    for _ in range(70):
        mid = (lo+hi)/2
        if value(mid)[1] > 0:
            lo = mid
        else:
            hi = mid
    return value(hi)[0], hi/penalty


def correct_velocity(x, nominal, mean, scale, dt, delta_tau, action_low,
                     action_high, gain=20., penalty=1000.):
    """All optimization variables are normalized generation velocities."""
    z = x.detach().double().requires_grad_(True)
    mu = torch.as_tensor(mean, dtype=z.dtype, device=z.device)
    sigma = torch.as_tensor(scale, dtype=z.dtype, device=z.device)
    error = residual(z*sigma+mu, dt)
    energy = .5*error.square().sum()
    gradient = torch.autograd.grad(energy, z)[0].detach().cpu().numpy()
    values = z.detach().cpu().numpy()
    lower, upper = np.full_like(values, -np.inf), np.full_like(values, np.inf)
    low = (np.asarray(action_low)-mean[:3])/scale[:3]
    high = (np.asarray(action_high)-mean[:3])/scale[:3]
    # Exact one-step Euler bounds, independent of the CLF relaxation.
    lower[:-1, :3] = (low-values[:-1, :3])/delta_tau
    upper[:-1, :3] = (high-values[:-1, :3])/delta_tau
    lower[[0, -1], 3:] = upper[[0, -1], 3:] = 0.
    lower[-1, :3] = upper[-1, :3] = 0.
    velocity, slack = solve_box_clf(nominal, gradient, -gain*energy.item(),
                                   lower, upper, penalty)
    if not np.isfinite(velocity).all():
        raise RuntimeError('Nonfinite QP correction')
    return velocity, dict(energy=energy.item(), clf_slack=slack,
        qp_constraint_error=max(0., float(np.sum(gradient*velocity)+gain*energy.item()-slack)))
