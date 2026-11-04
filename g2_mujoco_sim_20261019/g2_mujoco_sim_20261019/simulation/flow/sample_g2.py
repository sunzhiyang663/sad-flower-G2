"""Compare FM and action-box/soft-CLF sampling on the same noise and endpoints.

No obstacle CBF or planner acceptance is implemented here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from simulation.flow.model import G2Transformer
from simulation.flow.dataset import G2TrajectoryDataset
from simulation.flow.constraints import correct_velocity, residual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='simulation/flow/outputs/overfit_run1/latest.pt')
    parser.add_argument('--output', default='simulation/flow/outputs/constrained_run1')
    parser.add_argument('--split', choices=['train','validation','test'], default='train')
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--clf-gain', type=float, default=20.)
    parser.add_argument('--penalty', type=float, default=1000.)
    args = parser.parse_args()
    if args.steps < 1 or args.clf_gain <= 0 or args.penalty <= 0:
        parser.error('steps, clf-gain and penalty must be positive')
    checkpoint_path = ROOT/args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model = G2Transformer(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model'])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    data_dir = ROOT/checkpoint['config']['data']
    manifest_hash = hashlib.sha256((data_dir/'dataset_manifest.json').read_bytes()).hexdigest()
    if manifest_hash != checkpoint['config']['manifest_sha256']:
        raise ValueError('Dataset manifest differs from training checkpoint')
    dataset = G2TrajectoryDataset(data_dir, args.split, model.horizon)
    if not 0 <= args.index < len(dataset):
        raise ValueError('Window index out of range')
    mean = np.array(checkpoint['normalizer']['mean'])
    scale = np.array(checkpoint['normalizer']['scale'])
    if not np.allclose(mean, dataset.mean) or not np.allclose(scale, dataset.scale):
        raise ValueError('Normalization differs from checkpoint')
    sample = dataset[args.index]
    config = yaml.safe_load((ROOT/'simulation/map9_spread_forward5_full.yaml').read_text(encoding='utf-8'))
    v = config['vehicle']['speed_m_s']
    omega = config['vehicle']['max_yaw_speed_rad_s']
    # Conservative component box lies inside the translation-speed disk.
    high = np.array([v/np.sqrt(2), v/np.sqrt(2), omega])
    low = -high
    torch.manual_seed(args.seed)
    initial = torch.randn(model.horizon, 6)
    for k, state in sample.conditions.items():
        initial[k, 3:] = torch.as_tensor(state)
    initial[-1, :3] = 0.
    output = ROOT/args.output
    if output.exists() and any(output.iterdir()):
        raise SystemExit('Choose a new, empty --output directory')
    output.mkdir(parents=True, exist_ok=True)
    reports = {}
    for mode in ('fm', 'action_clf'):
        x = initial.clone()
        if mode == 'action_clf':
            x[:-1, :3] = torch.as_tensor(np.clip(x[:-1, :3].numpy(),
                (low-mean[:3])/scale[:3], (high-mean[:3])/scale[:3]), dtype=x.dtype)
        trace = []
        for k in range(args.steps):
            with torch.no_grad():
                nominal = model(x[None], torch.tensor([k/args.steps]))[0].numpy()
            nominal[[0,-1], 3:] = 0.
            nominal[-1, :3] = 0.
            if mode == 'action_clf':
                velocity, metrics = correct_velocity(x, nominal, mean, scale,
                    checkpoint['dt'], 1/args.steps, low, high, args.clf_gain, args.penalty)
                trace.append(dict(step=k, **metrics))
            else:
                velocity = nominal
            x = x+torch.as_tensor(velocity, dtype=x.dtype)/args.steps
            if not torch.isfinite(x).all():
                raise RuntimeError('Nonfinite sample')
        physical = x.double()*torch.as_tensor(scale)+torch.as_tensor(mean)
        states = physical[:, 3:].numpy()
        actions = physical[:-1, :3].numpy()
        error = residual(physical, checkpoint['dt']).numpy()
        reference = dataset.unnormalize(sample.trajectories)
        reports[mode] = dict(endpoint_error=float(np.abs(states[[0,-1]]-reference[[0,-1],3:]).max()),
            max_euler_residual=float(np.abs(error).max()), residual_l2=float(np.linalg.norm(error)),
            max_translation_speed=float(np.linalg.norm(actions[:,:2],axis=1).max()),
            max_yaw_speed=float(np.abs(actions[:,2]).max()),
            action_box_violation=float(max(0., (actions-high).max(), (low-actions).max())),
            collision_checked=False, accepted_for_execution=False)
        np.savez_compressed(output/f'{mode}.npz', states=states, actions=actions,
            timestamps=np.arange(model.horizon)*checkpoint['dt'], dt=checkpoint['dt'])
        (output/f'{mode}_trace.json').write_text(json.dumps(trace, indent=2), encoding='utf-8')
    report = dict(settings=vars(args), action_low=low.tolist(), action_high=high.tolist(),
        methods=reports, note='Local soft-CLF QP with exact Euler action bounds, not full SAD-Flower CBF/CLF. Constrained initialization clips actions from the shared random draw.')
    (output/'comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
