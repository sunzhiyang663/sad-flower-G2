"""Train a local G2 CFM baseline; --overfit-windows 4 is a pipeline check.

All relative paths resolve against the bundled G2 project root.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='simulation/flow/data/converted')
    parser.add_argument('--output', default='simulation/flow/outputs/overfit')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--horizon', type=int, default=64)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', help='Continue model/optimizer in a NEW output directory; not bitwise replay')
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--ode-steps', type=int, default=100)
    parser.add_argument('--overfit-windows', type=int, default=4,
                        help='Use only the first K training windows; 0 uses all')
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.width, args.depth, args.heads,
           args.log_every, args.ode_steps) < 1 or args.lr <= 0 or args.overfit_windows < 0:
        parser.error('Counts/lr must be positive; overfit-windows must be nonnegative')
    try:
        import numpy as np
        import torch
        from torch.utils.data import DataLoader, Subset
        from simulation.flow.dataset import G2TrajectoryDataset
        from simulation.flow.model import G2Transformer
        from simulation.flow.flow_matcher import G2FlowMatcher
    except ImportError as exc:
        raise SystemExit(f'Training dependency missing: {exc}. Install PyTorch and NumPy in the project environment.') from exc
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    dataset = G2TrajectoryDataset(ROOT/args.data, 'train', args.horizon, args.stride)
    validation = G2TrajectoryDataset(ROOT/args.data, 'validation', args.horizon, args.stride)
    selected = Subset(dataset, range(min(args.overfit_windows, len(dataset)))) if args.overfit_windows else dataset
    loader = DataLoader(selected, batch_size=args.batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False, num_workers=0)
    output = ROOT/args.output
    if output.exists() and any(output.iterdir()):
        raise SystemExit('Output is not empty. Choose a new --output to preserve earlier results.')
    output.mkdir(parents=True, exist_ok=True)
    dataset.save_normalizer(output/'normalizer.json')
    model_config = dict(horizon=args.horizon, width=args.width, depth=args.depth, heads=args.heads)
    model = G2Transformer(**model_config).to(device)
    matcher = G2FlowMatcher(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    manifest = dataset.manifest_path
    config = dict(vars(args), resolved_device=str(device), training_windows=len(selected),
        validation_windows=len(validation), physical_dt_s=dataset.dt,
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        architecture='Local torch Transformer baseline; not exact SAD-Flower architecture')
    start_step = 0
    if args.resume:
        previous = torch.load(ROOT/args.resume, map_location=device, weights_only=True)
        if previous['model_config'] != model_config or previous['config']['manifest_sha256'] != config['manifest_sha256']:
            raise ValueError('Resume architecture or dataset manifest differs')
        if previous['config']['overfit_windows'] != args.overfit_windows or previous['config']['stride'] != args.stride:
            raise ValueError('Resume must retain training-window selection and stride')
        if not np.allclose(previous['normalizer']['mean'], dataset.mean) or not np.allclose(previous['normalizer']['scale'], dataset.scale) or not np.isclose(previous['dt'],dataset.dt):
            raise ValueError('Resume normalization or physical dt differs')
        model.load_state_dict(previous['model'])
        optimizer.load_state_dict(previous['optimizer'])
        start_step = previous['step']
        if args.steps <= start_step:
            raise ValueError('--steps must exceed the saved step (total target steps)')
    (output/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')

    def move(batch):
        return (batch.trajectories.to(device), {k: v.to(device) for k, v in batch.conditions.items()},
                batch.loss_mask.to(device))

    def evaluate(loader):
        # Fixed random draws make measurements comparable without changing training RNG.
        with torch.random.fork_rng(devices=[device.index or 0] if device.type == 'cuda' else []):
            torch.manual_seed(args.seed+1)
            model.eval()
            total, count = 0., 0
            with torch.no_grad():
                for batch in loader:
                    loss, _ = matcher.loss(*move(batch))
                    total += loss.item()*len(batch.trajectories)
                    count += len(batch.trajectories)
            model.train()
            return total/count

    probe_loader = DataLoader(selected, batch_size=args.batch_size, shuffle=False)
    iterator = iter(loader)
    with (output/'metrics.jsonl').open('w', encoding='utf-8') as log:
        initial = dict(step=start_step, train_probe=evaluate(probe_loader), validation=evaluate(validation_loader))
        best_validation = initial['validation']
        log.write(json.dumps(initial)+'\n')
        print(initial, flush=True)
        for step in range(start_step+1, args.steps+1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = matcher.loss(*move(batch))
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss at step {step}')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            if step % args.log_every == 0 or step == args.steps:
                metrics = dict(step=step, loss=loss.item(), gradient_norm=float(norm),
                    train_probe=evaluate(probe_loader), validation=evaluate(validation_loader))
                log.write(json.dumps(metrics)+'\n')
                log.flush()
                print(metrics, flush=True)
                saved = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    model_config=model_config, normalizer=dict(mean=dataset.mean.tolist(), scale=dataset.scale.tolist()),
                    dt=dataset.dt, step=step, config=config)
                # Replace atomically so an interrupted write cannot corrupt latest.pt.
                torch.save(saved, output/'latest.tmp')
                (output/'latest.tmp').replace(output/'latest.pt')
                if metrics['validation'] < best_validation or not (output/'best.pt').exists():
                    best_validation = metrics['validation']
                    torch.save(saved, output/'best.pt')
    # Fixed endpoints and padded actions during Euler integration over tau in [0, 1].
    model.eval()
    sample = dataset[0]
    with torch.no_grad():
        x = torch.randn(1, args.horizon, 6, device=device)
        for index, state in sample.conditions.items():
            x[:, index, 3:] = torch.as_tensor(state, device=device)
        x[:, -1, :3] = 0.
        for k in range(args.ode_steps):
            velocity = model(x, torch.full((1,), k/args.ode_steps, device=device))
            velocity[:, [0, -1], 3:] = 0.
            velocity[:, -1, :3] = 0.
            x += velocity/args.ode_steps
        if not torch.isfinite(x).all():
            raise RuntimeError('Nonfinite generated sample')
        generated = dataset.unnormalize(x[0].cpu().numpy())
    reference = dataset.unnormalize(sample.trajectories)
    states, actions = generated[:, 3:], generated[:-1, :3]
    theta = states[:-1, 2]
    world = np.column_stack([np.cos(theta)*actions[:, 0]-np.sin(theta)*actions[:, 1],
        np.sin(theta)*actions[:, 0]+np.cos(theta)*actions[:, 1], actions[:, 2]])
    residual = np.diff(states, axis=0)-dataset.dt*world
    report = dict(endpoint_error=float(np.max(np.abs(states[[0,-1]]-reference[[0,-1], 3:]))),
        reference_state_rmse=float(np.sqrt(np.mean((states-reference[:, 3:])**2))),
        max_euler_residual=float(np.max(np.abs(residual))),
        max_translation_speed=float(np.linalg.norm(actions[:, :2], axis=1).max()),
        max_yaw_speed=float(np.abs(actions[:, 2]).max()), collision_checked=False,
        note='Training-window diagnostic only; no planner or execution acceptance claimed')
    np.savez_compressed(output/'sample.npz', states=states, actions=actions,
        timestamps=np.arange(args.horizon)*dataset.dt, reference=reference)
    (output/'sample_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(report, flush=True)


if __name__ == '__main__':
    main()
