"""Evaluate plain FM on held-out windows with fixed seeds; no safety filtering."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', default='simulation/flow/data/converted')
    parser.add_argument('--split', choices=['train','validation','test'], default='validation')
    parser.add_argument('--max-windows', type=int, default=20)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--candidates', type=int, default=4)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if min(args.max_windows,args.steps,args.candidates,args.stride)<1:
        parser.error('Window count, ODE steps, candidates and stride must be positive')
    import numpy as np
    import torch
    from simulation.flow.dataset import G2TrajectoryDataset
    from simulation.flow.fm_inference import load_model, sample, diagnose, save_route
    model, saved, _ = load_model(ROOT/args.checkpoint,args.device)
    dataset = G2TrajectoryDataset(ROOT/args.data,args.split,model.horizon,args.stride)
    if hashlib.sha256(dataset.manifest_path.read_bytes()).hexdigest() != saved['config']['manifest_sha256']:
        raise ValueError('Evaluation split manifest must match the training split definition')
    if not np.allclose(dataset.mean, saved['normalizer']['mean']) or not np.allclose(dataset.scale, saved['normalizer']['scale']):
        raise ValueError('Normalization mismatch')
    output = ROOT/args.output
    if output.exists() and any(output.iterdir()):
        raise SystemExit('Choose an empty output directory')
    output.mkdir(parents=True,exist_ok=True)
    reports,failures,timings = [],[],[]
    for index in range(min(args.max_windows,len(dataset))):
        name, offset = dataset.indices[index]
        reference = dataset.episodes[name][0][offset:offset+model.horizon]
        with np.load(dataset.directory/name) as record:
            loaded = bool(record['loaded'])
        if next(model.parameters()).is_cuda:torch.cuda.synchronize(next(model.parameters()).device)
        clock=time.perf_counter()
        try:
            states, actions = sample(model,saved,reference[0],reference[-1],args.steps,args.seed+index,args.candidates)
        except RuntimeError as exc:
            failures.append(dict(window=index,source=name,reason=str(exc),failed_candidates=args.candidates))
            continue
        if next(model.parameters()).is_cuda:torch.cuda.synchronize(next(model.parameters()).device)
        elapsed=time.perf_counter()-clock
        timings.append(dict(window=index,batch_seconds=elapsed,candidates=args.candidates,
                            amortized_seconds_per_candidate=elapsed/args.candidates))
        for candidate,(q,u) in enumerate(zip(states,actions)):
            filename = f'window_{index:04d}_candidate_{candidate:03d}.npz'
            save_route(output/filename,q,u,saved['dt'],loaded)
            metrics = diagnose(q,u,saved['dt'],reference[0],reference[-1])
            metrics['reference_position_rmse_m'] = float(np.sqrt(np.mean(np.sum((q[:,:2]-reference[:,:2])**2,axis=1))))
            metrics['reference_heading_rmse_rad'] = float(np.sqrt(np.mean(np.arctan2(np.sin(q[:,2]-reference[:,2]),np.cos(q[:,2]-reference[:,2]))**2)))
            reports.append(dict(file=filename, source=name, window_start=offset, loaded=loaded,seed=args.seed+index,**metrics))
        print(f'Window {index+1}: {args.candidates} candidates',flush=True)
    summary = {key:dict(mean=float(np.mean([r[key] for r in reports])),max=float(np.max([r[key] for r in reports])))
        for key in (reports[0] if reports else {}) if isinstance(reports[0][key],float)}
    report = dict(settings=vars(args),candidates=reports,summary=summary,
        generation_failures=failures,inference_timings=timings,
        requested_candidates=min(args.max_windows,len(dataset))*args.candidates,
        checkpoint_sha256=hashlib.sha256((ROOT/args.checkpoint).read_bytes()).hexdigest(),
        note='All candidates included; no best-of selection. Overlapping windows are correlated. No collision or execution success claim.')
    (output/'evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
