"""Standalone plain-FM inference: endpoints only; no dataset needed."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--start', nargs=3, type=float, required=True, metavar=('X','Y','THETA'))
    parser.add_argument('--goal', nargs=3, type=float, required=True, metavar=('X','Y','THETA'))
    parser.add_argument('--loaded', action='store_true', help='Metadata only, not a model condition')
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--candidates', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--speed-limit', type=float, default=.45)
    parser.add_argument('--yaw-limit', type=float, default=.45)
    args = parser.parse_args()
    from simulation.flow.fm_inference import load_model, sample, diagnose, save_route
    checkpoint = ROOT/args.checkpoint
    model, saved, _ = load_model(checkpoint, args.device)
    output = ROOT/args.output
    if output.exists() and any(output.iterdir()):
        raise SystemExit('Choose an empty output directory')
    states, actions = sample(model, saved, args.start, args.goal, args.steps, args.seed, args.candidates)
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    for i, (q,u) in enumerate(zip(states,actions)):
        name = f'candidate_{i:03d}.npz'
        save_route(output/name, q,u,saved['dt'],args.loaded)
        reports.append(dict(file=name, **diagnose(q,u,saved['dt'],args.start,args.goal,args.speed_limit,args.yaw_limit)))
    report = dict(settings=vars(args), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        horizon=model.horizon, duration_s=(model.horizon-1)*saved['dt'], candidates=reports,
        note='All candidates are unfiltered plain FM. Loaded flag is metadata only. Goal heading uses the supplied continuous angle branch.')
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
