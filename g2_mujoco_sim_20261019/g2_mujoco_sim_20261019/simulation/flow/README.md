# G2 dataset interface

For the complete plain-FM server workflow, read [SERVER_FM.md](SERVER_FM.md).
For expert data collection and restart commands, read [SERVER_COLLECTION.md](SERVER_COLLECTION.md).
For timed navigation route export and playback integration, read [NAVIGATION_FM.md](NAVIGATION_FM.md).
For multi-seed experiments, headless plots and summaries, read [EXPERIMENTS.md](EXPERIMENTS.md).
Its commands cover data checks, tests, training, continuation, independent
sampling and batch evaluation; they do not use the constrained sampling path.

All project-specific flow code lives here inside the `sad-flow/g2` Git repository.
`sad-flower-main` is reference source, not the destination for project changes.

`dataset.py` reads converted NPZ episodes and an explicit `dataset_manifest.json`.
Each entry contains `file`, `split`, and `sha256`. Split whole episodes before
extracting windows. Hash checks reject files changed after split creation.

Default windows have 64 states and 63 valid actions, stride 8, physical dt 0.1 s.
Channels are `[vx, vy, omega, x, y, theta]`. Endpoint conditions address the
state channels at rows 0 and 63. Heading must already be unwrapped.

Each sample is `G2Batch(trajectories, conditions, loss_mask)`. The final action
slot is zero in normalized coordinates. Its three loss entries, and the six
fixed endpoint-state loss entries, are masked out. Use
`G2FlowMatcher.loss(batch.trajectories, batch.conditions, batch.loss_mask)`.
The local `flow_matcher.py` implements linear CFM with zero noise scale.
During sampling, fix
endpoint-state derivatives and terminal-action padding derivatives to zero.
Only actions `[:-1]` participate in the dynamics residual and execution.

Normalization uses only full training episodes. `save_normalizer()` writes the
channel means, scales and dt; inference must use those saved statistics, never
recompute them from inference input. Validation values are not clipped.

Direct smoke check (does not require PyTorch or Gym):

```text
python simulation/flow/dataset.py simulation/flow/data/converted
```

The current three-route split is only for interface checks, not a meaningful
generalization benchmark. Train/validation are loaded routes and test is an
empty route. Before actual training/evaluation, expand and balance the data,
and either train separate payload modes or explicitly condition on that mode.
The dataset and local loss adapter do not import the reference repository or
its legacy Gym/D4RL environments. The dataset requires NumPy; the loss adapter
additionally requires PyTorch.

## Training entry point

`train_g2.py` and `model.py` are local to this repository. The network is a
small native PyTorch Transformer baseline, not an exact copy of SAD-Flower's
DiT architecture. No reference-repository imports or pretrained weights are used.
Training dependencies: NumPy and PyTorch (not OMPL, MuJoCo or Gym).

From the bundled project root:

```text
python simulation/flow/train_g2.py --steps 2 --log-every 1 --width 32 --depth 1 --output simulation/flow/outputs/smoke
python simulation/flow/train_g2.py --steps 2000 --overfit-windows 4 --device auto --output simulation/flow/outputs/overfit
```

Run the short check first. `--overfit-windows 0` uses all training windows.
Output directories must be empty, preventing accidental overwrite. Relative
paths always resolve against the bundled project root, not the shell directory.

Outputs: `config.json`, `normalizer.json`, `metrics.jsonl`, `latest.pt`,
`sample.npz`, `sample_report.json`. Checkpoints include model architecture,
weights, optimizer, normalization, physical dt and the dataset manifest hash.
They are saved at logging intervals and on the final step. `--resume` restores
model and optimizer in a new output directory (see SERVER_FM.md for limitations).
`best.pt` saves the best measured post-update validation checkpoint.
Fixed-probe training loss uses repeatable noise for comparison.

The final sample uses the first training window's endpoint conditions. Its
report measures endpoint error, reference state RMSE, Euler residual and speeds.
It is not collision-checked or accepted for execution. State RMSE combines
meters and radians and is only a debugging diagnostic. Neither loss reduction
nor this sample alone proves generalization or physical feasibility. Runtime
training and masked-gradient verification require PyTorch. Local CPU short
training, continuation, checkpoint loading, sampling and evaluation have passed;
server GPU execution remains to be checked.

## Constrained sampling

```text
python simulation/flow/sample_g2.py --output simulation/flow/outputs/constrained_run1
python simulation/flow/check_constraints.py
```

`constraints.py` minimizes squared change to the normalized FM velocity plus
a quadratic CLF slack penalty. It uses one scalar residual-energy CLF, fixed
endpoint velocities, and action boxes that constrain the next Euler iterate.
The scalar dual problem is solved by bisection. A soft CLF does not guarantee
zero residual or monotone residual decrease under finite integration steps.
This is a local baseline, not the full original SAD-Flower QP.

The translational component limit is `speed_m_s/sqrt(2)` in both directions,
so the box lies inside the speed disk. This is conservative and currently
does not implement a separate reverse-speed policy. Angular speed uses the
task limit. Limits are experimental configuration, not a verified robot API.
Action initialization is clipped into that box; endpoints are conditioned.
The FM baseline shares the same random draw but does not clip its actions.

The comparison saves physical trajectories, per-step QP diagnostics and a
summary. No obstacle CBF, collision checking or execution acceptance is
provided. Residual energy mixes meters and radians with equal numerical
weights. Tests compare the QP solution with SciPy SLSQP independently.
