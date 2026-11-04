"""NumPy-backed G2 windows, also compatible with torch DataLoader.

Run this file directly for a dependency-light dataset smoke check.
"""
import argparse
from collections import namedtuple
import hashlib
import json
from pathlib import Path

import numpy as np

G2Batch = namedtuple('G2Batch', 'trajectories conditions loss_mask')


class G2TrajectoryDataset:
    observation_dim = 3
    action_dim = 3

    def __init__(self, directory, split='train', horizon=64, stride=8, manifest=None):
        if horizon < 2 or stride < 1 or split not in ('train', 'validation', 'test'):
            raise ValueError('Invalid horizon, stride or split')
        self.directory = Path(directory)
        self.horizon = horizon
        manifest_path = Path(manifest) if manifest else self.directory/'dataset_manifest.json'
        if manifest is None and not manifest_path.exists():
            manifest_path = self.directory/'manifest.json'
        self.manifest_path = manifest_path
        definition = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
        names = [entry['file'] for entry in definition['episodes']]
        if len(names) != len(set(names)):
            raise ValueError('An episode may appear only once in the split manifest')
        self.episodes = {}
        training = []
        dts = []
        for entry in definition['episodes']:
            path = self.directory/entry['file']
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
                raise ValueError(f'Dataset hash mismatch: {path}')
            with np.load(path, allow_pickle=False) as record:
                states, actions = record['states'].copy(), record['actions'].copy()
                timestamps = record['timestamps'].copy()
                dt = float(record['dt'])
            if (states.ndim != 2 or states.shape[1] != 3 or
                    actions.shape != (len(states)-1, 3) or not np.isfinite(states).all()
                    or not np.isfinite(actions).all() or dt <= 0
                    or timestamps.shape != (len(states),)
                    or not np.allclose(np.diff(timestamps), dt)):
                raise ValueError(f'Invalid state/action/time alignment: {path}')
            if not np.allclose(np.diff(states[:, 2]),
                               np.arctan2(np.sin(np.diff(states[:, 2])), np.cos(np.diff(states[:, 2])))):
                raise ValueError(f'Heading must already be unwrapped: {path}')
            dts.append(dt)
            if entry['split'] == 'train':
                # Use actual actions only, never the terminal padding slot.
                training.append((states, actions))
            if entry['split'] == split:
                self.episodes[entry['file']] = (states, actions)
        if not training or not dts or not np.allclose(dts, dts[0]):
            raise ValueError('Training episodes required; physical dt must be consistent')
        self.dt = dts[0]
        # Deterministic statistics from training episodes only; identical across splits.
        state_data = np.concatenate([s for s, _ in training])
        action_data = np.concatenate([a for _, a in training])
        self.mean = np.r_[action_data.mean(0), state_data.mean(0)]
        self.scale = np.r_[action_data.std(0), state_data.std(0)]
        self.scale[self.scale < 1e-6] = 1.
        self.indices = []
        for name, (states, _) in self.episodes.items():
            starts = list(range(0, len(states)-horizon+1, stride))
            if starts and starts[-1] != len(states)-horizon:
                starts.append(len(states)-horizon)
            self.indices.extend((name, start) for start in starts)
        if not self.indices:
            raise ValueError(f'No full windows for split={split}, horizon={horizon}')

    def normalize(self, values):
        return (values-self.mean)/self.scale

    def unnormalize(self, values):
        return values*self.scale+self.mean

    def save_normalizer(self, path):
        Path(path).write_text(json.dumps(dict(mean=self.mean.tolist(), scale=self.scale.tolist(),
            channels=['vx', 'vy', 'omega', 'x', 'y', 'theta'], dt=self.dt), indent=2), encoding='utf-8')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        name, start = self.indices[index]
        states, actions = self.episodes[name]
        end = start+self.horizon
        values = np.zeros((self.horizon, 6), dtype=np.float64)
        values[:, 3:] = states[start:end]
        values[:-1, :3] = actions[start:end-1]
        normalized = self.normalize(values).astype(np.float32)
        normalized[-1, :3] = 0.  # Padding is zero in network coordinates.
        mask = np.ones_like(normalized)
        mask[-1, :3] = 0.
        mask[[0, -1], 3:] = 0.  # Fixed endpoint states have zero generation derivative.
        conditions = {0: normalized[0, 3:].copy(), self.horizon-1: normalized[-1, 3:].copy()}
        return G2Batch(normalized, conditions, mask)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    args = parser.parse_args()
    reports = {}
    for split in ('train', 'validation', 'test'):
        dataset = G2TrajectoryDataset(args.directory, split=split)
        sample = dataset[0]
        batch = np.stack([dataset[i].trajectories for i in range(min(4, len(dataset)))])
        name, start = dataset.indices[0]
        raw = dataset.unnormalize(sample.trajectories)
        states, actions = dataset.episodes[name]
        np.testing.assert_allclose(raw[:, 3:], states[start:start+64], atol=2e-6)
        np.testing.assert_allclose(raw[:-1, :3], actions[start:start+63], atol=2e-6)
        np.testing.assert_array_equal(sample.conditions[0], sample.trajectories[0, 3:])
        np.testing.assert_array_equal(sample.conditions[63], sample.trajectories[-1, 3:])
        assert not sample.loss_mask[-1, :3].any()
        assert not sample.loss_mask[[0, -1], 3:].any()
        reports[split] = dict(episodes=list(dataset.episodes), windows=len(dataset), batch_shape=list(batch.shape))
        if split == 'train':
            dataset.save_normalizer(Path(args.directory)/'normalizer.json')
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
