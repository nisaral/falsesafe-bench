from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class DynamicsPrediction:
    mean: np.ndarray
    epistemic_std: np.ndarray
    aleatoric_std: np.ndarray | None = None


def regression_arrays(features, targets) -> tuple[np.ndarray, np.ndarray]:
    features, targets = np.asarray(features, float), np.asarray(targets, float)
    if features.ndim != 2 or targets.ndim != 2 or not len(features) or len(features) != len(targets) or not features.shape[1] or not targets.shape[1]:
        raise ValueError('Nonempty matching feature/target matrices are required')
    if not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise ValueError('Regression data must be finite')
    return features, targets


class EnsembleMLP:
    def __init__(self, members: int = 5, hidden_sizes: tuple[int, ...] = (64, 64)):
        if members < 2 or not hidden_sizes or min(hidden_sizes) < 1:
            raise ValueError('Ensemble requires multiple members and positive layer widths')
        self.member_count = members
        self.hidden_sizes = hidden_sizes
        self.models = []
        self.fitted = False

    def fit(self, features, targets, *, seed: int, epochs: int = 200,
            learning_rate: float = 0.001, max_seconds: float = 600) -> EnsembleMLP:
        features, targets = regression_arrays(features, targets)
        if epochs < 1 or not 0 < max_seconds <= 600 or not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError('Invalid training parameters; default CPU budget cannot exceed ten minutes')
        self.fitted = False
        started = time.monotonic()
        mean, scale = features.mean(axis=0), features.std(axis=0)
        scale = np.where(scale > 0, scale, 1.0)
        inputs = torch.as_tensor((features - mean) / scale, dtype=torch.float32)
        observations = torch.as_tensor(targets, dtype=torch.float32)
        models = []
        for member_index in range(self.member_count):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + member_index)
                layers, previous_size = [], features.shape[1]
                for hidden_size in self.hidden_sizes:
                    layers.extend((nn.Linear(previous_size, hidden_size), nn.Tanh()))
                    previous_size = hidden_size
                layers.append(nn.Linear(previous_size, targets.shape[1]))
                model = nn.Sequential(*layers)
            generator = np.random.default_rng(np.random.SeedSequence([seed, member_index]))
            bootstrap = torch.as_tensor(generator.integers(0, len(features), len(features)), dtype=torch.long)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            for _epoch in range(epochs):
                if time.monotonic() - started > max_seconds:
                    raise TimeoutError('Ensemble training exceeded the declared budget; no partial fitted model returned')
                optimizer.zero_grad()
                loss = (model(inputs[bootstrap]) - observations[bootstrap]).square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite ensemble training loss')
                loss.backward()
                optimizer.step()
            model.eval()
            models.append(model)
        if time.monotonic() - started > max_seconds:
            raise TimeoutError('Ensemble training exceeded the declared budget')
        self.models, self.input_mean, self.input_scale = models, mean, scale
        self.fitted = True
        return self

    def predict(self, features) -> DynamicsPrediction:
        if not self.fitted:
            raise RuntimeError('Ensemble has not been trained; no synthetic predictions are supplied')
        features = np.asarray(features, float)
        if features.ndim != 2 or features.shape[1] != len(self.input_mean) or not np.isfinite(features).all():
            raise ValueError('Invalid prediction features')
        inputs = torch.as_tensor((features - self.input_mean) / self.input_scale, dtype=torch.float32)
        with torch.no_grad():
            predictions = torch.stack([model(inputs) for model in self.models])
        return DynamicsPrediction(predictions.mean(dim=0).numpy(), predictions.std(dim=0, correction=0).numpy())

    def save(self, path: str | Path) -> None:
        if not self.fitted:
            raise RuntimeError('Cannot save an invented trained checkpoint')
        torch.save({'model_type': 'ensemble_mlp', 'member_count': self.member_count,
                    'hidden_sizes': self.hidden_sizes, 'input_mean': torch.as_tensor(self.input_mean),
                    'input_scale': torch.as_tensor(self.input_scale),
                    'state_dicts': [model.state_dict() for model in self.models]}, Path(path))
