from __future__ import annotations

import time

import gpytorch
import numpy as np
import torch

from .ensemble import DynamicsPrediction, regression_arrays


class _ExactOutputGP(gpytorch.models.ExactGP):
    def __init__(self, inputs, targets, likelihood):
        super().__init__(inputs, targets, likelihood)
        self.mean = gpytorch.means.ConstantMean()
        self.covariance = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=inputs.shape[1]))

    def forward(self, inputs):
        return gpytorch.distributions.MultivariateNormal(self.mean(inputs), self.covariance(inputs))


class IndependentExactGP:
    def __init__(self, max_exact_samples: int = 2000):
        if max_exact_samples < 1:
            raise ValueError('Exact-GP sample budget must be positive')
        self.max_exact_samples = max_exact_samples
        self.fitted = False
        self.models = []
        self.likelihoods = []

    def fit(self, features, targets, *, seed: int, epochs: int = 100,
            learning_rate: float = 0.1, max_seconds: float = 600) -> IndependentExactGP:
        features, targets = regression_arrays(features, targets)
        if len(features) > self.max_exact_samples:
            raise ValueError('Exact-GP input exceeds its declared budget; approve a larger budget or a separate sparse-GP experiment, never subsample silently')
        if epochs < 1 or not 0 < max_seconds <= 600 or not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError('Invalid GP training parameters or CPU budget')
        self.fitted = False
        started = time.monotonic()
        mean, scale = features.mean(axis=0), features.std(axis=0)
        scale = np.where(scale > 0, scale, 1.0)
        inputs = torch.as_tensor((features - mean) / scale, dtype=torch.float64)
        observations = torch.as_tensor(targets, dtype=torch.float64)
        models, likelihoods = [], []
        for output_index in range(targets.shape[1]):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + output_index)
                likelihood = gpytorch.likelihoods.GaussianLikelihood().double()
                model = _ExactOutputGP(inputs, observations[:, output_index], likelihood).double()
            model.train()
            likelihood.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            objective = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
            for _epoch in range(epochs):
                if time.monotonic() - started > max_seconds:
                    raise TimeoutError('GP training exceeded the declared budget')
                optimizer.zero_grad()
                loss = -objective(model(inputs), observations[:, output_index])
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite GP training objective')
                loss.backward()
                optimizer.step()
            model.eval()
            likelihood.eval()
            models.append(model)
            likelihoods.append(likelihood)
        if time.monotonic() - started > max_seconds:
            raise TimeoutError('GP training exceeded the declared budget')
        self.models, self.likelihoods = models, likelihoods
        self.input_mean, self.input_scale = mean, scale
        self.fitted = True
        return self

    def predict(self, features) -> DynamicsPrediction:
        if not self.fitted:
            raise RuntimeError('GP has not been trained; no synthetic posterior is supplied')
        features = np.asarray(features, float)
        if features.ndim != 2 or features.shape[1] != len(self.input_mean) or not np.isfinite(features).all():
            raise ValueError('Invalid GP prediction features')
        inputs = torch.as_tensor((features - self.input_mean) / self.input_scale, dtype=torch.float64)
        means, epistemic, aleatoric = [], [], []
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for model, likelihood in zip(self.models, self.likelihoods, strict=True):
                latent = model(inputs)
                means.append(latent.mean.numpy())
                epistemic.append(latent.stddev.numpy())
                aleatoric.append(np.full(len(inputs), float(likelihood.noise.sqrt())))
        return DynamicsPrediction(np.column_stack(means), np.column_stack(epistemic), np.column_stack(aleatoric))
