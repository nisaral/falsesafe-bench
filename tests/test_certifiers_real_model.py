"""End-to-end certifier tests against a real, tiny EnsembleMLP (no stubs).

Guards the silent-fallback bug: the certifiers used to substitute a constant
sigma_ep=0.1 or the nominal residual when the model lacked predict_mean_std,
which hid interface mismatches. These tests fit a genuinely trained ensemble
and assert the real epistemic spread is used.
"""
from __future__ import annotations

import numpy as np
import pytest

from lcert.certifiers import ConformalCertifier, DistanceAwareCertifier
from lcert.models.ensemble import EnsembleMLP


@pytest.fixture(scope="module")
def tiny_model():
    rng = np.random.default_rng(0)
    n = 120
    features = np.column_stack([
        rng.uniform(-1.0, 3.0, size=n),
        rng.uniform(-2.0, 2.0, size=n),
        rng.uniform(-1.0, 1.0, size=n),
    ])
    targets = np.column_stack([features[:, 1], features[:, 2] + 0.1 * features[:, 0]])
    model = EnsembleMLP(members=2, hidden_sizes=(16,))
    model.fit(features, targets, seed=0, epochs=5, max_seconds=30.0)
    return model, features, targets


def test_ensemble_exposes_predict_mean_std(tiny_model):
    model, features, _ = tiny_model
    means, stds = model.predict_mean_std(features[:5])
    assert means.shape == (5, 2)
    assert stds.shape == (5, 2)
    assert np.all(stds >= 0.0)


def test_distance_aware_uses_real_model(tiny_model):
    model, features, targets = tiny_model
    cert = DistanceAwareCertifier()
    cert.fit(features, targets, model=model)
    expected = float(np.max(np.abs(targets[:, 1] - model.predict(features).mean[:, 1])))
    assert cert.epsilon_0 == pytest.approx(expected, rel=1e-9, abs=1e-9)


def test_distance_aware_rejects_model_without_interface(tiny_model):
    _, features, targets = tiny_model
    cert = DistanceAwareCertifier()

    class Duck:
        def predict(self, z):
            return z

    with pytest.raises(TypeError, match="predict_mean_std"):
        cert.fit(features, targets, model=Duck())


def test_conformal_c1a_uses_real_epistemic_std(tiny_model):
    model, features, targets = tiny_model
    cert = ConformalCertifier("C1a", alpha=0.1)
    cert.calibrate(features, targets, model=model)
    assert cert.is_calibrated
    # Constant fallback sigma_ep=0.1 would make q_hat * 0.1 flat everywhere.
    bound = cert.disturbance_bound(features[:8, :2], actions=0.0, model=model)
    _, stds = model.predict_mean_std(np.column_stack([features[:8, :2], np.zeros(8)]))
    assert np.allclose(bound, cert.q_hat * np.maximum(stds[:, 1], 1e-4))


def test_conformal_rejects_model_without_interface(tiny_model):
    _, features, targets = tiny_model

    class Duck:
        def predict(self, z):
            return z

    with pytest.raises(TypeError, match="predict_mean_std"):
        ConformalCertifier("C1a", alpha=0.1).calibrate(features, targets, model=Duck())
