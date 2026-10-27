from __future__ import annotations

import pytest
import numpy as np

from lcert.envs import SafeArenaEnv


def test_safe_arena_env_initialization_and_reset():
    env = SafeArenaEnv(task="slalom", shield="uncertainty", k=1.5)
    obs, info = env.reset(seed=42)

    assert obs.shape == (7,)
    assert isinstance(info, dict)
    assert info["interventions"] == 0


def test_safe_arena_env_step_unshielded_vs_shielded():
    env_unshielded = SafeArenaEnv(task="slalom", shield="none")
    env_unshielded.reset(seed=0)
    # Step toward obstacle center
    obs_u, reward_u, term_u, trunc_u, info_u = env_unshielded.step([0.0])
    assert info_u["intervened"] is False
    assert info_u["u_exec"] == 0.0

    env_shielded = SafeArenaEnv(task="slalom", shield="uncertainty", k=2.0)
    env_shielded.reset(seed=0)
    # Manually place robot right on the edge of the obstacle
    env_shielded._state = np.array([-0.8, 0.0, 0.0, 1.0], dtype=np.float32)
    obs_s, reward_s, term_s, trunc_s, info_s = env_shielded.step([0.0])

    # The shield should intervene to steer away from the obstacle
    assert info_s["intervened"] is True
    assert abs(info_s["u_exec"]) > 0.0


def test_safe_arena_env_multi_tasks():
    for task in ("slalom", "chasm", "braking", "drone", "adversary", "highway", "canyon"):
        env = SafeArenaEnv(task=task, shield="nominal")
        obs, info = env.reset()
        assert obs.shape == (7,)
        obs, reward, term, trunc, info = env.step(env.action_space.sample())
        assert obs.shape == (7,)
        assert isinstance(reward, float)
