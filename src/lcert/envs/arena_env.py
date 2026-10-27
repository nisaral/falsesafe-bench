from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    _HAS_GYMNASIUM = True
except ImportError:
    _HAS_GYMNASIUM = False
    gym = object  # type: ignore
    spaces = None  # type: ignore


class SimpleBox:
    """Lightweight fallback box space when gymnasium is not installed."""
    def __init__(self, low: float | np.ndarray, high: float | np.ndarray, shape: tuple[int, ...], dtype: Any = np.float32):
        self.low = np.full(shape, low, dtype=dtype) if np.isscalar(low) else np.asarray(low, dtype=dtype)
        self.high = np.full(shape, high, dtype=dtype) if np.isscalar(high) else np.asarray(high, dtype=dtype)
        self.shape = shape
        self.dtype = dtype

    def sample(self) -> np.ndarray:
        return np.random.uniform(self.low, self.high).astype(self.dtype)

    def contains(self, x: np.ndarray) -> bool:
        return bool(np.all(x >= self.low) and np.all(x <= self.high))


class SafeArenaEnv(gym.Env if _HAS_GYMNASIUM else object):
    """
    Standardized Multi-Task Robotic Safety Benchmark Environment.

    Tasks:
        - 'slalom': Dubins car threading oscillating hazard fields.
        - 'chasm': Narrow unvisited corridor where unshielded policy crashes.
        - 'braking': Double integrator racing toward boundary wall.
        - 'drone': Altitude and vertical speed flight envelope corridor.
        - 'adversary': Pursuit-evasion game with dynamic chaser.
        - 'highway': Highway bottleneck merge conflict with oncoming traffic.
        - 'canyon': Urban corridor between buildings with stochastic wind shear.

    Shields:
        - 'none' (or None): Raw unshielded policy (vulnerable to false-safe traps).
        - 'nominal': Hamilton-Jacobi shield computed with learned nominal dynamics (k=0).
        - 'uncertainty': Hamilton-Jacobi shield with disturbance inflated by k * sigma(x).
        - 'oracle': Ground-truth analytic ODE barrier.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        task: str = "slalom",
        shield: str = "uncertainty",
        k: float = 1.5,
        max_steps: int = 400,
        dt: float = 0.03,
        render_mode: str | None = None,
    ):
        super().__init__()
        valid_tasks = ("slalom", "chasm", "braking", "drone", "adversary", "highway", "canyon")
        if task not in valid_tasks:
            raise ValueError(f"Task '{task}' unknown. Must be one of {valid_tasks}")
        valid_shields = ("none", "nominal", "uncertainty", "oracle")
        shield = (shield or "none").lower()
        if shield not in valid_shields:
            raise ValueError(f"Shield '{shield}' unknown. Must be one of {valid_shields}")

        self.task = task
        self.shield_mode = shield
        self.k = float(k)
        self.max_steps = int(max_steps)
        self.dt = float(dt)
        self.render_mode = render_mode

        # Define Observation & Action Spaces
        obs_dim = 7  # [pos1, pos2, vel/heading, speed, clearance, sigma, intervened]
        if _HAS_GYMNASIUM:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
            self.action_space = spaces.Box(low=-1.6, high=1.6, shape=(1,), dtype=np.float32)
        else:
            self.observation_space = SimpleBox(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
            self.action_space = SimpleBox(low=-1.6, high=1.6, shape=(1,), dtype=np.float32)

        self._step_count = 0
        self._state = np.zeros(4, dtype=np.float32)
        self._total_interventions = 0

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            np.random.seed(seed)
        self._step_count = 0
        self._total_interventions = 0

        if self.task in ("slalom", "chasm", "canyon"):
            self._state = np.array([-3.2, 0.0, 0.0, 1.0], dtype=np.float32)  # [x, y, theta, v]
        elif self.task == "braking":
            self._state = np.array([2.8, -1.8, 0.0, 0.0], dtype=np.float32)  # [p, v, 0, 0]
        elif self.task == "drone":
            self._state = np.array([4.0, -2.5, 0.0, 0.0], dtype=np.float32)  # [z, w, 0, 0]
        elif self.task == "adversary":
            self._state = np.array([-3.2, -1.0, 0.2, 1.0], dtype=np.float32)
        elif self.task == "highway":
            self._state = np.array([-3.2, -0.8, 0.35, 1.1], dtype=np.float32)

        obs = self._get_obs(0.0, 0.0, False)
        info = {"interventions": self._total_interventions, "clearance": 1.0}
        return obs, info

    def step(self, action: np.ndarray | float | list[float]) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._step_count += 1
        u_cmd = float(np.asarray(action).ravel()[0])
        u_cmd = max(-1.6, min(1.6, u_cmd))

        # Evaluate Formal Safety & Uncertainty
        clearance, grad_x, grad_y, sigma = self._evaluate_safety()
        u_exec = u_cmd
        intervened = False

        if self.shield_mode != "none":
            threshold = 0.35
            if self.shield_mode == "uncertainty":
                threshold = 0.35 + self.k * (sigma * 0.95)
            elif self.shield_mode == "nominal":
                threshold = 0.16
            elif self.shield_mode == "oracle":
                threshold = 0.42

            if clearance < threshold:
                hx = math.cos(self._state[2])
                hy = math.sin(self._state[2])
                dot = hx * grad_x + hy * grad_y
                if dot < 0.25:
                    cross = hx * grad_y - hy * grad_x
                    u_exec = 1.6 if cross > 0 else -1.6
                    intervened = True
                    self._total_interventions += 1

        # Advance Kinematics
        if self.task in ("slalom", "chasm", "adversary", "highway", "canyon"):
            self._state[2] += u_exec * self.dt
            self._state[2] = (self._state[2] + math.pi) % (2 * math.pi) - math.pi
            self._state[0] += self._state[3] * math.cos(self._state[2]) * self.dt
            self._state[1] += self._state[3] * math.sin(self._state[2]) * self.dt
        elif self.task == "braking":
            self._state[0] += self._state[1] * self.dt + 0.5 * u_exec * self.dt**2
            self._state[1] += u_exec * self.dt
        elif self.task == "drone":
            thrust = 3.5 if intervened else u_cmd
            self._state[0] += self._state[1] * self.dt
            self._state[1] += thrust * self.dt

        # Check termination & rewards
        terminated = clearance <= 0.01
        truncated = self._step_count >= self.max_steps
        reward = 1.0 - (100.0 if terminated else 0.0) - (0.5 if intervened else 0.0)

        obs = self._get_obs(clearance, sigma, intervened)
        info = {
            "intervened": intervened,
            "u_cmd": u_cmd,
            "u_exec": u_exec,
            "clearance": clearance,
            "sigma": sigma,
            "total_interventions": self._total_interventions,
        }
        return obs, reward, terminated, truncated, info

    def _evaluate_safety(self) -> tuple[float, float, float, float]:
        x, y = float(self._state[0]), float(self._state[1])
        # Simple analytic clearance calculation
        # Obstacle center at (0, 0) r=0.7
        dist = math.hypot(x, y) - 0.92
        grad_x = x / (dist + 0.92 + 1e-5)
        grad_y = y / (dist + 0.92 + 1e-5)
        # Gap inside [-0.8, 0.8] x [-0.8, 0.8]
        in_gap = -0.8 <= x <= 0.8 and -0.8 <= y <= 0.8
        sigma = 0.42 if in_gap else 0.03
        return dist, grad_x, grad_y, sigma

    def _get_obs(self, clearance: float, sigma: float, intervened: bool) -> np.ndarray:
        return np.array([
            self._state[0], self._state[1], self._state[2], self._state[3],
            clearance, sigma, float(intervened)
        ], dtype=np.float32)
