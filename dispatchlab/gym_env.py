"""Optional Gymnasium adapter, with padded numeric observations and action masks."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .environment import DispatchEnv, NAMES


class GymDispatchEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, config=None, render_mode=None):
        super().__init__()
        self.core = DispatchEnv(config)
        c = self.core.config
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Dict(
            {
                "vehicle": spaces.MultiDiscrete(
                    [c.horizon + 1, c.horizon + 1, 4, 4, 6]
                ),
                "orders": spaces.Box(
                    0, c.horizon + c.deadline_max + 1, (c.max_orders, 5), dtype=np.int64
                ),
            }
        )

    def _encode(self, obs):
        rows = np.zeros((self.core.config.max_orders, 5), dtype=np.int64)
        for i, o in enumerate(obs["orders"]):
            rows[i] = [
                o["id"] + 1,
                o["dest"],
                o["deadline"],
                o["created"],
                o["status"] + 1,
            ]
        return {
            "vehicle": np.array(
                [
                    obs["time"],
                    obs["remaining_shift"],
                    obs["location"],
                    obs["target"],
                    obs["travel_remaining"],
                ],
                dtype=np.int64,
            ),
            "orders": rows,
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs, info = self.core.reset(seed=seed, options=options)
        info["invalid_action"] = False
        return self._encode(obs), info

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError("Action outside action space")
        invalid = not self.core.action_mask()[action]
        obs, reward, terminated, truncated, info = self.core.step(
            0 if invalid else int(action)
        )
        # Off-the-shelf agents may ignore masks. Make all Discrete(5) actions
        # well-defined while discouraging impossible moves; valid moves match core.
        if invalid:
            reward -= 1.0
            self.core.metrics["reward"] -= 1.0
            info["metrics"] = self.core.metrics.copy()
        info["invalid_action"] = bool(invalid)
        return self._encode(obs), reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "ansi":
            return (
                f"t={self.core.t} location={NAMES[self.core.location]} "
                f"in_transit={self.core.remaining} orders={len(self.core.orders)} "
                f"delivered={self.core.metrics['delivered']}"
            )
