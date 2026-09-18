"""Linear semi-gradient Q-learning in NumPy, with explicit action features.

This is function approximation, NOT a tabular method or a deep neural network.
Learned weights start at zero. There is no heuristic fallback at evaluation.
"""

import json
from pathlib import Path
import numpy as np
from .environment import TRAVEL

FEATURE_NAMES = [
    "bias",
    "wait",
    "return",
    "deliver",
    "travel",
    "deliver_count",
    "urgent_deliveries",
    "late_deliveries",
    "pickup_count",
    "depot_backlog",
    "onboard_count",
    "time_left",
    "wait_with_cargo",
    "wait_at_depot",
    "return_empty",
    "return_with_cargo",
    "delivery_slack",
    "other_urgent",
    "return_backlog",
    "deliver_time_left",
    "return_time_left",
    "wait_after_cutoff",
    "same_destination_batch",
]


def features(obs, action):
    orders = obs["orders"]
    cargo = [o for o in orders if o["status"]]
    waiting = len(orders) - len(cargo)
    target = action - 1 if action else obs["location"]
    distance = int(TRAVEL[obs["location"], target]) if action else 0
    matches = [o for o in cargo if o["dest"] == target] if action > 1 else []
    slack = [o["deadline"] - obs["time"] - distance for o in matches]
    urgent = sum(s <= 3 for s in slack)
    late = sum(s < 0 for s in slack)
    other_urgent = sum(
        o["deadline"] - obs["time"] <= distance + 3 for o in cargo if o not in matches
    )
    w, r, d = float(action == 0), float(action == 1), float(action > 1)
    left = obs["remaining_shift"] / 48.0
    # Normalised, bounded features keep a single small learning rate practical.
    x = [
        1,
        w,
        r,
        d,
        distance / 5,
        len(matches) / 3,
        urgent / 3,
        late / 3,
        r * min(waiting, 3 - len(cargo)) / 3,
        waiting / 5,
        len(cargo) / 3,
        left,
        w * len(cargo) / 3,
        w * (obs["location"] == 0),
        r * (not cargo),
        r * len(cargo) / 3,
        np.clip(min(slack, default=0) / 16, -1, 1),
        other_urgent / 3,
        r * waiting / 5,
        d * left,
        r * left,
        w * (obs["time"] >= 36),
        float(len(matches) > 1),
    ]
    return np.asarray(x, dtype=float)


class LinearQAgent:
    def __init__(self, seed=0, alpha=0.015, gamma=0.98):
        self.weights = np.zeros(len(FEATURE_NAMES))
        self.rng = np.random.default_rng(seed)
        self.alpha, self.gamma = alpha, gamma

    def values(self, obs, mask):
        return {
            int(a): float(self.weights @ features(obs, int(a)))
            for a in np.flatnonzero(mask)
        }

    def act(self, obs, mask, epsilon=0.0):
        valid = np.flatnonzero(mask)
        if len(valid) == 1:
            return int(valid[0])
        if self.rng.random() < epsilon:
            return int(self.rng.choice(valid))
        values = self.values(obs, mask)
        # Deterministic ties keep evaluation reproducible.
        return max(values, key=values.get)

    def update(self, obs, action, discounted_reward, duration, nxt, mask, done):
        x = features(obs, action)
        bootstrap = 0.0 if done else max(self.values(nxt, mask).values())
        target = discounted_reward + self.gamma**duration * bootstrap
        error = target - self.weights @ x
        self.weights += self.alpha * error * x / (1 + x @ x)
        if not np.all(np.isfinite(self.weights)):
            raise FloatingPointError("Non-finite learning weights")

    def save(self, path):
        Path(path).write_text(
            json.dumps(
                dict(
                    features=FEATURE_NAMES,
                    weights=self.weights.tolist(),
                    alpha=self.alpha,
                    gamma=self.gamma,
                ),
                indent=2,
            )
        )

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        if data["features"] != FEATURE_NAMES:
            raise ValueError("Model feature schema mismatch")
        agent = cls(alpha=data["alpha"], gamma=data["gamma"])
        agent.weights = np.asarray(data["weights"], dtype=float)
        return agent


def train_episode(env, agent, seed, epsilon):
    obs, info = env.reset(seed=seed)
    done = False
    while not done:
        action = agent.act(obs, info["action_mask"], epsilon)
        nxt, reward, done, _, next_info = env.step(action)
        total, duration = reward, 1
        while nxt["travel_remaining"] and not done:
            nxt, reward, done, _, next_info = env.step(0)
            total += agent.gamma**duration * reward
            duration += 1
        # Semi-Markov update discounts both transit rewards and future value.
        agent.update(obs, action, total, duration, nxt, next_info["action_mask"], done)
        obs, info = nxt, next_info
    return env.metrics["reward"]
