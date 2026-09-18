"""Fixed-tick dispatch simulation. Core API mirrors Gymnasium without requiring it.

Every action consumes ONE tick. Travel continues across subsequent ticks; only
WAIT is valid while in transit. This avoids variable-duration discount mistakes.
"""

from dataclasses import asdict, dataclass
import numpy as np

NAMES = ("Depot", "North", "East", "South")
POSITIONS = ((0, 0), (-2, 2), (2, 1), (0, -2))
# Symmetric integer travel times; distances also define travel cost.
TRAVEL = np.array([[0, 3, 2, 2], [3, 0, 4, 5], [2, 4, 0, 3], [2, 5, 3, 0]])
ACTION_NAMES = (
    "Wait",
    "Return to depot",
    "Deliver North",
    "Deliver East",
    "Deliver South",
)


@dataclass(frozen=True)
class Config:
    horizon: int = 48
    arrival_cutoff: int = 36
    arrival_probability: float = 0.38
    capacity: int = 3
    max_orders: int = 5
    deadline_min: int = 9
    deadline_max: int = 16
    delivery_reward: float = 10.0
    travel_cost: float = 0.35
    late_cost: float = 0.7
    unfinished_cost: float = 8.0

    def __post_init__(self):
        if not 0 <= self.arrival_probability <= 1:
            raise ValueError("arrival_probability must be in [0, 1]")
        if not 0 < self.arrival_cutoff <= self.horizon:
            raise ValueError("arrival_cutoff must lie within the shift")
        if not 1 <= self.capacity <= self.max_orders:
            raise ValueError("Require 1 <= capacity <= max_orders")
        if not 0 < self.deadline_min <= self.deadline_max:
            raise ValueError("Invalid deadline interval")


class DispatchEnv:
    """Observable orders only; future arrivals never appear in observations/info.

    Order status: 0 waiting at depot, 1 onboard. No expiration or cancellation.
    Overflow is rejected and penalised, preventing reward gains by queue blocking.
    Loading at depot is automatic, earliest deadline first, and costs no time.
    """

    def __init__(self, config=None):
        self.config = config or Config()
        self.rng = np.random.default_rng()
        self.done = True

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        c = self.config
        self.t = self.location = self.target = self.remaining = 0
        self.orders = []
        self.done = False
        self.metrics = dict(
            requests=0,
            delivered=0,
            on_time=0,
            rejected=0,
            travel_ticks=0,
            total_delay=0,
            reward=0.0,
            unfinished=0,
        )
        self.next_id = 0
        # Pre-generate exogenous demand independently of the policy's actions.
        self._schedule = {}
        for t in range(c.arrival_cutoff):
            if t == 0 or self.rng.random() < c.arrival_probability:
                probs = (
                    [0.6, 0.25, 0.15] if t < c.arrival_cutoff / 2 else [0.15, 0.25, 0.6]
                )
                dest = int(self.rng.choice([1, 2, 3], p=probs))
                deadline = t + int(
                    self.rng.integers(c.deadline_min, c.deadline_max + 1)
                )
                self._schedule[t] = (dest, deadline)
        self._arrivals()
        self._load()
        return self.observation(), self.info()

    def _arrivals(self):
        if self.t not in self._schedule:
            return 0.0
        dest, deadline = self._schedule[self.t]
        self.metrics["requests"] += 1
        if len(self.orders) == self.config.max_orders:
            self.metrics["rejected"] += 1
            return -self.config.unfinished_cost
        self.orders.append(
            dict(
                id=self.next_id, dest=dest, deadline=deadline, created=self.t, status=0
            )
        )
        self.next_id += 1
        return 0.0

    def _load(self):
        if self.location != 0 or self.remaining:
            return
        free = self.config.capacity - sum(o["status"] for o in self.orders)
        for o in sorted(self.orders, key=lambda o: (o["deadline"], o["id"])):
            if free and o["status"] == 0:
                o["status"] = 1
                free -= 1

    def action_mask(self):
        mask = np.zeros(5, dtype=bool)
        mask[0] = True
        if self.done or self.remaining:
            return mask
        mask[1] = self.location != 0
        for o in self.orders:
            if o["status"] == 1 and o["dest"] != self.location:
                mask[o["dest"] + 1] = True
        return mask

    def observation(self):
        # Canonical order makes behaviour independent of Python container identity.
        return dict(
            time=self.t,
            remaining_shift=self.config.horizon - self.t,
            location=self.location,
            target=self.target,
            travel_remaining=self.remaining,
            orders=[o.copy() for o in sorted(self.orders, key=lambda o: o["id"])],
        )

    def info(self):
        return dict(action_mask=self.action_mask(), metrics=self.metrics.copy())

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode finished; call reset()")
        if (
            not isinstance(action, (int, np.integer))
            or not 0 <= action < 5
            or not self.action_mask()[action]
        ):
            raise ValueError(f"Infeasible action: {action}")
        c = self.config
        reward = 0.0
        if action > 0:
            self.target = int(action - 1)
            self.remaining = int(TRAVEL[self.location, self.target])
        if self.remaining:
            self.remaining -= 1
            self.metrics["travel_ticks"] += 1
            reward -= c.travel_cost
            if self.remaining == 0:
                self.location = self.target
        self.t += 1
        # Deliver only on arrival, not at departure. All matching parcels unload.
        if self.remaining == 0:
            delivered = [
                o for o in self.orders if o["status"] and o["dest"] == self.location
            ]
            for o in delivered:
                delay = max(0, self.t - o["deadline"])
                self.metrics["delivered"] += 1
                self.metrics["on_time"] += int(delay == 0)
                self.metrics["total_delay"] += delay
                reward += c.delivery_reward
                self.orders.remove(o)
                # Other overdue ticks were already charged while this order waited.
                reward -= c.late_cost * int(delay > 0)
        reward -= c.late_cost * sum(self.t > o["deadline"] for o in self.orders)
        if self.t < c.horizon:
            reward += self._arrivals()
            self._load()
        self.done = self.t >= c.horizon
        if self.done:
            self.metrics["unfinished"] = len(self.orders)
            reward -= c.unfinished_cost * len(self.orders)
        self.metrics["reward"] += reward
        return self.observation(), float(reward), self.done, False, self.info()

    def snapshot(self):
        return {
            **self.observation(),
            "metrics": self.metrics.copy(),
            "config": asdict(self.config),
        }
