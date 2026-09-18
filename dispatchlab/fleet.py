"""DispatchLab Fleet: a two-vehicle, energy-constrained delivery environment.

Joint commands are selected before either vehicle moves. One step always equals
one clock tick. Exogenous arrivals and route delays are independent of actions.
"""

from dataclasses import dataclass
from copy import deepcopy
import numpy as np

NODES = ("Hub", "Market", "North", "Station", "Campus", "South", "Riverside")
XY = np.array([[3, 3], [0, 0], [3, 0], [6, 0], [0, 6], [3, 6], [6, 6]])
DIST = np.maximum(1, np.ceil(np.abs(XY[:, None] - XY[None, :]).sum(-1) / 2)).astype(int)
np.fill_diagonal(DIST, 0)
COMMANDS = ("Hold / charge", "Return to hub") + tuple("Serve " + n for n in NODES[1:])


@dataclass(frozen=True)
class FleetConfig:
    horizon: int = 60
    cutoff: int = 44
    demand_rate: float = 0.62
    battery: int = 18
    charge_rate: int = 3
    capacity: int = 3
    max_orders: int = 10
    deadline_low: int = 10
    deadline_high: int = 18
    traffic_probability: float = 0.18
    traffic_extra: int = 2

    def __post_init__(self):
        if not 0 < self.cutoff <= self.horizon:
            raise ValueError("cutoff must be within horizon")
        if self.demand_rate < 0 or not 0 <= self.traffic_probability <= 1:
            raise ValueError("Invalid demand/traffic")
        if self.battery < 2 * DIST[0].max() or self.charge_rate < 1:
            raise ValueError("Battery must support a round trip to every node")
        if (
            not 1 <= self.capacity <= self.max_orders
            or self.deadline_low < 1
            or self.deadline_high < self.deadline_low
        ):
            raise ValueError("Invalid capacity/deadlines")


class FleetEnv:
    def __init__(self, config=None):
        self.config = config or FleetConfig()
        self.done = True

    def reset(self, seed=0):
        c = self.config
        # Separate deterministic streams: demand is unchanged if traffic changes.
        a, b = np.random.SeedSequence(seed).spawn(2)
        rng, trng = np.random.default_rng(a), np.random.default_rng(b)
        self._arrivals = {}
        for t in range(c.cutoff):
            n = int(rng.poisson(c.demand_rate)) + (2 if t == 0 else 0)
            probs = (
                [0.27, 0.23, 0.20, 0.10, 0.10, 0.10]
                if t < c.cutoff / 2
                else [0.10, 0.10, 0.10, 0.20, 0.23, 0.27]
            )
            self._arrivals[t] = [
                dict(
                    dest=int(rng.choice(range(1, 7), p=probs)),
                    deadline=t + int(rng.integers(c.deadline_low, c.deadline_high + 1)),
                    created=t,
                    priority=2 if rng.random() < 0.25 else 1,
                )
                for _ in range(n)
            ]
        self._delays = np.zeros((c.horizon, 7, 7), dtype=int)
        for t in range(c.horizon):
            prob = min(0.95, c.traffic_probability + (0.25 if 18 <= t < 36 else 0))
            self._delays[t] = (trng.random((7, 7)) < prob) * c.traffic_extra
        self.t = self.next_id = 0
        self.done = False
        self.orders = []
        self.vehicles = [
            dict(
                id=i,
                location=0,
                target=0,
                remaining=0,
                duration=0,
                battery=c.battery,
                cooldown=0,
            )
            for i in range(2)
        ]
        self.metrics = dict(
            requests=0,
            delivered=0,
            on_time=0,
            rejected=0,
            priority_requests=0,
            priority_on_time=0,
            unfinished=0,
            travel_ticks=0,
            distance=0,
            energy_used=0,
            energy_charged=0,
            delay=0,
            reward=0.0,
        )
        self.events = []
        # Initial overflow costs are charged on the first transition, not lost.
        self._pending_cost = self._reveal()
        self._load()
        return self.observation(), self.mask()

    def _reveal(self):
        penalty = 0.0
        for entry in self._arrivals.get(self.t, []):
            self.metrics["requests"] += 1
            self.metrics["priority_requests"] += int(entry["priority"] == 2)
            oid = self.next_id
            self.next_id += 1
            if len(self.orders) >= self.config.max_orders:
                self.metrics["rejected"] += 1
                penalty -= 12 * entry["priority"]
                self.events.append(f"Order {oid + 1} rejected: queue full")
            else:
                self.orders.append(dict(**entry, id=oid, vehicle=-1))
                self.events.append(
                    f"Order {oid + 1} received for {NODES[entry['dest']]}"
                )
        return penalty

    def _load(self):
        # Alternate first access to the depot queue; prevents permanent V1 bias.
        for i in (self.t % 2, 1 - self.t % 2):
            v = self.vehicles[i]
            if v["location"] != 0 or v["remaining"] or v["cooldown"]:
                continue
            free = self.config.capacity - sum(o["vehicle"] == i for o in self.orders)
            for o in sorted(
                self.orders, key=lambda o: (-o["priority"], o["deadline"], o["id"])
            ):
                if free and o["vehicle"] == -1:
                    o["vehicle"] = i
                    free -= 1

    def observation(self):
        c = self.config
        return dict(
            time=self.t,
            horizon=c.horizon,
            cutoff=c.cutoff,
            rush=int(18 <= self.t < 36),
            battery_capacity=c.battery,
            charge_rate=c.charge_rate,
            capacity=c.capacity,
            traffic_probability=c.traffic_probability,
            traffic_extra=c.traffic_extra,
            vehicles=deepcopy(self.vehicles),
            orders=deepcopy(self.orders),
        )

    def mask(self):
        masks = []
        for i, v in enumerate(self.vehicles):
            m = np.zeros(8, dtype=bool)
            m[0] = True
            if not self.done and not v["remaining"] and not v["cooldown"]:
                if v["location"] != 0 and v["battery"] >= DIST[v["location"], 0]:
                    m[1] = True
                for o in self.orders:
                    d = o["dest"]
                    if (
                        o["vehicle"] == i
                        and d != v["location"]
                        and v["battery"] >= DIST[v["location"], d] + DIST[d, 0]
                    ):
                        m[d + 1] = True
            masks.append(m)
        return (masks[0][:, None] & masks[1][None, :]).reshape(-1)

    def step(self, action):
        if self.done:
            raise RuntimeError("Reset after terminal")
        if (
            not isinstance(action, (int, np.integer))
            or not 0 <= action < 64
            or not self.mask()[action]
        ):
            raise ValueError(f"Infeasible joint command {action}")
        c = self.config
        self.events = []
        reward = self._pending_cost
        self._pending_cost = 0
        commands = divmod(int(action), 8)
        for i, (v, command) in enumerate(zip(self.vehicles, commands)):
            if command:
                dest = command - 1
                distance = int(DIST[v["location"], dest])
                v["target"] = dest
                v["duration"] = distance + int(
                    self._delays[self.t, v["location"], dest]
                )
                v["remaining"] = v["duration"]
                v["battery"] -= distance
                self.metrics["distance"] += distance
                self.metrics["energy_used"] += distance
                self.events.append(f"V{i + 1} dispatched to {NODES[dest]}")
            if v["remaining"]:
                v["remaining"] -= 1
                self.metrics["travel_ticks"] += 1
                reward -= 0.3
                if v["remaining"] == 0:
                    v["location"] = v["target"]
                    v["cooldown"] = 1
                    for o in list(self.orders):
                        if o["vehicle"] == i and o["dest"] == v["location"]:
                            delay = max(0, self.t + 1 - o["deadline"])
                            self.metrics["delivered"] += 1
                            self.metrics["on_time"] += int(delay == 0)
                            self.metrics["priority_on_time"] += int(
                                delay == 0 and o["priority"] == 2
                            )
                            self.metrics["delay"] += delay
                            reward += 10 * o["priority"] - 0.35 * o["priority"] * int(
                                delay > 0
                            )
                            self.orders.remove(o)
                            self.events.append(
                                f"V{i + 1} delivered order {o['id'] + 1}"
                                + (f" ({delay} late)" if delay else " on time")
                            )
            else:
                if v["cooldown"]:
                    v["cooldown"] -= 1
                if v["location"] == 0:
                    added = min(c.charge_rate, c.battery - v["battery"])
                    v["battery"] += added
                    self.metrics["energy_charged"] += added
                    reward -= 0.04 * added
        self.t += 1
        reward -= 0.35 * sum(
            o["priority"] for o in self.orders if self.t > o["deadline"]
        )
        if self.t < c.horizon:
            reward += self._reveal()
            self._load()
        self.done = self.t >= c.horizon
        if self.done:
            self.metrics["unfinished"] = len(self.orders)
            reward -= 12 * sum(o["priority"] for o in self.orders)
        self.metrics["reward"] += reward
        return self.observation(), float(reward), self.done, self.mask()

    def snapshot(self):
        return dict(
            **self.observation(), metrics=self.metrics.copy(), events=self.events.copy()
        )


def local_masks(mask):
    grid = np.asarray(mask).reshape(8, 8)
    return grid.any(axis=1), grid.any(axis=0)


def encode(obs):
    """Fixed 164-dimensional observation for default 10-order configuration.

    Includes a full padded order list (not future demand), plus useful aggregates.
    The feature schema deliberately fixes two vehicles and ten active orders.
    """
    if len(obs["orders"]) > 10:
        raise ValueError("Neural encoder supports at most ten orders")
    x = [
        obs["time"] / obs["horizon"],
        (obs["horizon"] - obs["time"]) / obs["horizon"],
        obs["rush"],
        float(obs["time"] >= obs["cutoff"]),
        obs["traffic_probability"],
        obs["traffic_extra"] / 4,
        obs["battery_capacity"] / 18,
        obs["charge_rate"] / 3,
    ]
    for i, v in enumerate(obs["vehicles"]):
        x.extend([float(v["location"] == n) for n in range(7)])
        x.extend([float(v["target"] == n) for n in range(7)])
        x.extend(
            [
                v["remaining"] / 10,
                v["battery"] / obs["battery_capacity"],
                v["cooldown"],
                sum(o["vehicle"] == i for o in obs["orders"]) / 3,
            ]
        )
    orders = sorted(obs["orders"], key=lambda o: (o["deadline"], o["id"]))
    for j in range(10):
        if j >= len(orders):
            x.extend([0.0] * 12)
            continue
        o = orders[j]
        x.extend([float(o["dest"] == n) for n in range(1, 7)])
        x.extend([float(o["vehicle"] == v) for v in (-1, 0, 1)])
        x.extend(
            [
                np.clip((o["deadline"] - obs["time"]) / 20, -3, 1),
                o["priority"] / 2,
                (obs["time"] - o["created"]) / 60,
            ]
        )
    return np.asarray(x, dtype=np.float32)


OBS_SIZE = 164
