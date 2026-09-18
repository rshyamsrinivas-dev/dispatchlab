"""Fair, non-oracle baselines. Policies receive observations, never the environment."""

import numpy as np
from .environment import TRAVEL


def heuristic(obs, mask, mode="deadline", batch_threshold=1):
    if obs["travel_remaining"]:
        return 0
    onboard = [o for o in obs["orders"] if o["status"]]
    waiting = [o for o in obs["orders"] if not o["status"]]
    if onboard:
        if obs["location"] == 0 and len(onboard) < batch_threshold:
            slack = min(
                o["deadline"] - obs["time"] - TRAVEL[0, o["dest"]] for o in onboard
            )
            if slack > 3 and obs["remaining_shift"] > 10:
                return 0
        if mode == "nearest":
            order = min(
                onboard,
                key=lambda o: (TRAVEL[obs["location"], o["dest"]], o["deadline"]),
            )
        else:
            order = min(
                onboard,
                key=lambda o: (o["deadline"], TRAVEL[obs["location"], o["dest"]]),
            )
        return order["dest"] + 1
    if mask[1]:
        return 1  # Return proactively, including when the depot queue is empty.
    return 0


def random_action(obs, mask, rng):
    return int(rng.choice(np.flatnonzero(mask)))
