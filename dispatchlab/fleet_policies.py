"""Reactive and short-horizon route planning baselines, using only observed work."""

from itertools import permutations
from .fleet import DIST, local_masks


def expected_trip(obs, origin, dest):
    p = min(0.95, obs["traffic_probability"] + (0.25 if obs["rush"] else 0))
    return float(DIST[origin, dest]) + p * obs["traffic_extra"]


def plan_route(obs, i, local_mask):
    """Exhaustively order up to three onboard destinations, plus return to hub.

    Route cost uses expected travel, service stops, lateness, and energy. Replan
    after each stop. Does not solve joint assignment or anticipate future orders.
    """
    v = obs["vehicles"][i]
    cargo = [o for o in obs["orders"] if o["vehicle"] == i]
    dests = sorted({o["dest"] for o in cargo})
    if not dests:
        return 1 if local_mask[1] else 0
    best = (float("inf"), 0)
    for route in permutations(dests):
        if not local_mask[route[0] + 1]:
            continue
        battery = v["battery"]
        pos = v["location"]
        clock = float(obs["time"])
        cost = 0.0
        feasible = True
        for dest in route:
            battery -= int(DIST[pos, dest])
            if battery < DIST[dest, 0]:
                feasible = False
                break
            duration = expected_trip(obs, pos, dest)
            clock += duration
            cost += 0.3 * duration
            for o in cargo:
                if o["dest"] == dest:
                    cost += 0.35 * o["priority"] * max(0, clock - o["deadline"])
                    cost += 12 * o["priority"] * int(clock > obs["horizon"])
            clock += 1
            pos = dest
        if feasible:
            cost += 0.3 * expected_trip(obs, pos, 0)
            if cost < best[0]:
                best = (cost, route[0] + 1)
    if best[0] < float("inf"):
        return best[1]
    return 1 if local_mask[1] else 0


def fleet_heuristic(obs, mask, mode="deadline", batch=1):
    commands = []
    for i, m in enumerate(local_masks(mask)):
        v = obs["vehicles"][i]
        cargo = [o for o in obs["orders"] if o["vehicle"] == i]
        valid = [o for o in cargo if m[o["dest"] + 1]]
        if m.sum() == 1:
            commands.append(0)
            continue
        if v["location"] == 0 and cargo:
            # All heuristics obey the same charge rule; safety is always masked.
            if v["battery"] < min(obs["battery_capacity"], 10):
                commands.append(0)
                continue
            slack = min(o["deadline"] - obs["time"] - DIST[0, o["dest"]] for o in cargo)
            if len(cargo) < batch and slack > 4 and obs["time"] < obs["cutoff"]:
                commands.append(0)
                continue
        if valid:
            if mode == "planner":
                command = plan_route(obs, i, m)
            elif mode == "nearest":
                command = (
                    min(
                        valid,
                        key=lambda o: (DIST[v["location"], o["dest"]], o["deadline"]),
                    )["dest"]
                    + 1
                )
            else:
                command = (
                    min(valid, key=lambda o: (o["deadline"], -o["priority"]))["dest"]
                    + 1
                )
        else:
            command = 1 if m[1] else 0
        commands.append(command)
    return commands[0] * 8 + commands[1]
