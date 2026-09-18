"""Advanced experiment: separate train/validation/test seeds; no oracle baselines."""

import argparse, json, time, hashlib
from pathlib import Path
from dataclasses import asdict, replace
import numpy as np
import torch
from .fleet import FleetEnv, FleetConfig, encode, COMMANDS, XY, NODES
from .fleet_policies import fleet_heuristic
from .deep_agent import DoubleDQN
from .experiment import write_csv, paired_ci


def rollout(config, seed, policy, record=False, agent=None):
    env = FleetEnv(config)
    obs, mask = env.reset(seed)
    frames = (
        [dict(**env.snapshot(), commands=["Shift begins"] * 2, alternatives=[])]
        if record
        else []
    )
    latencies = []
    while not env.done:
        start = time.perf_counter()
        action = int(policy(obs, mask))
        latencies.append((time.perf_counter() - start) * 1000)
        alternatives = []
        if record and agent is not None and mask.sum() > 1:
            q = agent.q_values(obs)
            best = sorted(np.flatnonzero(mask), key=lambda a: -q[a])[:3]
            alternatives = [
                dict(
                    action=int(a),
                    commands=[COMMANDS[a // 8], COMMANDS[a % 8]],
                    value=float(q[a]),
                )
                for a in best
            ]
        obs, reward, done, mask = env.step(action)
        if record:
            frames.append(
                dict(
                    **env.snapshot(),
                    commands=[COMMANDS[action // 8], COMMANDS[action % 8]],
                    action=action,
                    step_reward=reward,
                    alternatives=alternatives,
                )
            )
    m = env.metrics.copy()
    m.update(
        on_time_rate=m["on_time"] / max(1, m["requests"]),
        completion_rate=m["delivered"] / max(1, m["requests"]),
        priority_service=m["priority_on_time"] / max(1, m["priority_requests"]),
        distance_per_delivery=m["distance"] / max(1, m["delivered"]),
        mean_delay=m["delay"] / max(1, m["delivered"]),
        latency_median_ms=float(np.median(latencies)),
    )
    return m, frames


def score(config, seeds, policy):
    return float(np.mean([rollout(config, s, policy)[0]["reward"] for s in seeds]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=1500)
    p.add_argument("--test-days", type=int, default=150)
    p.add_argument("--validation-days", type=int, default=40)
    p.add_argument("--output", default="advanced")
    p.add_argument(
        "--train-only",
        action="store_true",
        help="Freeze models without opening the test set",
    )
    p.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Evaluate models already selected on validation",
    )
    args = p.parse_args()
    if (
        not 1 <= args.episodes < 50000
        or not 1 <= args.test_days <= 1000
        or not 1 <= args.validation_days <= 1000
    ):
        p.error("Seed range bounds: episodes <50000; evaluation days <=1000")
    root = Path(args.output)
    for d in ["models", "results", "demo"]:
        (root / d).mkdir(parents=True, exist_ok=True)
    cfg = FleetConfig()
    val = range(900000, 900000 + args.validation_days)
    replicas = [101, 202, 303]
    history = []
    selected = []
    agents = []
    if not args.evaluate_only:
        # Baseline selection uses the same validation split as checkpoint choice.
        batch_scores = {
            k: score(cfg, val, lambda o, m, k=k: fleet_heuristic(o, m, "planner", k))
            for k in [1, 2, 3]
        }
        best_batch = max(batch_scores, key=batch_scores.get)
        print("Planner validation:", batch_scores, flush=True)
        for index, seed in enumerate(replicas):
            agent = DoubleDQN(seed=seed)
            rng = np.random.default_rng(seed + 10)
            env = FleetEnv(cfg)
            step_count = 0
            best = -np.inf
            best_episode = 0
            rewards = []
            losses = []
            for episode in range(args.episodes):
                obs, mask = env.reset(300000 + index * 50000 + episode)
                epsilon = max(0.08, 1 - 0.92 * episode / (args.episodes * 0.8))
                while not env.done:
                    if rng.random() < epsilon:
                        # Guided exploration is explicit: 70% reactive rule, 30%
                        # uniform feasible action. Evaluation has no rule fallback.
                        action = (
                            fleet_heuristic(obs, mask, "deadline")
                            if rng.random() < 0.7
                            else int(rng.choice(np.flatnonzero(mask)))
                        )
                    else:
                        action = agent.act(obs, mask)
                    nxt, reward, done, nmask = env.step(action)
                    agent.buffer.add(
                        encode(obs), action, reward / 20.0, encode(nxt), done, nmask
                    )
                    step_count += 1
                    if step_count % 4 == 0:
                        loss = agent.optimise()
                        if loss is not None:
                            losses.append(loss)
                    obs, mask = nxt, nmask
                rewards.append(env.metrics["reward"])
                if (episode + 1) % 150 == 0 or episode + 1 == args.episodes:
                    v = score(cfg, val, agent.act)
                    history.append(
                        dict(
                            seed=seed,
                            episode=episode + 1,
                            validation_reward=v,
                            training_reward=float(np.mean(rewards[-150:])),
                            loss=float(np.mean(losses[-1000:])) if losses else 0,
                        )
                    )
                    if v > best:
                        best = v
                        best_episode = episode + 1
                        agent.save(root / "models" / f"ddqn_{seed}.pt")
                    print(
                        f"replica={seed} episode={episode + 1} validation={v:.1f} best={best:.1f}",
                        flush=True,
                    )
            selected.append(
                dict(
                    seed=seed,
                    episode=best_episode,
                    validation_reward=best,
                    gradient_updates=agent.updates,
                )
            )
            agents.append(DoubleDQN.load(root / "models" / f"ddqn_{seed}.pt"))
        winner = int(np.argmax([x["validation_reward"] for x in selected]))
        agents[winner].save(root / "models" / "ddqn.pt")
        manifest = dict(
            config=asdict(cfg),
            replicas=replicas,
            selected=selected,
            demo_seed=replicas[winner],
            planner_batch=best_batch,
            planner_validation=batch_scores,
            train_episodes=args.episodes,
            validation_days=args.validation_days,
            torch_version=torch.__version__,
            numpy_version=np.__version__,
            algorithm="Dueling Double DQN, masked targets, replay, Polyak target, guided exploration",
            reward_scale=20,
            gamma=0.97,
            learning_rate=0.0003,
            replay_capacity=30000,
            batch_size=64,
            update_every_ticks=4,
            warmup_transitions=1024,
            target_tau=0.01,
        )
        (root / "results" / "training.json").write_text(json.dumps(manifest, indent=2))
        write_csv(root / "results" / "learning.csv", history)
    else:
        manifest = json.loads((root / "results" / "training.json").read_text())
        agents = [
            DoubleDQN.load(root / "models" / f"ddqn_{seed}.pt") for seed in replicas
        ]
        best_batch = manifest["planner_batch"]
    if args.train_only:
        return
    scenarios = {
        "standard": cfg,
        "demand_surge": replace(cfg, demand_rate=0.95),
        "traffic_disruption": replace(cfg, traffic_probability=0.65, traffic_extra=3),
        "energy_constraint": replace(cfg, battery=12, charge_rate=2),
    }
    rows = []
    summary = {}
    comparisons = {}
    for si, (name, config) in enumerate(scenarios.items()):
        seeds = list(range(1000000 + si * 1000, 1000000 + si * 1000 + args.test_days))
        policies = {
            "nearest": lambda o, m: fleet_heuristic(o, m, "nearest"),
            "deadline": lambda o, m: fleet_heuristic(o, m, "deadline"),
            "planner": lambda o, m: fleet_heuristic(o, m, "planner", best_batch),
        }
        policies.update({f"ddqn_{seed}": a.act for seed, a in zip(replicas, agents)})
        summary[name] = {}
        per_policy = {}
        for label, policy in policies.items():
            values = []
            for seed in seeds:
                m, _ = rollout(config, seed, policy)
                values.append(m)
                rows.append(dict(scenario=name, policy=label, scenario_seed=seed, **m))
            per_policy[label] = values
            summary[name][label] = {
                k: float(np.mean([m[k] for m in values])) for k in values[0]
            }
        summary[name]["ddqn_mean"] = {
            k: float(np.mean([summary[name][f"ddqn_{s}"][k] for s in replicas]))
            for k in per_policy["nearest"][0]
        }
        comparisons[name] = {}
        rl = np.array(
            [[m["reward"] for m in per_policy[f"ddqn_{s}"]] for s in replicas]
        ).mean(0)
        for baseline in ["nearest", "deadline", "planner"]:
            diffs = rl - np.array([m["reward"] for m in per_policy[baseline]])
            comparisons[name][baseline] = dict(
                difference=float(diffs.mean()), paired_scenario_ci=paired_ci(diffs)
            )
        print(
            name,
            {k: round(v["reward"], 1) for k, v in summary[name].items()},
            flush=True,
        )
    report = dict(
        **manifest,
        test_days=args.test_days,
        summary=summary,
        comparisons=comparisons,
        scenario_configs={n: asdict(c) for n, c in scenarios.items()},
    )
    (root / "results" / "summary.json").write_text(json.dumps(report, indent=2))
    write_csv(root / "results" / "episodes.csv", rows)
    winner = DoubleDQN.load(root / "models" / "ddqn.pt")
    replays = {}
    for name, config in scenarios.items():
        for seed in [2000001, 2000002, 2000003]:
            for label, policy in {
                "Double DQN": winner.act,
                "Route planner": lambda o, m: fleet_heuristic(
                    o, m, "planner", best_batch
                ),
                "Earliest deadline": lambda o, m: fleet_heuristic(o, m, "deadline"),
            }.items():
                metrics, frames = rollout(
                    config,
                    seed,
                    policy,
                    True,
                    winner if label == "Double DQN" else None,
                )
                replays[f"{name}|{seed}|{label}"] = dict(metrics=metrics, frames=frames)
    data = dict(report=report, replays=replays, nodes=NODES, positions=XY.tolist())
    (root / "demo" / "replays.json").write_text(json.dumps(data, separators=(",", ":")))
    # Checksums make model and result provenance inspectable.
    paths = list((root / "models").glob("*.pt")) + [
        root / "results" / "episodes.csv",
        root / "results" / "summary.json",
    ]
    checksums = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
    }
    (root / "results" / "checksums.json").write_text(json.dumps(checksums, indent=2))


if __name__ == "__main__":
    main()
