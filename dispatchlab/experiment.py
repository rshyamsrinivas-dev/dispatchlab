"""Reproduce training, validation selection, held-out evaluation, and replay data."""

import argparse
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np
from .agent import LinearQAgent, train_episode
from .environment import Config, DispatchEnv, ACTION_NAMES
from .policies import heuristic, random_action


def run_episode(config, seed, policy, record=False):
    env = DispatchEnv(config)
    obs, info = env.reset(seed=seed)
    frames = [dict(**env.snapshot(), action="Shift starts")]
    done = False
    while not done:
        action = int(policy(obs, info["action_mask"]))
        obs, reward, done, _, info = env.step(action)
        if record:
            frames.append(
                dict(**env.snapshot(), action=ACTION_NAMES[action], step_reward=reward)
            )
    m = env.metrics.copy()
    m["on_time_rate"] = m["on_time"] / max(1, m["requests"])
    m["completion_rate"] = m["delivered"] / max(1, m["requests"])
    m["mean_delay"] = m["total_delay"] / max(1, m["delivered"])
    m["travel_per_delivery"] = m["travel_ticks"] / max(1, m["delivered"])
    return m, frames


def evaluate(config, seeds, policy):
    return [run_episode(config, int(seed), policy)[0] for seed in seeds]


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_ci(differences, seed=73):
    rng = np.random.default_rng(seed)
    values = np.asarray(differences)
    means = values[rng.integers(len(values), size=(2000, len(values)))].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=3000)
    p.add_argument("--test-days", type=int, default=300)
    p.add_argument("--validation-days", type=int, default=80)
    p.add_argument("--output", default=".")
    args = p.parse_args()
    if (
        not 1 <= args.episodes <= 20000
        or not 1 <= args.test_days <= 10000
        or not 1 <= args.validation_days <= 10000
    ):
        p.error(
            "episodes: 1..20000; evaluation days: 1..10000 (preserves disjoint seed ranges)"
        )
    root = Path(args.output)
    for folder in ("results", "models", "demo"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    c = Config()
    val_seeds = range(80000, 80000 + args.validation_days)
    # Tune a batching heuristic on validation, not test. Nearest and EDF are
    # parameter-free; retain all three so a weak baseline cannot hide results.
    baseline_scores = {}
    for k in (1, 2, 3):
        policy = lambda o, m, k=k: heuristic(o, m, "deadline", k)
        baseline_scores[k] = float(
            np.mean([r["reward"] for r in evaluate(c, val_seeds, policy)])
        )
    best_k = max(baseline_scores, key=baseline_scores.get)
    history, validation, agents = [], [], []
    for train_seed in (11, 22, 33):
        agent = LinearQAgent(seed=train_seed)
        env = DispatchEnv(c)
        best_score, best_weights, best_episode = -np.inf, None, None
        recent = []
        for episode in range(args.episodes):
            epsilon = max(0.05, 0.9 * (1 - episode / (0.85 * args.episodes)))
            # Each replica has its own training demand; all ranges below validation.
            seed = (train_seed // 11 - 1) * 20000 + episode
            recent.append(train_episode(env, agent, seed, epsilon))
            if (episode + 1) % 250 == 0 or episode == args.episodes - 1:
                score = float(
                    np.mean([r["reward"] for r in evaluate(c, val_seeds, agent.act)])
                )
                history.append(
                    dict(
                        training_seed=train_seed,
                        episode=episode + 1,
                        training_reward=float(np.mean(recent[-250:])),
                        validation_reward=score,
                    )
                )
                if score > best_score:
                    best_score, best_weights, best_episode = (
                        score,
                        agent.weights.copy(),
                        episode + 1,
                    )
                print(
                    f"seed={train_seed} episode={episode + 1} validation={score:.2f}",
                    flush=True,
                )
        agent.weights = best_weights
        agent.save(root / "models" / f"linear_q_seed_{train_seed}.json")
        agents.append(agent)
        validation.append(
            dict(training_seed=train_seed, reward=best_score, checkpoint=best_episode)
        )
    # Demo model is chosen on validation before any test results are computed.
    best_index = int(np.argmax([v["reward"] for v in validation]))
    agents[best_index].save(root / "models" / "linear_q.json")
    scenarios = {
        "standard": c,
        "demand_surge": replace(c, arrival_probability=0.6),
        "tight_deadlines": replace(c, deadline_min=6, deadline_max=10),
    }
    all_rows, summary, comparisons = [], {}, {}
    for scenario_index, (name, config) in enumerate(scenarios.items()):
        seeds = range(
            100000 + scenario_index * 10000,
            100000 + scenario_index * 10000 + args.test_days,
        )
        policies = {
            "nearest": lambda o, m: heuristic(o, m, "nearest"),
            "deadline": lambda o, m: heuristic(o, m, "deadline"),
            "batching": lambda o, m: heuristic(o, m, "deadline", best_k),
        }
        results = {}
        for label, policy in policies.items():
            results[label] = evaluate(config, seeds, policy)
        for i, agent in enumerate(agents):
            results[f"linear_q_{(i + 1) * 11}"] = evaluate(config, seeds, agent.act)
        # Separate RNG per episode makes random baseline reproducible regardless
        # of how many other policies are evaluated.
        results["random"] = [
            run_episode(
                config,
                seed,
                lambda o, m, rng=np.random.default_rng(seed): random_action(o, m, rng),
            )[0]
            for seed in seeds
        ]
        summary[name] = {}
        for label, rows in results.items():
            summary[name][label] = {
                metric: float(np.mean([r[metric] for r in rows])) for metric in rows[0]
            }
            for seed, row in zip(seeds, rows):
                all_rows.append(
                    dict(scenario=name, policy=label, scenario_seed=seed, **row)
                )
        rl = np.array(
            [[r["reward"] for r in results[f"linear_q_{s}"]] for s in (11, 22, 33)]
        )
        # Average over training replicas within each demand scenario, then paired
        # scenario bootstrap. This interval is conditional on these three replicas.
        diff = rl.mean(axis=0) - np.array([r["reward"] for r in results["batching"]])
        comparisons[name] = dict(
            mean_reward_difference=float(diff.mean()),
            paired_scenario_bootstrap_95_ci=paired_ci(diff),
            training_replica_mean_rewards=rl.mean(axis=1).tolist(),
        )
        summary[name]["linear_q_mean"] = {
            metric: float(
                np.mean([summary[name][f"linear_q_{s}"][metric] for s in (11, 22, 33)])
            )
            for metric in results["nearest"][0]
        }
    write_csv(root / "results" / "episodes.csv", all_rows)
    write_csv(root / "results" / "learning.csv", history)
    report = dict(
        config=asdict(c),
        training_episodes_per_seed=args.episodes,
        test_days_per_scenario=args.test_days,
        validation_days=args.validation_days,
        training_seeds=[11, 22, 33],
        validation=validation,
        selected_model_seed=[11, 22, 33][best_index],
        batching_threshold=best_k,
        batching_validation=baseline_scores,
        summary=summary,
        comparisons=comparisons,
        notes="Synthetic data only. CIs resample paired scenarios after averaging three trained policies; they are not full training-uncertainty intervals.",
    )
    (root / "results" / "summary.json").write_text(json.dumps(report, indent=2))
    # Predetermined replay seeds, never searched for flattering examples.
    replays = {}
    for name, config in scenarios.items():
        for seed in (200001, 200002, 200003):
            policies = {
                "Learned policy": agents[best_index].act,
                "Nearest delivery": lambda o, m: heuristic(o, m, "nearest"),
                "Earliest deadline": lambda o, m: heuristic(o, m, "deadline"),
                "Tuned batching": lambda o, m: heuristic(o, m, "deadline", best_k),
            }
            for label, policy in policies.items():
                metrics, frames = run_episode(config, seed, policy, record=True)
                replays[f"{name}|{seed}|{label}"] = dict(metrics=metrics, frames=frames)
    payload = dict(replays=replays, results=report)
    (root / "demo" / "replays.json").write_text(
        json.dumps(payload, separators=(",", ":"))
    )
    print(
        json.dumps(
            dict(
                selected_model_seed=report["selected_model_seed"],
                comparisons=comparisons,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
