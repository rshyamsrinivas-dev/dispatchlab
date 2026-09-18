"""Continue validation-selected policies with fresh replay and a lower step size.

This stage uses only training and validation data. Run before held-out evaluation.
Optimiser/replay reset is explicit; this is fine-tuning, not exact resumption.
"""

from pathlib import Path
import sys, json, csv
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dispatchlab.deep_agent import DoubleDQN
from dispatchlab.fleet import FleetEnv, FleetConfig, encode
from dispatchlab.fleet_policies import fleet_heuristic
from dispatchlab.fleet_experiment import score
from dispatchlab.experiment import write_csv

root = Path(__file__).resolve().parents[1] / "advanced"
manifest = json.loads((root / "results/training.json").read_text())
if "refinement" in manifest:
    raise SystemExit(
        "Refinement already recorded. Reproduce from the stage-one training command first."
    )
with (root / "results/learning.csv").open() as f:
    history = list(csv.DictReader(f))
cfg = FleetConfig()
episodes = 2000
for index, seed in enumerate(manifest["replicas"]):
    path = root / "models" / f"ddqn_{seed}.pt"
    agent = DoubleDQN(seed=seed)
    old = DoubleDQN.load(path)
    agent.online.load_state_dict(old.online.state_dict())
    agent.target.load_state_dict(old.online.state_dict())
    for group in agent.optimizer.param_groups:
        group["lr"] = 1e-4
    rng = np.random.default_rng(seed + 5000)
    env = FleetEnv(cfg)
    steps = 0
    losses = []
    returns = []
    selected = manifest["selected"][index]
    best = selected["validation_reward"]
    for ep in range(episodes):
        o, m = env.reset(600000 + index * 50000 + ep)
        epsilon = max(0.05, 0.25 - 0.2 * ep / (episodes * 0.8))
        while not env.done:
            if rng.random() < epsilon:
                a = (
                    fleet_heuristic(o, m, "deadline")
                    if rng.random() < 0.7
                    else int(rng.choice(np.flatnonzero(m)))
                )
            else:
                a = agent.act(o, m)
            nxt, r, done, nmask = env.step(a)
            agent.buffer.add(encode(o), a, r / 20, encode(nxt), done, nmask)
            steps += 1
            if steps % 4 == 0:
                loss = agent.optimise()
                if loss is not None:
                    losses.append(loss)
            o, m = nxt, nmask
        returns.append(env.metrics["reward"])
        if (ep + 1) % 200 == 0:
            s = score(
                cfg, range(900000, 900000 + manifest["validation_days"]), agent.act
            )
            history.append(
                dict(
                    seed=seed,
                    episode=manifest["train_episodes"] + ep + 1,
                    validation_reward=s,
                    training_reward=float(np.mean(returns[-200:])),
                    loss=float(np.mean(losses[-1000:])),
                )
            )
            if s > best:
                best = s
                agent.save(path)
                selected.update(
                    episode=manifest["train_episodes"] + ep + 1, validation_reward=best
                )
            print(
                f"refine seed={seed} extra={ep + 1} validation={s:.1f} retained={best:.1f}",
                flush=True,
            )
    selected["gradient_updates"] += agent.updates
winner = max(manifest["selected"], key=lambda x: x["validation_reward"])
DoubleDQN.load(root / "models" / f"ddqn_{winner['seed']}.pt").save(
    root / "models/ddqn.pt"
)
manifest["demo_seed"] = winner["seed"]
manifest["refinement"] = dict(
    episodes_per_seed=episodes,
    learning_rate=0.0001,
    epsilon_start=0.25,
    epsilon_end=0.05,
    training_seed_starts=[600000, 650000, 700000],
    optimiser_and_replay_reset=True,
)
manifest["total_episodes_per_seed"] = manifest["train_episodes"] + episodes
(root / "results/training.json").write_text(json.dumps(manifest, indent=2))
write_csv(root / "results/learning.csv", history)
