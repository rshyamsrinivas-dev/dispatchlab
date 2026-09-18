"""Run a trained policy and print each decision, with its learned action values."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dispatchlab.agent import LinearQAgent
from dispatchlab.environment import DispatchEnv, ACTION_NAMES, NAMES

root = Path(__file__).resolve().parents[1]
agent = LinearQAgent.load(root / "models" / "linear_q.json")
env = DispatchEnv()
obs, info = env.reset(seed=200001)
done = False
while not done:
    action = agent.act(obs, info["action_mask"])
    if not obs["travel_remaining"]:
        print(
            f"t={obs['time']:02} {NAMES[obs['location']]:5} orders={len(obs['orders'])} -> {ACTION_NAMES[action]}"
        )
        print(
            {
                ACTION_NAMES[k]: round(v, 2)
                for k, v in agent.values(obs, info["action_mask"]).items()
            }
        )
    obs, reward, done, _, info = env.step(action)
print(env.metrics)
