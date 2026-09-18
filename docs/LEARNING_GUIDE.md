# Start here

You do not need to understand neural networks to work through this project.
Use Python functions, lists, dictionaries, NumPy arrays, and basic probability.

## 1. Watch before reading the algorithm

Open `demo/index.html` in your browser. Click Play. Pause when one policy returns
to the depot while the other continues a delivery. Look at loaded parcels,
deadlines, and the queue. Ask: what future choices did that action remove?

The animated GIF in the README is also generated from this simulator.

## 2. Learn the five objects

| Term | Here |
|---|---|
| Environment | Orders, travel times, depot, vehicle, deadlines, and clock |
| Agent | The program choosing the next feasible action |
| State | Everything currently observable about that world |
| Action | Wait, return, or visit a delivery destination |
| Reward | The score earned or lost after acting |
| Episode | One complete 48-tick shift |
| Policy | A rule for choosing an action from the current observation |

The state is not a prediction target. The agent acts, changes the world, observes
a reward and a new state, and repeats. A trip made now removes capacity to handle
orders that arrive during travel. That is why this is sequential decision-making.

## 3. Inspect one step

From the repository root:

```python
from dispatchlab import DispatchEnv
env = DispatchEnv()
observation, info = env.reset(seed=200001)
print(observation)
print(info['action_mask'])
next_observation, reward, terminated, truncated, info = env.step(0)
print(next_observation, reward)
```

Now read `environment.py` in this order: `reset`, `action_mask`, `step`. Follow one
parcel by ID. It cannot arrive before the vehicle finishes its trip. At the end,
every requested order must be delivered, rejected, or still unfinished.

## 4. Understand a simple policy first

Read `policies.py`. Nearest delivery selects the closest destination among loaded
parcels. Earliest deadline prioritises urgency. Both return when empty. The batching
rule can wait at the depot for more parcels, unless deadlines or closing time make
waiting unattractive. Its threshold is chosen using validation shifts only.

These are meaningful baselines. RL has to earn the extra complexity.

## 5. Understand the learner

The agent uses **linear semi-gradient Q-learning**, not a Q-table or a neural network.
It approximates the value of an action as a weighted sum of 23 readable features:

`Q(state, action) = weights dot features(state, action)`

Features describe current cargo, travel needed, matching deliveries, urgency,
depot backlog, and time left. The feature design supplies domain knowledge; the
weights are learned from zero through experience. It is not feature-free discovery.

To explore, the agent sometimes chooses a random feasible action. Otherwise, it
chooses the action with the highest estimated Q-value. Exploration decreases
during training and is zero during validation and evaluation.

For a trip lasting `d` ticks, accumulate the discounted rewards along the trip:

`G = reward_1 + gamma * reward_2 + ... + gamma**(d-1) * reward_d`

Then the target is `G + gamma**d * max Q(next_state, next_action)`. At the end of
the shift, omit the future-value term. The prediction error moves the weights
towards this target. The update is divided by `1 + features dot features` to
moderate large steps. Read `train_episode` and then `update` in `agent.py`.

This decision-to-decision update is **semi-Markov**: a dispatch choice can occupy
several ticks. The code discounts elapsed travel correctly rather than treating a
five-tick journey as if it took one tick.

Linear off-policy Q-learning has no general convergence guarantee. The bundled
learning curves show early instability; finite-weight checks and validation
checkpoint selection help detect problems but do not prove convergence.

## 6. Inspect the trained policy

```bash
python scripts/inspect_shift.py
```

This prints feasible action values for every decision in a fixed replay shift.
They are learned estimates of discounted reward, not calibrated probabilities.
Individual feature weights are not causal effects because features overlap.

## 7. Reproduce, then change one assumption

Run the benchmark exactly as documented before experimenting. To conduct a new
experiment, use a different output directory and fresh validation/test seed
ranges when you have already inspected the original test results. Do not tune on
the bundled test shifts and continue calling them unseen.

Useful first extensions: add a nonzero depot loading time; change the reward
weights; replace independent arrivals with clustered requests. Change one thing,
retrain, compare all baselines, and explain what changed.

## Interview questions to answer yourself

1. Why is nearest delivery not always the best choice?
2. What information is visible, and what would count as future-data leakage?
3. Why does the agent need time remaining in its state?
4. Why can reward improve while a customer-facing metric gets worse?
5. Why are identical order streams necessary for policy comparisons?
6. How could the simulator make a learned policy look better than it is?
7. What changes would be necessary before even piloting this in real operations?

## References

- [Gymnasium custom environments](https://gymnasium.farama.org/introduction/create_custom_env/)
- [Gymnasium environment API](https://gymnasium.farama.org/api/env/)
- [Sutton and Barto, Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)

This is my original educational implementation. It does not reproduce a
published logistics benchmark, company system, or private dataset.
