# Learn the advanced version in layers

You do not need to start with the neural network. Keep the original beginner
notebook as the entry point, then follow this order.

## 1. Explain one fleet decision

Open the Fleet demo. Pause with a vehicle at the hub and read its battery,
loaded orders, deadlines, and the event log. Explain why each masked command
would be impossible. A low battery should restrict movement, not merely incur
a penalty after the agent strands a vehicle.

Read `fleet.py`: `reset`, `mask`, then `step`. Check the conservation identities:
requests = delivered + rejected + active; initial energy + charged − used =
current fleet energy. These are simulator correctness properties, independent
of whether any model learns well.

## 2. Understand the planner

Read `fleet_policies.py`. With three destinations there are at most six complete
stop orders. Enumerating them is cheap. Work out one route's travel, service,
deadline and energy costs by hand. Its success is useful evidence: some problems
benefit more from explicit planning than a neural network trained on limited data.

## 3. Move from linear Q-values to a neural value function

The beginner version estimates a weighted sum of explicit action features. The
advanced version takes a padded state vector through two 128-unit ReLU layers.
A value head estimates the state's general value; an advantage head estimates
the relative value of 64 joint commands. Combining them produces Q-values.
This is a **dueling architecture**, not two independent learning agents.

## 4. Understand Double DQN

The online network selects the best feasible next action. A separate target
network evaluates that action. Separating selection and evaluation reduces one
source of overestimation. It does not guarantee correct values or convergence.

The target is:

`reward / 20 + 0.97 * (1 - terminal) * target_Q(next_state, online_argmax)`

Every fleet step takes one tick, so there is no variable-duration discount in
this version. Both the action choice and the next-state maximisation use masks.

## 5. Why replay and a target network help

The replay buffer holds up to 30,000 transitions. Training samples batches of
64 instead of fitting only the immediately previous step. Learning begins after
1,024 transitions, with an update every four environment ticks. A slowly updated
target network changes the prediction target less abruptly. The model uses
Huber loss, gradient clipping, and Adam.

## 6. Know exactly how exploration works

Training uses epsilon exploration. Of exploratory decisions, 70% follow the
earliest-deadline rule and 30% sample a uniformly random feasible joint command.
All actions and their outcomes become normal replay transitions. There is no
supervised imitation loss and no rule fallback during evaluation. This is
**rule-guided exploration**, and it must be disclosed when discussing results.

Stage one trains from random initial weights. A second, validation-driven stage
starts from selected weights, resets replay and optimiser, lowers the learning
rate, and uses fresh training seeds. It is fine-tuning, not exact continuation
of the original optimiser state. Both stages occur before held-out evaluation.

## 7. Read results without overselling them

Checkpoint and batching choices use validation only. Compare reward, completion,
overall on-time service, priority service and distance. Do not say RL is better
because its architecture is newer. A planner win is a finding to explain, not
a number to hide. The original one-vehicle scores cannot be compared directly
with the new fleet task because the environment and objective changed.

## 8. Follow one API request

`POST /simulate` validates a seed, policy, and scenario, then executes a fresh
Python rollout and returns metrics and frames. It is not retrieving a stored
example. The HTML demo uses bundled replays offline and offers fresh simulations
when served by the local API. This separates environment logic, policy inference,
evaluation, API handling, and presentation.

## Interview defence

- Why constrain energy with action masks rather than reward alone?
- Why compare to a planner, and where is that planner still limited?
- Why does low test latency not imply production readiness?
- What assumptions make the state fully observable or approximately observable?
- What data would you need to calibrate arrivals, travel, service, and costs?
- How would you prevent test-set reuse while improving the agent?

References:

- [PyTorch DQN tutorial](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
- [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
- [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)
