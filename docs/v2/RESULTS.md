# What the fleet benchmark showed

I kept the route planner in the comparison because a logistics project needs a credible operational baseline. The neural model has to justify itself on measured outcomes.

## Experiment setup

- Three neural replicas: seeds 101, 202, 303. Each trained for 1,500 episodes, then 2,000 refinement episodes on fresh training scenarios.
- Stage-one demand seeds: 300000–301499, 350000–351499, 400000–401499. Refinement: 600000–601999, 650000–651999, 700000–701999.
- Validation: 40 fixed shifts, seeds 900000–900039. Checkpoints and planner batching are selected only here. Refinement was chosen after observing validation weakness, before opening the test set.
- Refinement resets replay and Adam state, uses a 0.0001 learning rate, and retains an earlier checkpoint if validation gets worse.
- Test: 150 fresh shifts per condition, starting at 1000000, 1001000, 1002000, and 1003000. Every policy faces matched exogenous streams.
- Replay examples: 2000001, 2000002, 2000003. These were fixed independently of results.
- Guided exploration: 70% of exploratory actions use earliest-deadline dispatch, 30% are random feasible commands. Evaluation has no heuristic fallback.
- Neural checkpoints contain learned weights, not demonstration recordings. The selected demo model is chosen by validation reward.

## Results

Percentages below average per-shift ratios. Their denominator is all requested orders, including rejected and unfinished work. Priority service uses all priority requests. Reward is in synthetic points.

### Standard

| Policy | Reward | On time | Completed | Priority on time | Distance / delivery |
|---|---:|---:|---:|---:|---:|
| Nearest | 80.86 | 39.7% | 74.0% | 59.3% | 3.40 |
| Earliest deadline | 60.59 | 37.3% | 71.7% | 53.8% | 3.55 |
| Route planner | 75.51 | 38.7% | 72.8% | 58.8% | 3.41 |
| Double DQN | 15.74 | 37.1% | 66.7% | 55.2% | 3.55 |

Neural minus planner reward: **-59.77**, paired scenario-bootstrap 95% interval **[-70.08, -49.96]**.

### Demand surge

| Policy | Reward | On time | Completed | Priority on time | Distance / delivery |
|---|---:|---:|---:|---:|---:|
| Nearest | -132.67 | 23.4% | 51.0% | 41.7% | 3.39 |
| Earliest deadline | -155.37 | 21.4% | 49.5% | 35.1% | 3.53 |
| Route planner | -120.97 | 24.2% | 51.1% | 45.0% | 3.34 |
| Double DQN | -199.77 | 21.8% | 46.1% | 37.4% | 3.51 |

Neural minus planner reward: **-78.80**, paired scenario-bootstrap 95% interval **[-89.18, -67.77]**.

### Traffic disruption

| Policy | Reward | On time | Completed | Priority on time | Distance / delivery |
|---|---:|---:|---:|---:|---:|
| Nearest | -94.09 | 23.0% | 55.3% | 35.9% | 3.53 |
| Earliest deadline | -111.13 | 21.1% | 54.0% | 28.7% | 3.72 |
| Route planner | -83.11 | 23.1% | 55.9% | 39.1% | 3.48 |
| Double DQN | -132.43 | 21.1% | 51.2% | 32.5% | 3.55 |

Neural minus planner reward: **-49.31**, paired scenario-bootstrap 95% interval **[-57.87, -40.44]**.

### Limited battery

| Policy | Reward | On time | Completed | Priority on time | Distance / delivery |
|---|---:|---:|---:|---:|---:|
| Nearest | 27.24 | 32.9% | 67.8% | 47.1% | 3.45 |
| Earliest deadline | 4.52 | 29.5% | 65.4% | 38.3% | 3.60 |
| Route planner | -99.94 | 29.8% | 53.7% | 40.6% | 3.39 |
| Double DQN | -85.99 | 32.1% | 56.2% | 44.1% | 3.74 |

Neural minus planner reward: **+13.95**, paired scenario-bootstrap 95% interval **[-10.14, +38.24]**.

## Interpretation

The route planner is stronger than the neural agent in the standard scenario in this run. I am retaining the result rather than presenting a more complex model as an automatic improvement. The experiment demonstrates a working RL environment and evaluation pipeline; it does not establish that this DQN should replace a planner.

The original one-vehicle results belong to a different task. Its 87% on-time result cannot be carried over to the fleet version. More vehicles, uncertain travel, charging, service times and a different reward make this a harder problem.

The next learning experiments would be a more structured state representation, a hierarchical policy that selects dispatch strategies, and a larger predeclared training budget. Those are future work, not completed features.

## Uncertainty and scope

Intervals use 2,000 paired resamples of scenario-level differences after averaging the three neural replicas. They condition on those trained policies and this simulator. They do not include full training or simulator uncertainty. The report also retains each replica separately. Comparisons are descriptive, without multiplicity adjustment.

No real demand, road network, driver data, or business savings were used or measured. The route planner is not a globally optimal solver. The model uses known operating parameters and a fixed-size state encoding. See ENVIRONMENT.md for the information boundaries.

## Verification

Tests cover battery conservation, parcel accounting, simultaneous travel, service timing, action masks, model save/load, neural gradient updates, API validation, fresh reproducible simulations, and the original Gymnasium adapter. API requests are tested directly. Local HTML browser navigation is blocked in this execution environment, so browser end-to-end playback has not been verified. The static assets are rendered from simulator data and visually inspected.

Raw episode rows, learning curves, selected checkpoints, training settings, and SHA-256 checksums are included under advanced/.
