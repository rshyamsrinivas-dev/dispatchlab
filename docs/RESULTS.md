# Evaluation report

All results below were measured in the included simulator. No operational or commercial results are claimed.

## Protocol

- Training: 3 independent replicas, 3,000 episodes each; alpha 0.015; gamma 0.98; zero initial weights.
- Validation: 80 fixed shifts (seeds 80000–80079); checkpoint evaluated every 250 episodes.
- Training scenario seeds: 0–2999, 20000–22999, 40000–42999, one range per replica.
- Each replica retains its highest validation-reward checkpoint. The demo model is the replica with highest validation reward.
- Saved checkpoints: [{'training_seed': 11, 'reward': 112.76875000000004, 'checkpoint': 2750}, {'training_seed': 22, 'reward': 112.50187500000007, 'checkpoint': 2750}, {'training_seed': 33, 'reward': 112.24875000000006, 'checkpoint': 2750}]. Demo model: training seed 11.
- Batching threshold chosen on validation from 1, 2, 3: 3. Validation rewards: {'1': 102.70250000000006, '2': 99.04250000000005, '3': 107.40500000000006}.
- Fixed learning hyperparameters; no test-based hyperparameter search.
- Test: 300 shifts each at seeds 100000–100299 (standard), 110000–110299 (surge), 120000–120299 (tight deadlines).
- Replay examples: fixed seeds 200001, 200002, 200003, separate from all benchmark shifts.
- Frozen policies, zero exploration, identical offered demand within every policy comparison.
- Baselines: uniform random feasible actions; nearest loaded delivery; earliest deadline; validation-tuned batching.
- Baselines are simple dispatch heuristics, not a mixed-integer optimiser or a production routing system.

## Metrics

All table entries are per-shift means. On-time rate and completion rate divide by ALL requests, including rejected and unfinished work. Ratios are calculated per shift and then averaged. Travel/delivery is travel ticks divided by completed deliveries; a zero-delivery shift uses denominator 1. Mean delay is conditional on completed deliveries, so it should be read alongside completion rate. Reward points are not money.

## Standard demand

| Policy | Reward | On time / requested | Completed / requested | Travel / delivery |
|---|---:|---:|---:|---:|
| Random feasible | 67.91 | 57.0% | 83.3% | 3.11 |
| Nearest delivery | 102.89 | 79.7% | 92.8% | 3.34 |
| Earliest deadline | 102.95 | 80.2% | 92.6% | 3.36 |
| Tuned batching | 99.12 | 74.2% | 91.1% | 3.01 |
| Linear Q (3-run mean) | 113.45 | 87.0% | 95.9% | 3.02 |

Paired reward differences (learned policy, averaged across 3 replicas, minus each baseline):

| Comparator | Difference | 95% paired scenario-bootstrap interval |
|---|---:|---:|
| Nearest delivery | +10.56 | [+8.34, +13.06] |
| Earliest deadline | +10.50 | [+8.00, +13.08] |
| Tuned batching | +14.33 | [+11.47, +17.30] |

Training-replica mean test rewards: 113.39, 113.05, 113.91.

## Demand surge

| Policy | Reward | On time / requested | Completed / requested | Travel / delivery |
|---|---:|---:|---:|---:|
| Random feasible | 15.02 | 33.5% | 59.2% | 3.00 |
| Nearest delivery | 67.64 | 49.5% | 70.2% | 3.12 |
| Earliest deadline | 64.54 | 47.7% | 69.2% | 3.15 |
| Tuned batching | 67.16 | 46.8% | 69.6% | 2.90 |
| Linear Q (3-run mean) | 95.37 | 63.8% | 76.5% | 2.74 |

Paired reward differences (learned policy, averaged across 3 replicas, minus each baseline):

| Comparator | Difference | 95% paired scenario-bootstrap interval |
|---|---:|---:|
| Nearest delivery | +27.73 | [+23.23, +32.24] |
| Earliest deadline | +30.83 | [+26.38, +35.05] |
| Tuned batching | +28.21 | [+23.67, +32.87] |

Training-replica mean test rewards: 95.02, 95.24, 95.84.

## Tighter deadlines

| Policy | Reward | On time / requested | Completed / requested | Travel / delivery |
|---|---:|---:|---:|---:|
| Random feasible | 47.36 | 37.3% | 83.6% | 3.19 |
| Nearest delivery | 89.83 | 58.4% | 93.1% | 3.41 |
| Earliest deadline | 89.65 | 56.2% | 92.9% | 3.41 |
| Tuned batching | 82.36 | 49.5% | 91.0% | 3.27 |
| Linear Q (3-run mean) | 101.08 | 71.2% | 95.5% | 3.14 |

Paired reward differences (learned policy, averaged across 3 replicas, minus each baseline):

| Comparator | Difference | 95% paired scenario-bootstrap interval |
|---|---:|---:|
| Nearest delivery | +11.25 | [+8.54, +14.16] |
| Earliest deadline | +11.44 | [+8.60, +14.46] |
| Tuned batching | +18.72 | [+15.22, +22.36] |

Training-replica mean test rewards: 101.28, 100.88, 101.09.

## Uncertainty and interpretation

Intervals use 2,000 bootstrap resamples of paired scenario-level differences, after averaging the three trained replicas. They condition on these replicas and the simulator distribution. They do not capture all training uncertainty, model-selection uncertainty, simulator mismatch, or deployment risk. Training-replica means are reported separately. Multiple comparisons are descriptive and not multiplicity-adjusted.

All three test scenarios favour the learned policies in average reward in this run. The standard-demand result also improves on-time delivery and completion compared with the simple heuristics. Validation-tuned batching generalises worse than the parameter-free baselines on the standard test set; all results are shown rather than hiding this reversal.

Demand surge raises arrival probability to 0.60. Tight deadlines replace 9–16 ticks with 6–10. The agent was not retrained on either stress scenario. These are limited distribution changes, not proof of general real-world robustness.

## Reproducibility

Bundled outputs: `episodes.csv` (6,300 policy-shift rows), `learning.csv`, `summary.json`, three checkpoint files plus the selected demo model, and replay JSON. Versions: Python 3.12.14, NumPy 2.3.5. Full plotting/adapter versions are in `requirements-repro.txt`.

Browser replay QA: HTML structure, embedded trajectories, and JavaScript syntax were checked. This execution environment blocked local HTML browser navigation, so end-to-end browser interaction was not verified here. The PNG/GIF outputs were rendered and visually inspected.
