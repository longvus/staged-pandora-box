# Staging grid: the value of staging

Completed the 660-cell full grid using known independent-arm laws and exact rational arithmetic. There is no learned surrogate and no recorded-curve input. The planning and observation laws coincide.

The staged index agrees exactly with direct joint Bellman optimization across 48,840 state evaluations and 74,160 advance-action checks. All maximal-index actions and action-loss identities pass. These are computational checks of specified finite laws, not a proof or an empirical comparison with another optimizer.

## Design and interpretation

Two three-stage arms reveal two noisy signals followed by a Bernoulli terminal reward. Signal accuracy has five levels; signal and terminal costs have four and five levels. Each of the three collection regimes has 100 constant-cost cells, 100 history-cost cells, and 20 deterministic controls. The terminal marginal is fixed at Bernoulli(1/2) in both stochastic families. All cells are retained; there are no Monte Carlo estimates or confidence intervals. State counts include separate solves on related grid points and are not independent observations.

For terminal-only collection and constant costs, the accuracy intervention isolates information. For intermediate collection, it also changes dependence between eligible rewards. In the history-cost family it changes the distribution of the mismatch surcharge. Comparisons between collection sets change which observations are eligible for the objective; they are mechanism interventions, not fair optimizer comparisons.

## Main findings

1. Under terminal-only collection and constant costs, utility never decreases as signals become more accurate: 62 of 80 adjacent comparisons increase and 18 tie. The mean perfect-minus-uninformative utility difference over the 20 cost pairs is 49/320 = 0.153125. This is a descriptive average over the chosen grid, not an estimate for practical tasks.
2. At k=.02 and c=.6, terminal-only utility rises from zero with uninformative signals to 171/800 = 0.213750 with perfect signals. The latter policy opens 1.5 arms in expectation but completes only 0.75, illustrating adaptive abandonment. Terminal usage can increase with information when it makes an otherwise unprofitable inspection worthwhile; it need not always decrease.
3. With every stage collectable and constant costs, utility decreases in 66 of the 80 adjacent accuracy comparisons. Here early readings are rewards as well as information: changing accuracy changes their dependence and the opportunities to retain a high early score. This intervention does not isolate information alone. The index still matches Bellman exactly in every case.
4. Deterministic controls show that later collection matters even without learning. At k=.02 and c=.8, terminal-only utility is 0.16, whereas every-stage collection yields 0.46 by collecting the intermediate 0.5 reward and stopping before the expensive terminal stage. These are different collection objectives applied to the same underlying process.
5. History-dependent costs preserve exact index/Bellman agreement throughout all 300 cells. Expanding the collectable set never lowers optimal utility at a fixed law and cost setting; the audit checks this across all three regimes.

## Signal-accuracy effects over the complete cost grid

Counts below describe 80 adjacent-accuracy comparisons per row (20 cost pairs times four increments). Endpoint contrasts compare a=1 with a=1/2 over the same 20 cost pairs.

| Cost law | Collection | Increases / ties / decreases | Mean endpoint utility change | Endpoint range |
|---|---|---:|---:|---:|
| Constant costs | Terminal only: {3} | 62 / 18 / 0 | 0.153125 | [0.000000, 0.322813] |
| Constant costs | Mixed: {1,3} | 30 / 34 / 16 | 0.036094 | [-0.030312, 0.109375] |
| Constant costs | Every stage: {1,2,3} | 14 / 0 / 66 | -0.041016 | [-0.086563, 0.021875] |
| History-dependent costs | Terminal only: {3} | 64 / 16 / 0 | 0.169004 | [0.000000, 0.372031] |
| History-dependent costs | Mixed: {1,3} | 31 / 36 / 13 | 0.046270 | [-0.010000, 0.161562] |
| History-dependent costs | Every stage: {1,2,3} | 15 / 0 / 65 | -0.042510 | [-0.081250, 0.042578] |

## Illustrative cells

These transparent examples accompany the complete-grid summaries; they are not a selected performance benchmark.

| Cell | Exact utility | Expected advances by stage | Expected opened but unfinished arms |
|---|---:|---|---:|
| constant_cost/terminal/a=1/2/k=1/50/c=3/5 | 0 (0.000000) | 0, 0, 0 | 0 |
| constant_cost/terminal/a=1/k=1/50/c=3/5 | 171/800 (0.213750) | 3/2, 3/4, 3/4 | 3/4 |
| constant_cost/all/a=1/2/k=1/50/c=3/5 | 23/50 (0.460000) | 3/2, 3/2, 0 | 3/2 |
| constant_cost/all/a=1/k=1/50/c=3/5 | 321/800 (0.401250) | 3/2, 3/4, 0 | 3/2 |
| deterministic/terminal/a=None/k=1/50/c=4/5 | 4/25 (0.160000) | 1, 1, 1 | 0 |
| deterministic/all/a=None/k=1/50/c=4/5 | 23/50 (0.460000) | 1, 1, 0 | 1 |

## Action-loss diagnostics

Each row is a uniform mean over its declared family/collection grid. Stop after one advance uses the optimal initial action, then stops. Force lowest index first forces one initial action and resumes the optimal rule. One extra at stop forces the cheapest remaining advance at the first optimal stopping point, then resumes the optimal rule. These are constructed diagnostic deviations, not published acquisition methods. All four policies, including zero-loss cases, are recorded.

| Family | Collection | Diagnostic | Mean utility loss | Stop loss | Advance loss | Positive loss cells |
|---|---|---|---:|---:|---:|---:|
| Constant costs | Terminal only: {3} | Exact index | 0.000000 | 0.000000 | 0.000000 | 0/100 |
| Constant costs | Terminal only: {3} | Stop after one advance | 0.259936 | 0.259936 | 0.000000 | 76/100 |
| Constant costs | Terminal only: {3} | Force lowest index first | 0.031911 | 0.000000 | 0.031911 | 100/100 |
| Constant costs | Terminal only: {3} | One extra at stop | 0.044337 | 0.000000 | 0.044337 | 99/100 |
| Constant costs | Mixed: {1,3} | Exact index | 0.000000 | 0.000000 | 0.000000 | 0/100 |
| Constant costs | Mixed: {1,3} | Stop after one advance | 0.189790 | 0.189790 | 0.000000 | 100/100 |
| Constant costs | Mixed: {1,3} | Force lowest index first | 0.010551 | 0.000000 | 0.010551 | 100/100 |
| Constant costs | Mixed: {1,3} | One extra at stop | 0.043894 | 0.000000 | 0.043894 | 100/100 |
| Constant costs | Every stage: {1,2,3} | Exact index | 0.000000 | 0.000000 | 0.000000 | 0/100 |
| Constant costs | Every stage: {1,2,3} | Stop after one advance | 0.285875 | 0.285875 | 0.000000 | 100/100 |
| Constant costs | Every stage: {1,2,3} | Force lowest index first | 0.013787 | 0.000000 | 0.013787 | 100/100 |
| Constant costs | Every stage: {1,2,3} | One extra at stop | 0.121044 | 0.000000 | 0.121044 | 100/100 |
| History-dependent costs | Terminal only: {3} | Exact index | 0.000000 | 0.000000 | 0.000000 | 0/100 |
| History-dependent costs | Terminal only: {3} | Stop after one advance | 0.234336 | 0.234336 | 0.000000 | 74/100 |
| History-dependent costs | Terminal only: {3} | Force lowest index first | 0.034326 | 0.000000 | 0.034326 | 100/100 |
| History-dependent costs | Terminal only: {3} | One extra at stop | 0.048907 | 0.000000 | 0.048907 | 100/100 |
| History-dependent costs | Mixed: {1,3} | Exact index | 0.000000 | 0.000000 | 0.000000 | 0/100 |
| History-dependent costs | Mixed: {1,3} | Stop after one advance | 0.172901 | 0.172901 | 0.000000 | 100/100 |
| History-dependent costs | Mixed: {1,3} | Force lowest index first | 0.010140 | 0.000000 | 0.010140 | 100/100 |
| History-dependent costs | Mixed: {1,3} | One extra at stop | 0.049220 | 0.000000 | 0.049220 | 100/100 |
| History-dependent costs | Every stage: {1,2,3} | Exact index | 0.000000 | 0.000000 | 0.000000 | 0/100 |
| History-dependent costs | Every stage: {1,2,3} | Stop after one advance | 0.267573 | 0.267573 | 0.000000 | 100/100 |
| History-dependent costs | Every stage: {1,2,3} | Force lowest index first | 0.013964 | 0.000000 | 0.013964 | 100/100 |
| History-dependent costs | Every stage: {1,2,3} | One extra at stop | 0.136326 | 0.000000 | 0.136326 | 100/100 |
| Deterministic controls | Terminal only: {3} | Exact index | 0.000000 | 0.000000 | 0.000000 | 0/20 |
| Deterministic controls | Terminal only: {3} | Stop after one advance | 0.541250 | 0.541250 | 0.000000 | 19/20 |
| Deterministic controls | Terminal only: {3} | Force lowest index first | 0.054688 | 0.000000 | 0.054688 | 20/20 |
| Deterministic controls | Terminal only: {3} | One extra at stop | 0.048438 | 0.000000 | 0.048438 | 19/20 |
| Deterministic controls | Mixed: {1,3} | Exact index | 0.000000 | 0.000000 | 0.000000 | 0/20 |
| Deterministic controls | Mixed: {1,3} | Stop after one advance | 0.541250 | 0.541250 | 0.000000 | 19/20 |
| Deterministic controls | Mixed: {1,3} | Force lowest index first | 0.054688 | 0.000000 | 0.054688 | 20/20 |
| Deterministic controls | Mixed: {1,3} | One extra at stop | 0.048438 | 0.000000 | 0.048438 | 19/20 |
| Deterministic controls | Every stage: {1,2,3} | Exact index | 0.000000 | 0.000000 | 0.000000 | 0/20 |
| Deterministic controls | Every stage: {1,2,3} | Stop after one advance | 0.626250 | 0.626250 | 0.000000 | 20/20 |
| Deterministic controls | Every stage: {1,2,3} | Force lowest index first | 0.054688 | 0.000000 | 0.054688 | 20/20 |
| Deterministic controls | Every stage: {1,2,3} | One extra at stop | 0.054688 | 0.000000 | 0.054688 | 20/20 |

## What this strengthens

The study adds a controlled explanation of signal value, adaptive abandonment, designated collection, and history-dependent costs. It also quantifies the stop-versus-advance loss accounting on specified deviations. It does not establish computational scaling, robustness to model misspecification, or performance on practical learning curves. The roughly seven-second local execution time is not a scaling result; every arm here has only three binary stages.

## Figures

`staging_signal_utility.pdf`: utility, change relative to uninformative signals, and terminal usage for constant costs at the protocol-specified slice k=.02; every terminal cost and accuracy is shown. `staging_initial_indices.pdf` uses the same slice for both cost laws. `staging_cost_grid.pdf` adds a c=.6 heatmap, and `staging_deterministic_controls.pdf` shows all deterministic cells. The c=.6 display slice and the illustrative table are for exposition; the complete-grid analysis and records are the evidence.

## Reproduction

From the project root:

```sh
python -m pytest tests/test_p2_staging_study.py tests/test_staged_pandora_exact.py -q
python next_paper/p2_staging_study.py --out next_paper/p2_staging_study/rerun
python next_paper/report_p2_staging_study.py --run next_paper/p2_staging_study/rerun
```

The runner refuses an existing output directory. Source and protocol hashes are frozen before evaluation, records are checkpointed cell by cell, and the report verifies the full grid and loss accounting. See `design_freeze.json`, `summary.json`, `audit.json`, and `artifact_manifest.json` for the run's design, sources and hashes.
