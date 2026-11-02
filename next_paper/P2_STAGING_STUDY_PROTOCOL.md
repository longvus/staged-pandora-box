# Staging grid protocol

No learned surrogate, recorded curves, or PB-GI comparison is used. The
runner writes a timestamped design/source hash before evaluating the first
cell; an existing output directory cannot be reused.

## Laws and full grid

Two independent arms each have three stages. In each stochastic arm, an
unobserved terminal reward H is Bernoulli(1/2). Conditional on H, the binary
signals S1 and S2 are independent and each equals H with probability a.
Observed readings are (2 S1/5, 3 S2/5, H). Conditional terminal probabilities
are obtained by exact Bayes conditioning on both signals. Thus, except at
special parameter values, the last scalar reading alone is not sufficient.
Impossible zero-probability histories are omitted, including at a=1.

Full factorial grid, without seeds or sampling:

* a in {1/2, 5/8, 3/4, 7/8, 1} (uninformative through perfect).
* Base cost of each signal k in {1/200, 1/50, 1/20, 1/10}.
* Base terminal cost c in {1/20, 1/5, 2/5, 3/5, 4/5}.
* Collectable stages: {3}, {1,3}, and {1,2,3}.
* Arm cost multipliers: 1 and 5/4; all rewards/costs are in the same units.
* Initial incumbent is zero; ties stop, then favor the smaller arm id.

There are 300 cells in each of two families:

1. Constant stage costs: multiplied costs (k,k,c).
2. History-dependent costs: multiplied costs
   (k, k(1+S1/2), c(1+1{S1 != S2}/2)). The latter are costs known from
   observed history before the corresponding next advance.

Sixty deterministic controls cross the same k,c,collection grids: arm 1
has readings (0,1/2,1), arm 2 has (0,3/8,3/4), and costs (k,k,c) with
the same arm multipliers. These isolate later collection without learning.
Total: 660 cells. The full grid is retained, including ties, stopping cases,
zero benefit, and any nonmonotonic effect. There is no outcome-based selection.

Within a family and collection regime, the terminal marginal remains fixed
when a changes. Only for terminal-only collection with constant costs does
this isolate signal information without changing collectable reward
dependence or realized cost distributions. For intermediate collection,
changing a also changes dependence between collectable readings; collection
regimes themselves have different sets of eligible rewards. Do not call all
these contrasts pure information gains. In the history-cost family, changing
a also changes the probability of the mismatch surcharge.

## Independent reference, evaluation, and diagnostics

Use exact Fraction arithmetic throughout. The existing direct joint Bellman
solver enumerates actions without using reservations. Independently compute
the arm-wise covered-call values and maximal-index policy. At every state
visited by the joint solver, check value, all maximal-index actions, and
advance-loss formulas. Evaluate the index policy's expected final reward,
cost, utility, number of advances at each stage, and opened-but-unfinished
arms by exact probability-weighted recursion. No Monte Carlo confidence
intervals are relevant to these completely enumerated laws.

Evaluate three predefined diagnostic deviations, all on the same true law:

* stop_after_one: take the index policy's initial action, if any, then stop.
* lowest_index_first: force the initially lowest-index available arm once,
  even when stopping is optimal, then follow the exact index policy.
* one_extra_at_stop: follow the exact index policy until its first stop;
  if an arm remains, advance the cheapest currently available arm once,
  breaking ties by id, then resume the exact index policy.

These are constructed action-loss diagnostics, not competing published
optimization algorithms. Verify separately for every cell and diagnostic
that Bellman optimum minus policy utility equals accumulated stop loss plus
accumulated advance loss. Policy evaluation recomputes state action values
through the independent Bellman reference. Report all three, including
zero-loss/tied decisions. Aggregate means are uniform descriptive means over
the declared grid, not estimates for a task population.

## Recovery checks and outputs

Tests cover classical one-stage prices, a continuous-law example, analytic uninformative/perfect terminal-only signal
cases, Bayes dependence on both readings, history-dependent costs, and a
deterministic intermediate-collection case. All-collectable and terminal-only
cells check special cases against direct Bellman, not empirical superiority.

Outputs: per-cell rational records; summary and audit; grid figures for
utility, gain relative to a=1/2, initial index and expected stage usage;
history-cost and deterministic-control panels; a stop/advance-loss table;
and a standalone findings report. Figure slices use k=1/50, with every c
and a. Full records cover every k.

Use one process, checkpoint each completed cell, and record the
environment. This is not a scaling benchmark. Any failure must stop the
run and remain recorded; a changed design or source requires a new output
directory and design hash.
