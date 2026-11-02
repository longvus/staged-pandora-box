# Independent-tree verification protocol

Executed by `p2_index_finite_study.py`.

* 24 seeds, starting from 2026102400, for each of four collectable regimes:
  all stages of a three-stage arm; terminal only on a four-stage arm;
  stages {1,3}; and stages {2,4}.
* Two independent arms, binary observation trees. At each history, two distinct
  readings are sampled from {0,1/4,1/2,3/4,1}; branch probabilities are 1/4,
  1/2, or 3/4; next-stage cost is sampled independently at each history from
  {1/100,3/100,3/50,1/10}. Thus within-arm predictions and costs may depend on
  the whole history.
* Evaluate both the empty state with incumbent zero and the state after three
  random feasible advances. The arm choices and observations for that state
  sampling do not use a stopping policy or any index. Both states belong to
  the same seed; no claim of their statistical independence is made.
* A direct Bellman solver computes the optimum independently of covered calls.
  Check the covered-call value, all maximal-index actions, and each action-loss
  formula at every reachable state, in exact rational arithmetic.
* Evaluate the staged index and two actual collapsed-completion index policies:
  commit to the next collectable stage, or advance once and recompute. At every
  completion the ordinary reservation price uses its marginal reward and its
  expected total cost. Rank by that price, not by expected improvement net cost.
* Decompose every policy's expected loss into its terminal stop loss and its
  accumulated advance losses, checking the total against the Bellman value.
* Report all failures (assertions abort on a nonzero exact discrepancy), state
  and action counts, and descriptive pooled loss shares for each policy/regime.
  Report computational time separately; these small exact-arithmetic runs are
  not representative of GP decision latency.
* Also verify an explicit history-dependence example: two own-arm histories
  have latest reading 1/2 and the same stage/incumbent, but terminal rewards 0
  versus 1, hence different indices. This illustrates why the last scalar
  reading alone is not always a sufficient Markov state.

All records and machine-readable summaries are retained.
