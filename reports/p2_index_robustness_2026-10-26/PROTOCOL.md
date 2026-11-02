# Shared-law audit protocol

This is a mathematical implementation
check and an illustration of bound size, not a real-data performance study.

1. Enumerate the true joint-law Bellman problem separately from the product-marginal
   potential. Use rational arithmetic throughout. Check every feasible observation
   history and every available action, including histories the index never visits.
2. Verify the signed residual identity, the independent action-gap formula, the
   call-function and path-TV envelopes, the binary-latent envelope when applicable,
   and the telescoping identity for index, optimal, always-stop, and always-advance
   policies. Check both the refined occupancy bound and the uniform bounds.
3. Grid: 12 deterministic seeded binary-latent base models; 9 coupling values
   0, 1/100, 1/50, 1/20, 1/10, 1/5, 2/5, 3/5, 1; all four choices of collectable
   first stages. Full own-arm marginal path laws, stage costs and prior remain fixed
   within each coupling sweep. Conditional success probabilities shrink linearly
   towards their prior marginal. No selection based on the resulting performance.
4. Additional 12 arbitrary finite three-stage joint laws, with own-history cost
   overrides and both signs of dependence possible. These are not latent-factor
   instances and only test the general theorem.
5. Keep the tied-index positive-regret witness separate from the grid.
   It is a constructed illustration, not a randomly selected performance result.
6. Include explicit pure-information and parity diagnostic examples. The latter
   checks why root-only or pairwise independence is insufficient.

Seed: 20261026. No surrogate fitting, no recorded curves, no PB-GI comparison.
Report loose bounds, zero gaps and negative signed corrections as observed.
