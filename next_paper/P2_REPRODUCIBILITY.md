# Code and archived checks

The paper is `tex/p2_arxiv.tex`. Its numerical appendix checks the covered-call representation,
action and policy losses, comparative statics, resumption, and the shared-law bounds, including the
binary-component prior sweep. This document maps each check to its implementation and records.

All studies use specified finite laws and exact `fractions.Fraction` arithmetic. Decimal entries in
the paper's tables are rounded displays of the archived rational values. The joint Bellman references
and the arm-wise covered-call calculations use separate recursions. No learned surrogate is fitted.

## Implementations and records

Paths are relative to the repository root.

| Check | Implementation / protocol | Records |
|---|---|---|
| Independent trees: 192 starting instances | `next_paper/staged_pandora_exact.py`, `next_paper/p2_index_finite_study.py`, `next_paper/P2_INDEX_STUDY_PROTOCOL.md` | `next_paper/p2_index_study/records.jsonl`, `summary.json` |
| Staging grid: 660 cells and policy-loss checks | `next_paper/p2_staging_study.py`, `next_paper/report_p2_staging_study.py`, `next_paper/P2_STAGING_STUDY_PROTOCOL.md` | `next_paper/p2_staging_study/run_v1/` |
| Comparative statics and resumption | `next_paper/p2_comparative_statics_study.py` | `next_paper/p2_comparative_statics/run_v1/`: `records.jsonl`, `random_tree_check.json`, `summary.json` |
| Shared-law audit: 453 cells | `next_paper/p2_index_robustness.py`, `next_paper/run_p2_index_robustness.py`, `reports/p2_index_robustness_2026-10-26/PROTOCOL.md` | `reports/p2_index_robustness_2026-10-26/`: `models.json`, `records.json`, `summary.json` |
| Binary-component prior sweep: 240 cells | `next_paper/p2_concentration_sweep.py`; reuses the shared-law audit's 12 base models | `next_paper/p2_concentration_sweep/run_v1/`: `records.json`, `summary.json`, `design_freeze.json`, `source_snapshot/`, `artifact_manifest.json` |
| Positive regret and the shared-nuisance control | `next_paper/p2_shared_law_controls.py`; the positive-regret construction is also in the shared-law runner | Constructed rows in the shared-law records; `reports/p2_shared_law_controls/generic_comparison.json`, especially its `nuisance` entry |
| Collection-induced switching | `next_paper/p2_switching_scout.py`, `next_paper/p2_switching_witness.py`, `next_paper/p2_switching_certificate.py`, `verify.py` | `reports/p2_switching_certificate/`: `certificate.json` (cost interval), `fixed_cost_certificates.json` (three costs, four collection regimes) |

The paper gives the distributions, cost functions, collection sets, random seeds, tie conventions and
definitions needed to interpret the checks. The shared-law `models.json` additionally preserves every
realized base model.

## How the runners record their work

The staging runner writes the design and source hashes before the first cell, writes each completed
record, and records the environment. Its directory holds `design_freeze.json`, source snapshots,
summaries and `artifact_manifest.json`. `report_p2_staging_study.py` checks completeness, policy
accounting and source hashes. The comparative study verifies the staging manifest before rebuilding
its paired problems and records its own source snapshot and manifest.

The no-return solver prevents further advances of an arm once it is left, while retaining its observed
collectable rewards. The resumption-bound calculation carries the last active arm and the set of
previously left arms from the fixed starting history. The shared-law audit checks all feasible
prefixes, including histories the index would not reach. Action-order variants with the same observed
prefixes are equivalent for these static joint-law models. The refined sharing bound uses the selected
Bellman-optimal policy's occupancy; its advance-first, smaller-label tie convention therefore matters
even though the optimal value is unchanged.

The prior sweep fixes each base model's component laws, signal reliabilities and costs, uses all four
collection regimes, and varies only the initial prior over five values. Its 240 cells run the same
feasible-history and policy checks as the shared-law audit. Each uniform bound maximizes over all
feasible future histories; the displayed table takes the largest bound over the 48 cells at each prior.

## Checks

From the repository root, with `pytest` and `sympy` installed:

```sh
python -m pytest tests
```

The tests cover single-stage recovery, continuous intermediate collection, empty collectable suffixes,
atomic boundaries, own-history costs, policy losses, independent recovery, premature stopping under
sharing, root-only control failure, negative corrections and the nuisance control.

The switching certificate over the whole cost interval, and the fixed-cost certificates, are recomputed and compared with the archive by

```sh
python verify.py
```

The cost-interval certificate alone is `python next_paper/p2_switching_certificate.py --out <file>`.

It verifies the rational-function decisions on both sides of the algebraic split point, the split point
and interval endpoints, the collection-rights contrasts, threshold ordering and strict root-action
regions. Internal comparison-log counts can depend on traversal order; the certified regions and
inequalities are the mathematical outputs.

For fresh numerical runs, use each runner's `--out` option with a new directory and compare with the
committed records. The shared-law runner expects `PROTOCOL.md` in its output directory.

## Paper build

From `tex/`:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error p2_arxiv.tex
```
