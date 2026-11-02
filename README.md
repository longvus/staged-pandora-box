# The Staged Pandora's Box

Code, exact-arithmetic verification records and LaTeX source for the paper
*The Staged Pandora's Box: Reservation Indices for Multi-Fidelity Bayesian Optimization*.

## Layout

| Path | Contents |
|---|---|
| `next_paper/staged_pandora_exact.py` | Finite-tree model, covered-call recursion, reservation index, index policy and the independent joint Bellman solver |
| `next_paper/p2_*.py` | The numerical checks of the paper's appendix, one runner per study |
| `next_paper/p2_*/`, `reports/` | Their records: per-cell exact values, summaries, design hashes and manifests |
| `next_paper/P2_REPRODUCIBILITY.md` | Which runner and record belong to which check, and the commands that rerun them |
| `tests/` | Analytic examples and exact-arithmetic tests |
| `tex/` | The paper (`p2_arxiv.tex`, sections in `p2_index/`) |

Every study uses fully specified finite laws and `fractions.Fraction` arithmetic; no data is downloaded
and no surrogate is fitted. Decimal entries in the paper's tables are rounded displays of the archived
rational values.

## Setup

```bash
conda env create -f environment.yml      # Python 3.12, env name staged-pandora
conda activate staged-pandora
```

`sympy` is needed only by the switching certificate and `matplotlib` only by the staging report.

## Tests

```bash
python -m pytest tests
```

## Rerun a study

Each runner refuses an existing output directory, so give `--out` a new path and compare it with the
committed records. From the repository root:

```bash
python next_paper/p2_index_finite_study.py --out next_paper/p2_index_study_rerun
python next_paper/p2_staging_study.py --out next_paper/p2_staging_study/rerun
python next_paper/report_p2_staging_study.py --run next_paper/p2_staging_study/rerun
python next_paper/p2_comparative_statics_study.py --out next_paper/p2_comparative_statics/rerun
python next_paper/run_p2_index_robustness.py --out reports/shared_law_rerun       # needs PROTOCOL.md copied there
python next_paper/p2_concentration_sweep.py --out next_paper/p2_concentration_sweep/rerun
python next_paper/p2_shared_law_controls.py
python next_paper/p2_switching_certificate.py --out reports/switching_certificate_rerun.json
```

The studies are deterministic: a rerun reproduces the committed per-cell records exactly.
`next_paper/P2_REPRODUCIBILITY.md` maps each check in the paper to its runner and records.

## Verify the switching proof

`verify.py` recomputes the cost-interval certificate behind Proposition 6 (Appendix D.2), checks the
fixed-cost Bellman certificates under all four collection regimes against a separate scalar solver,
and compares the mathematical conclusions with the archived certificates in
`reports/p2_switching_certificate/`. It needs `sympy` and about a minute:

```bash
python verify.py            # prints VERIFIED and writes verification/
```

Run it without Python's `-O` option: the proof implementation uses assertions as checks.

## Build the paper

```bash
cd tex
latexmk -pdf p2_arxiv.tex
```

## License

MIT, see `LICENSE`.
