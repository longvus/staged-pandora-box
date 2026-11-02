#!/usr/bin/env python3
"""Concentration sweep for the shared-component corollary.

Reuses the 12 latent base models of the shared-law study at full component
contrast and all four collection choices, and varies only the prior
p = P(Theta = 1) toward zero. Component laws, costs and collection are fixed;
only the belief about the shared component concentrates. For each cell the
exact audit records regret, the refined and uniform bounds, and the binary
dispersion factor 2p(1-p). The output directory must not exist.
"""
import argparse
from datetime import datetime, timezone
from fractions import Fraction as F
import hashlib
from itertools import product
import json
from pathlib import Path
import platform
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from p2_index_robustness import latent_model
from staged_pandora_exact import clear_caches

STUDY = HERE.parent/"reports/p2_index_robustness_2026-10-26"
PRIORS = tuple(map(F, ("1/2", "1/4", "1/10", "1/40", "1/160")))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=HERE/"p2_concentration_sweep"/"run_v1")
    args = ap.parse_args()
    bases = json.loads((STUDY/"models.json").read_text())
    assert len(bases) == 12
    args.out.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), HERE/"p2_index_robustness.py", HERE/"staged_pandora_exact.py"]
    freeze = dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),
                  models_sha256=sha(STUDY/"models.json"), priors=[str(p) for p in PRIORS],
                  sources={p.name: sha(p) for p in sources}, python=sys.version, platform=platform.platform())
    (args.out/"design_freeze.json").write_text(json.dumps(freeze, indent=2)+"\n")
    records = []
    for b in bases:
        a, bb = tuple(map(F, b["a"])), tuple(map(F, b["b"]))
        rs = tuple(map(F, b["reliability"]))
        costs = tuple(tuple(map(F, cs)) for cs in b["costs"])
        for collect, p in product(product((False, True), repeat=2), PRIORS):
            m = latent_model(a, bb, reliability=rs, costs=costs, collect=collect, p=p)
            r = m.verify()
            assert 0 <= r["regret"] <= r["refined_bound"] <= r["call_uniform"] <= r["tv_uniform"]
            records.append(dict(base=b["base"], collect=list(collect), prior=str(p),
                                dispersion=str(2*p*(1-p)), **{k: str(v) for k, v in r.items()}))
            clear_caches()
    (args.out/"records.json").write_text(json.dumps(records, indent=2)+"\n")
    by = {}
    for p in PRIORS:
        rs = [r for r in records if F(r["prior"]) == p]
        by[str(p)] = dict(cells=len(rs), positive_regret=sum(F(r["regret"]) > 0 for r in rs),
                          max_regret=str(max(F(r["regret"]) for r in rs)),
                          max_call_uniform=str(max(F(r["call_uniform"]) for r in rs)),
                          max_tv_uniform=str(max(F(r["tv_uniform"]) for r in rs)),
                          max_tv_uniform_over_dispersion=str(max(F(r["tv_uniform"])/F(r["dispersion"]) for r in rs)))
    summary = dict(cells=len(records), by_prior=by, completed_at_utc=datetime.now(timezone.utc).isoformat())
    (args.out/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    snap = args.out/"source_snapshot"
    snap.mkdir()
    for p in sources:
        (snap/p.name).write_bytes(p.read_bytes())
    files = sorted(p for p in args.out.rglob("*") if p.is_file())
    (args.out/"artifact_manifest.json").write_text(json.dumps({str(p.relative_to(args.out)): sha(p) for p in files}, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
