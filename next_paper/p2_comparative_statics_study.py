#!/usr/bin/env python3
"""Exact comparative statics and resumption study on the staging grid.

For every cell of the 660-cell staging study (reused from its records),
compute in rational arithmetic:

* the exact comparative-statics identity for every dominated transition in
  the grid (larger collectable set, lower signal cost, lower terminal cost,
  constant-cost versus history-dependent-cost family), coupling the two
  covered-call values of each arm on that arm's own path;
* the utility of planning with the terminal-only indices inside a problem
  with intermediate collection, against the two-sided bound
  V(terminal) <= V_pi <= V(current);
* the exact no-return optimum, the resumption gap V* - V_NR, whether the
  index policy resumes, and the min-cut lower bound rho.

A seeded random finite-tree check repeats the identity on arbitrary binary
trees with enlarged collectable sets and scaled costs. Outputs are rational
records, a summary, source snapshots and a manifest; the directory must not
exist beforehand.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction as F
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import platform
import random
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p2_staging_study as st
from staged_pandora_exact import (Node, Oracle, clear_caches, nested, pack, random_arm,
                                  reservation, successor, vmax)

STAGING = HERE/"p2_staging_study"/"run_v1"
REGIMES = {"terminal": (3,), "mixed": (1, 3), "all": (1, 2, 3)}
NEXT_REGIME = {"terminal": "mixed", "mixed": "all"}
RANDOM_SEED = 20261030
RANDOM_PAIRS = 60


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(None)
def nested_pair(a, b):
    """Joint law of (Z_a, Z_b) for two arms with identical readings and branching."""
    if a is None:
        assert b is None
        return None, None, {(None, None): F(1)}
    wd = defaultdict(F)
    for (p, y, ca), (pb, yb, cb) in zip(a.branches, b.branches):
        assert p == pb and y == yb
        for (za, zb), pz in nested_pair(ca, cb)[2].items():
            wd[(vmax(y if a.collect else None, za), vmax(y if b.collect else None, zb))] += p*pz
    ma, mb = defaultdict(F), defaultdict(F)
    for (wa, wb), p in wd.items():
        ma[wa] += p
        mb[wb] += p
    assert pack(ma) == nested(a)[1] and pack(mb) == nested(b)[1]
    ga, gb = reservation(pack(ma), a.cost), reservation(pack(mb), b.cost)
    zd = defaultdict(F)
    for (wa, wb), p in wd.items():
        zd[(None if wa is None or ga is None else min(ga, wa),
            None if wb is None or gb is None else min(gb, wb))] += p
    return ga, gb, dict(zd)


def pathwise_ordered(a, b):
    """Z_b >= Z_a on every path, G_b >= G_a, recursively through the tree."""
    if a is None:
        return True
    ga, gb, zd = nested_pair(a, b)
    if ga is not None and (gb is None or gb < ga):
        return False
    if any(zb is None and za is not None or (za is not None and zb < za) for za, zb in zd):
        return False
    return all(pathwise_ordered(ca, cb) for (_, _, ca), (_, _, cb) in zip(a.branches, b.branches))


def overshoot(arms_a, arms_b, q=F(0)):
    """E[(max_j Z_b,j - max{q, max_j Z_a,j})^+], arms independent."""
    out = {(q, q): F(1)}
    for a, b in zip(arms_a, arms_b):
        nxt = defaultdict(F)
        for (ra, rb), pr in out.items():
            for (za, zb), pz in nested_pair(a, b)[2].items():
                nxt[(vmax(ra, za), vmax(rb, zb))] += pr*pz
        out = nxt
    return sum(p*max(rb-ra, F(0)) for (ra, rb), p in out.items())


def foreign_index_value(plan_arms, true_arms):
    """Plan with plan_arms' indices; incumbents and payoffs follow true_arms."""
    @lru_cache(None)
    def run(ps, ts, q):
        i = st.index_action(ps, q)
        if i is None:
            return q
        node = ts[i]
        val = -node.cost
        for (p, y, pc), (_, _, tc) in zip(ps[i].branches, node.branches):
            ts2, q2 = successor(ts, q, i, y, tc)
            val += p*run(ps[:i]+(pc,)+ps[i+1:], ts2, q2)
        return val
    return run(plan_arms, true_arms, F(0))


def noreturn_value(arms):
    """Optimal value when an arm left before exhaustion can never be resumed."""
    @lru_cache(None)
    def run(ss, q, cur):
        best = q
        for j, node in enumerate(ss):
            if node is None:
                continue
            ss_j = ss if cur in (-1, j) else ss[:cur]+(None,)+ss[cur+1:]
            v = -node.cost
            for p, y, child in node.branches:
                nxt, q2 = successor(ss_j, q, j, y, child)
                v += p*run(nxt, q2, j if child is not None else -1)
            best = max(best, v)
        return best
    return run(arms, F(0), -1)


def resumption_bound(arms, oracle):
    """(index policy resumes with positive probability, rho) from the root."""
    def delta_min(ss, q, idx):
        gaps = [oracle.value(ss, q)-q]
        gaps += [oracle.action_gap(ss, q, j) for j, s in enumerate(ss) if s is not None and j != idx]
        return min(gaps)

    @lru_cache(None)
    def run(ss, q, cur, left):
        i = st.index_action(ss, q)
        if i is None:
            return False, F(0)
        if i in left:
            return True, delta_min(ss, q, i)
        left2 = frozenset(left | ({cur} if cur not in (-1, i) and ss[cur] is not None else set()))
        below, cost = False, F(0)
        for p, y, child in ss[i].branches:
            b, c = run(*successor(ss, q, i, y, child), i if child is not None else -1, left2)
            below |= b
            cost += p*c
        return (True, min(delta_min(ss, q, i), cost)) if below else (False, F(0))
    return run(arms, F(0), -1, frozenset())


def relax(node, extra, factor, stage=1):
    """Same readings; collectable set enlarged by `extra`, costs scaled by `factor`."""
    if node is None:
        return None
    return Node(node.cost*factor, node.collect or stage in extra,
                tuple((p, y, relax(child, extra, factor, stage+1)) for p, y, child in node.branches))


def random_check(rng):
    records = []
    for pair in range(RANDOM_PAIRS):
        depth = rng.choice((2, 3, 4))
        base = tuple(random_arm(rng, depth, rng.choice(((depth,), (1, depth), tuple(range(1, depth+1))))) for _ in range(2))
        extra = tuple(sorted(rng.sample(range(1, depth+1), rng.randint(0, depth))))
        factor = rng.choice((F(1), F(1, 2), F(3, 4)))
        larger = tuple(relax(a, extra, factor) for a in base)
        oa, ob = Oracle(), Oracle()
        va, vb = oa.value(base, F(0)), ob.value(larger, F(0))
        ordered = all(pathwise_ordered(a, b) for a, b in zip(base, larger))
        gain = overshoot(base, larger)
        assert ordered and vb-va == gain >= 0
        records.append(dict(pair=pair, depth=depth, extra=extra, cost_factor=str(factor),
                            value_base=str(va), value_larger=str(vb), formula=str(gain)))
        clear_caches()
        nested_pair.cache_clear()
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=HERE/"p2_comparative_statics"/"run_v1")
    args = ap.parse_args()
    manifest = json.loads((STAGING/"artifact_manifest.json").read_text())
    assert all(sha(STAGING/name) == digest for name, digest in manifest.items())
    rows = [json.loads(l) for l in (STAGING/"records.jsonl").read_text().splitlines()]
    assert len(rows) == 660
    args.out.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), HERE/"staged_pandora_exact.py", HERE/"p2_staging_study.py"]
    freeze = dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),
                  staging_records_sha256=sha(STAGING/"records.jsonl"),
                  sources={str(p.relative_to(HERE)): sha(p) for p in sources},
                  random_seed=RANDOM_SEED, random_pairs=RANDOM_PAIRS,
                  python=sys.version, executable=sys.executable, platform=platform.platform())
    (args.out/"design_freeze.json").write_text(json.dumps(freeze, indent=2)+"\n")
    started = time.monotonic()

    def key_of(r):
        return (r["family"], r["accuracy"], r["signal_cost"], r["terminal_cost"], r["regime"])
    cells = {key_of(r): r for r in rows}
    arms = {}
    for key in cells:
        fam, acc, k, c, reg = key
        arms[key] = st.make_case(fam, None if acc is None else F(acc), F(k), F(c), REGIMES[reg])[0]
    ks = [str(x) for x in st.SIGNAL_COSTS]
    cs = [str(x) for x in st.TERMINAL_COSTS]
    out_rows = []
    transitions = Counter()
    foreign = Counter()
    resumption = Counter()
    with (args.out/"records.jsonl").open("x") as stream:
        for number, (key, r) in enumerate(cells.items(), 1):
            fam, acc, k, c, reg = key
            V = F(r["vstar"])
            oracle = Oracle()
            assert oracle.value(arms[key], F(0)) == V
            targets = []
            if reg in NEXT_REGIME:
                targets.append(("collectable_set", (fam, acc, k, c, NEXT_REGIME[reg])))
            if ks.index(k) > 0:
                targets.append(("signal_cost", (fam, acc, ks[ks.index(k)-1], c, reg)))
            if cs.index(c) > 0:
                targets.append(("terminal_cost", (fam, acc, k, cs[cs.index(c)-1], reg)))
            if fam == "history_cost":
                targets.append(("cost_family", ("constant_cost", acc, k, c, reg)))
            trans = []
            for kind, key2 in targets:
                V2 = F(cells[key2]["vstar"])
                ordered = all(pathwise_ordered(a, b) for a, b in zip(arms[key], arms[key2]))
                gain = overshoot(arms[key], arms[key2])
                assert ordered and V2-V == gain >= 0
                transitions[kind] += 1
                trans.append(dict(kind=kind, target_cell=cells[key2]["cell"], value_gain=str(V2-V), formula=str(gain)))
            row = dict(cell=r["cell"], family=fam, regime=reg, accuracy=acc, signal_cost=k, terminal_cost=c,
                       vstar=str(V), transitions=trans)
            if reg != "terminal":
                key0 = (fam, acc, k, c, "terminal")
                V0 = F(cells[key0]["vstar"])
                vf = foreign_index_value(arms[key0], arms[key])
                loss, rights = V-vf, V-V0
                assert 0 <= loss <= rights
                foreign[(reg, "cells")] += 1
                foreign[(reg, "zero_loss")] += loss == 0
                foreign[(reg, "full_loss")] += loss == rights and rights > 0
                row["terminal_index_planning"] = dict(utility=str(vf), loss=str(loss), rights_value=str(rights))
            vr = noreturn_value(arms[key])
            resumes, rho = resumption_bound(arms[key], oracle)
            gap = V-vr
            assert 0 <= rho <= gap
            assert (gap > 0) == resumes == (rho > 0)
            resumption[(fam, "cells")] += 1
            resumption[(fam, "positive")] += gap > 0
            resumption[(fam, "bound_tight")] += gap > 0 and rho == gap
            row["resumption"] = dict(noreturn_value=str(vr), gap=str(gap), index_resumes=resumes, rho=str(rho))
            stream.write(json.dumps(row)+"\n")
            out_rows.append(row)
            clear_caches()
            nested_pair.cache_clear()
            if number % 110 == 0:
                print(f"{number}/660 {time.monotonic()-started:.1f}s", flush=True)
    random_records = random_check(random.Random(RANDOM_SEED))
    (args.out/"random_tree_check.json").write_text(json.dumps(random_records, indent=2)+"\n")
    gaps = defaultdict(list)
    for row in out_rows:
        gaps[row["family"]].append(F(row["resumption"]["gap"]))
    summary = dict(cells=660, transitions={k: v for k, v in transitions.items()},
                   transitions_total=sum(transitions.values()), formula_failures=0,
                   terminal_index_planning={f"{reg}/{what}": n for (reg, what), n in foreign.items()},
                   resumption={f"{fam}/{what}": n for (fam, what), n in resumption.items()},
                   resumption_max_gap={fam: str(max(v)) for fam, v in gaps.items()},
                   resumption_mean_gap={fam: str(sum(v)/len(v)) for fam, v in gaps.items()},
                   random_pairs=len(random_records), elapsed_seconds=time.monotonic()-started,
                   completed_at_utc=datetime.now(timezone.utc).isoformat())
    (args.out/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    snap = args.out/"source_snapshot"
    snap.mkdir()
    for p in sources:
        (snap/p.name).write_bytes(p.read_bytes())
    files = sorted(p for p in args.out.rglob("*") if p.is_file())
    (args.out/"artifact_manifest.json").write_text(json.dumps(
        {str(p.relative_to(args.out)): sha(p) for p in files}, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
