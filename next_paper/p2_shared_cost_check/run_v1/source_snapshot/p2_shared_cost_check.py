#!/usr/bin/env python3
"""Exact check of the sharing bound when an arm's cost may depend on every arm's history.

Costs follow the standing assumption: the cost of advancing arm i is known before the
advance and may depend on the whole observed history. Each arm's local problem at a
history S uses c_{j,S}(s), the cost of its next stage once only j has advanced from S to
own state s. All quantities are exact rationals.

Three families, all with two arms, q0 = 0 and unobserved starting arms:
  latent     the 12 binary-latent base models of the shared-law study at component
             contrast lambda in {1/5, 1}, four collection choices, and a terminal cost
             k_i(theta) = c_i (1 + sign_i * s * (theta - p)) with sharing s in {0, 1/2, 1};
             the cost paid is its posterior mean, so it moves with the other arm's readings;
  arbitrary  12 arbitrary joint laws on two binary three-stage paths;
  product    12 independent pairs of binary three-stage paths.
In the last two, the next cost of arm j is its base cost times 1 + (1/2) x, with x the
number of ones observed on the other arm. In the product family the paths are
independent, so every sharing effect comes from costs.

Checked at every feasible history and available advance: the signed residual identity,
the product-problem action gap, equality of the advanced arm's frozen and updated local
problems, |Gamma| <= eps (local values) <= the TV-plus-cost envelope, and, for the latent
family, the common-component bound with its cost-sensitivity term. Checked per cell:
policy telescoping for four policies and regret <= refined <= signed <= 2H eps_bar.
The output directory must not exist.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from fractions import Fraction as F
from functools import lru_cache
import hashlib
from itertools import product
import json
from pathlib import Path
import platform
import random
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from p2_index_robustness import Joint, Spec, latent_model, shrink, tv
from staged_pandora_exact import Node, clear_caches, nested

STUDY = HERE.parent/"reports/p2_index_robustness_2026-10-26"
LAMBDAS = (F(1, 5), F(1))
SHARING = (F(0), F(1, 2), F(1))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def same(a, b):
    """Structural equality of local arm trees (Node compares by identity)."""
    if a is None or b is None:
        return a is b
    return (a.cost == b.cost and a.collect == b.collect and len(a.branches) == len(b.branches)
            and all(pa == pb and ya == yb and same(ca, cb)
                    for (pa, ya, ca), (pb, yb, cb) in zip(a.branches, b.branches)))


class SharedCostJoint(Joint):
    """Joint law whose advance costs are functions of the whole history."""

    def __init__(self, worlds, specs, cost_fn, cmax, latent=None, cost_gap=None):
        self.cost_fn, self.cost_gap = cost_fn, cost_gap
        super().__init__(worlds, specs, latent=latent)
        self.cmax = cmax
        self.corollary = lru_cache(None)(self.corollary)

    @staticmethod
    def with_arm(h, j, prefix):
        return h[:j]+(prefix,)+h[j+1:]

    def cost(self, h, i):
        c = self.cost_fn(h, i)
        assert 0 < c <= self.cmax
        return c

    def nodes(self, h):
        worlds = self.matching(h)

        def arm(i, prefix):
            if len(prefix) == len(self.specs[i].costs):
                return None
            d = defaultdict(F)
            for paths, w in worlds:
                if paths[i][:len(prefix)] == prefix:
                    d[paths[i][len(prefix)]] += w
            mass = sum(d.values())
            return Node(self.cost(self.with_arm(h, i, prefix), i), self.specs[i].collect[len(prefix)],
                        tuple((w/mass, y, arm(i, prefix+(y,))) for y, w in sorted(d.items())))
        return tuple(arm(i, prefix) for i, prefix in enumerate(h))

    def own_prefixes(self, h, j):
        """Own states of arm j, reachable from h, at which an advance remains."""
        out = set()
        for path in self.marginal(h, j):
            for k in range(len(path)):
                out.add(h[j]+path[:k])
        return out

    def cost_change(self, h, nxt, j):
        return max((abs(self.cost(self.with_arm(nxt, j, s), j)-self.cost(self.with_arm(h, j, s), j))
                    for s in self.own_prefixes(nxt, j)), default=F(0))

    def corollary(self, h, i):
        """eta_i * sum_j (B_j Lambda_j + 1_j nu_j Lambda^c_j) for a binary component."""
        p0, _ = self.latent
        _, m0 = self.component_marginal(h, i, 0)
        _, m1 = self.component_marginal(h, i, 1)
        p = p0*m1/((1-p0)*m0+p0*m1)
        if p in (0, 1):
            return F(0)
        di = tv(self.component_marginal(h, i, 0, True)[0], self.component_marginal(h, i, 1, True)[0])
        total = F(0)
        for j in range(self.n):
            if j == i:
                continue
            dj = tv(self.component_marginal(h, j, 0)[0], self.component_marginal(h, j, 1)[0])
            remaining = len(self.specs[j].costs)-len(h[j])
            costs = self.cost_gap(j, len(h[j])) if any(self.specs[j].collect[len(h[j]):]) else F(0)
            total += self.oscillation(h, j)*dj+remaining*costs
        return 2*p*(1-p)*di*total

    def metrics(self, h, i):
        roots, gamma, call_eps, tv_eps, law_eps, frozen_expect = self.nodes(h), *[F(0)]*5
        cost_terms = F(0)
        for p, nxt, y in self.children(h, i):
            child = next(child for _, reading, child in roots[i].branches if reading == y)
            assert same(child, self.nodes(nxt)[i])  # frozen advanced arm = its updated local problem
            frozen = roots[:i]+(child,)+roots[i+1:]
            frozen_v = self.oracle.covered_value(frozen, self.q(nxt))
            frozen_expect += p*frozen_v
            gamma += p*(self.potential(nxt)-frozen_v)
            for j in range(self.n):
                if j == i:
                    continue
                old, new = nested(roots[j])[2], nested(self.nodes(nxt)[j])[2]
                knots = {self.q(nxt), F(1)} | {z for z, _ in old+new
                                                    if z is not None and self.q(nxt) <= z <= 1}
                eta = max(abs(self.call(old, r)-self.call(new, r)) for r in knots)
                law = self.oscillation(h, j)*tv(self.marginal(h, j), self.marginal(nxt, j))
                remaining = len(self.specs[j].costs)-len(h[j])
                extra = remaining*self.cost_change(h, nxt, j) if any(self.specs[j].collect[len(h[j]):]) else F(0)
                assert eta <= law+extra
                call_eps += p*eta
                tv_eps += p*(law+extra)
                law_eps += p*law
                cost_terms += p*extra
        d = self.potential(h)+self.cost(h, i)-frozen_expect
        assert d == self.oracle.action_gap(roots, self.q(h), i) >= 0
        assert abs(gamma) <= call_eps <= tv_eps
        assert -self.cost(h, i)+sum(p*self.potential(nxt) for p, nxt, _ in self.children(h, i)) \
            - self.potential(h) == gamma-d
        if self.latent is not None:
            assert self.latent_envelope(h, i) == law_eps  # binary law term, as before
            assert call_eps <= self.corollary(h, i)
        self.cost_seen = max(getattr(self, "cost_seen", F(0)), cost_terms)
        return d, gamma, call_eps, tv_eps

    def verify(self):
        r = super().verify()
        states = self.all_histories()
        max_cor = max((self.corollary(h, i) for h in states for i in self.available(h)), default=F(0)) \
            if self.latent is not None else None
        if max_cor is not None:
            assert r["regret"] <= 2*self.horizon*max_cor
        r["max_cost_term"] = getattr(self, "cost_seen", F(0))
        r["corollary_uniform"] = None if max_cor is None else 2*self.horizon*max_cor
        return r


def latent_shared(b, lam, collect, s):
    a, bb = tuple(map(F, b["a"])), tuple(map(F, b["b"]))
    p = F(b["p"])
    rs = tuple(map(F, b["reliability"]))
    costs = tuple(tuple(map(F, cs)) for cs in b["costs"])
    base = latent_model(shrink(a, p, lam), shrink(bb, p, lam), reliability=rs, costs=costs,
                        collect=collect, p=p)
    _, components = base.latent
    sign = (1, -1)

    def k(i, theta, stage):
        c = costs[i][stage]
        return c*(1+sign[i]*s*(theta-p)) if stage == 1 else c

    @lru_cache(None)
    def posterior(h):
        mass = []
        for theta in (0, 1):
            mass.append(sum(w for paths, w in components[theta]
                            if all(path[:len(prefix)] == prefix for path, prefix in zip(paths, h))))
        return p*mass[1]/((1-p)*mass[0]+p*mass[1])

    def cost_fn(h, i):
        stage, post = len(h[i]), posterior(h)
        return (1-post)*k(i, 0, stage)+post*k(i, 1, stage)

    def cost_gap(j, stage):
        return max(abs(k(j, 1, t)-k(j, 0, t)) for t in range(stage, 2))

    cmax = max(k(i, theta, t) for i in range(2) for theta in (0, 1) for t in range(2))
    return SharedCostJoint(base.worlds, base.specs, cost_fn, cmax, latent=base.latent, cost_gap=cost_gap)


def cross_cost(specs):
    def cost_fn(h, i):
        other = sum(h[1-i])
        return specs[i].cost(h[i])*(1+F(1, 2)*other)
    cmax = max(c for s in specs for c in s.costs+tuple(c for _, c in s.overrides))*F(5, 2)
    return cost_fn, cmax


def three_stage_specs(base):
    return tuple(Spec((F(1, 20), F(1, 10), F(1, 5)),
                      (bool((base+i) % 2), bool((base+i) % 3), True),
                      (((F(1),), F(1, 4)), ((F(0), F(1)), F(1, 25)))) for i in range(2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=HERE/"p2_shared_cost_check"/"run_v1")
    args = ap.parse_args()
    bases = json.loads((STUDY/"models.json").read_text())
    assert len(bases) == 12
    args.out.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), HERE/"p2_index_robustness.py", HERE/"staged_pandora_exact.py"]
    freeze = dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),
                  models_sha256=sha(STUDY/"models.json"), lambdas=[str(x) for x in LAMBDAS],
                  sharing=[str(x) for x in SHARING], seed=20261110,
                  sources={p.name: sha(p) for p in sources}, python=sys.version, platform=platform.platform())
    (args.out/"design_freeze.json").write_text(json.dumps(freeze, indent=2)+"\n")
    records = []

    def run(m, meta):
        r = m.verify()
        assert 0 <= r["regret"] <= r["refined_bound"] <= r["call_uniform"] <= r["tv_uniform"]
        records.append(meta | {k: (None if v is None else str(v)) for k, v in r.items()})
        clear_caches()

    for b in bases:
        for lam, collect, s in product(LAMBDAS, product((False, True), repeat=2), SHARING):
            run(latent_shared(b, lam, collect, s),
                dict(kind="latent", base=b["base"], lam=str(lam), collect=list(collect), sharing=str(s)))
    rng = random.Random(20261110)
    for base in range(12):
        weights = [rng.randint(1, 9) for _ in range(64)]
        worlds = [((bits[:3], bits[3:]), F(w, sum(weights)))
                  for bits, w in zip(product((F(0), F(1)), repeat=6), weights)]
        specs = three_stage_specs(base)
        run(SharedCostJoint(worlds, specs, *cross_cost(specs)), dict(kind="arbitrary", base=base))
    for base in range(12):
        arms = []
        for _ in range(2):
            w = [rng.randint(1, 9) for _ in range(8)]
            arms.append({bits: F(x, sum(w)) for bits, x in zip(product((F(0), F(1)), repeat=3), w)})
        worlds = [((x0, x1), arms[0][x0]*arms[1][x1]) for x0 in arms[0] for x1 in arms[1]]
        specs = three_stage_specs(base)
        run(SharedCostJoint(worlds, specs, *cross_cost(specs)), dict(kind="product", base=base))

    (args.out/"records.json").write_text(json.dumps(records, indent=2)+"\n")
    by = {}
    for kind in ("latent", "arbitrary", "product"):
        rs = [r for r in records if r["kind"] == kind]
        by[kind] = dict(cells=len(rs), histories=sum(int(r["states"]) for r in rs),
                        actions=sum(int(r["actions"]) for r in rs),
                        positive_regret=sum(F(r["regret"]) > 0 for r in rs),
                        max_regret=str(max(F(r["regret"]) for r in rs)),
                        positive_cost_term=sum(F(r["max_cost_term"]) > 0 for r in rs),
                        max_call_uniform=str(max(F(r["call_uniform"]) for r in rs)))
    lat = [r for r in records if r["kind"] == "latent"]
    by["latent"]["no_sharing_positive_cost_term"] = sum(F(r["max_cost_term"]) > 0 for r in lat if r["sharing"] == "0")
    summary = dict(cells=len(records), histories=sum(v["histories"] for v in by.values()),
                   actions=sum(v["actions"] for v in by.values()), by_kind=by,
                   completed_at_utc=datetime.now(timezone.utc).isoformat())
    (args.out/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    snap = args.out/"source_snapshot"
    snap.mkdir()
    for p in sources:
        (snap/p.name).write_bytes(p.read_bytes())
    files = sorted(p for p in args.out.rglob("*") if p.is_file())
    (args.out/"artifact_manifest.json").write_text(
        json.dumps({str(p.relative_to(args.out)): sha(p) for p in files}, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
