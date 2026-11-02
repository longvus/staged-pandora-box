"""Exact finite-tree model and independent Bellman verification for staged Pandora.

None denotes an exhausted arm, and denotes minus infinity only inside value
distributions. All numerical probabilities, readings and costs are Fractions.
"""
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction as F
from functools import lru_cache
from itertools import product


@dataclass(frozen=True, eq=False)
class Node:
    cost: F
    collect: bool
    branches: tuple  # (probability, reading, successor)

    def __post_init__(self):
        assert self.cost > 0 and self.branches
        assert sum(p for p, _, _ in self.branches) == 1
        assert all(p > 0 and F(0) <= y <= F(1) for p, y, _ in self.branches)
        assert len({y for _, y, _ in self.branches}) == len(self.branches)


def vmax(x, y):
    return y if x is None else x if y is None else max(x, y)


def pack(d):
    return tuple(sorted(d.items(), key=lambda z: (z[0] is not None, z[0] or F(0))))


def reservation(dist, cost):
    atoms = sorted(((w, p) for w, p in dist if w is not None), reverse=True)
    if not atoms:
        return None
    mass = total = F(0)
    for j, (w, p) in enumerate(atoms):
        mass += p
        total += p * w
        root = (total - cost) / mass
        if j + 1 == len(atoms) or root >= atoms[j + 1][0]:
            assert sum(p * max(w - root, F(0)) for w, p in atoms) == cost
            return root
    raise AssertionError("reservation root missing")


@lru_cache(None)
def nested(node):
    if node is None:
        return None, ((None, F(1)),), ((None, F(1)),)
    wd = defaultdict(F)
    for p, y, child in node.branches:
        for z, pz in nested(child)[2]:
            wd[vmax(y if node.collect else None, z)] += p * pz
    wdist = pack(wd)
    g = reservation(wdist, node.cost)
    zd = defaultdict(F)
    for w, p in wdist:
        zd[None if w is None or g is None else min(g, w)] += p
    return g, wdist, pack(zd)


def max_distribution(q, distributions):
    out = {q: F(1)}
    for dist in distributions:
        updated = defaultdict(F)
        for r, pr in out.items():
            for z, pz in dist:
                updated[vmax(r, z)] += pr * pz
        out = updated
    return pack(out)


@lru_cache(None)
def completion(node):
    """Distribution and expected cost through the next collectable stage."""
    if node is None:
        return (), F(0)
    if node.collect:
        d = defaultdict(F)
        for p, y, _ in node.branches:
            d[y] += p
        return pack(d), node.cost
    d, cost = defaultdict(F), node.cost
    for p, _, child in node.branches:
        dc, cc = completion(child)
        if not dc:
            return (), F(0)  # the finite ladders used here have a fixed collectable set
        cost += p * cc
        for y, py in dc:
            d[y] += p * py
    return pack(d), cost


def successor(states, q, i, y, child):
    return states[:i] + (child,) + states[i + 1:], max(q, y) if states[i].collect else q


class Oracle:
    def __init__(self):
        self.solved = {}

        @lru_cache(None)
        def value(states, q):
            qs = {}
            for i, node in enumerate(states):
                if node is not None:
                    qs[i] = -node.cost + sum(
                        p * value(*successor(states, q, i, y, child))
                        for p, y, child in node.branches
                    )
            v = max([q] + list(qs.values()))
            self.solved[states, q] = v, qs
            return v

        self.value = value

    def covered_value(self, states, q):
        return sum(z * p for z, p in max_distribution(q, [nested(s)[2] for s in states]))

    def action_gap(self, states, q, i):
        g, wd, _ = nested(states[i])
        rd = max_distribution(q, [nested(s)[2] for j, s in enumerate(states) if j != i])
        gain = F(0) if g is None else sum(
            pw * pr * max(w - max(r, g), F(0))
            for (w, pw), (r, pr) in product(wd, rd) if w is not None
        )
        return states[i].cost - gain

    def verify(self, states, q):
        self.value(states, q)
        count = 0
        for (ss, incumbent), (v, qs) in list(self.solved.items()):
            assert self.covered_value(ss, incumbent) == v
            gs = [nested(s)[0] for s in ss]
            finite = [g for g in gs if g is not None]
            if not finite or max(finite) <= incumbent:
                assert v == incumbent
            else:
                assert all(qs[i] == v for i, g in enumerate(gs) if g == max(finite))
            for i, qi in qs.items():
                assert self.action_gap(ss, incumbent, i) == v - qi >= 0
                count += 1
        return len(self.solved), count

    def policy(self, states, q, kind):
        assert kind in ("index", "committed", "recomputed")

        @lru_cache(None)
        def run(ss, incumbent, locked):
            if locked >= 0:
                i = locked
            else:
                scores = []
                for j, s in enumerate(ss):
                    if s is None:
                        continue
                    if kind == "index":
                        g = nested(s)[0]
                    else:
                        dist, cost = completion(s)
                        g = reservation(dist, cost) if dist else None
                    if g is not None:
                        scores.append((g, -j, j))
                if not scores or max(scores)[0] <= incumbent:
                    return incumbent, self.value(ss, incumbent) - incumbent, F(0)
                i = max(scores)[2]
            node = ss[i]
            val, stop_loss, advance_loss = -node.cost, F(0), self.action_gap(ss, incumbent, i)
            for p, y, child in node.branches:
                nxt, qi = successor(ss, incumbent, i, y, child)
                lock = i if kind == "committed" and not node.collect else -1
                v, st, ad = run(nxt, qi, lock)
                val += p * v
                stop_loss += p * st
                advance_loss += p * ad
            assert self.value(ss, incumbent) - val == stop_loss + advance_loss
            return val, stop_loss, advance_loss

        return run(states, q, -1)


def random_arm(rng, depth, collectable, stage=1):
    if stage > depth:
        return None
    ys = rng.sample([F(j, 4) for j in range(5)], 2)
    p = rng.choice([F(1, 4), F(1, 2), F(3, 4)])
    return Node(rng.choice([F(1, 100), F(3, 100), F(3, 50), F(1, 10)]),
                stage in collectable,
                tuple((prob, y, random_arm(rng, depth, collectable, stage + 1))
                      for prob, y in zip((p, 1 - p), ys)))


def clear_caches():
    nested.cache_clear()
    completion.cache_clear()
