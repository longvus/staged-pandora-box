"""Exact finite-joint-law checks for the recomputed staged-index robustness result.

All probabilities and values are rational.  The true Bellman solver uses joint
worlds; the index and its potential use products of current posterior marginals.
No surrogate fitting or performance comparator is involved.
"""
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction as F
from functools import lru_cache
from itertools import product

from staged_pandora_exact import Node, Oracle, nested


def tv(a, b):
    return sum(abs(a.get(x, F(0)) - b.get(x, F(0))) for x in a.keys() | b.keys()) / 2


@dataclass(frozen=True)
class Spec:
    costs: tuple
    collect: tuple
    overrides: tuple = ()  # (own observed prefix, cost of the next stage)

    def cost(self, prefix):
        return dict(self.overrides).get(prefix, self.costs[len(prefix)])


class Joint:
    def __init__(self, worlds, specs, q0=F(0), latent=None):
        self.worlds = tuple((paths, w) for paths, w in worlds if w)
        self.specs, self.q0, self.latent = tuple(specs), q0, latent
        self.n = len(specs)
        self.root = ((),) * self.n
        self.horizon = sum(len(s.costs) for s in specs)
        self.cmax = max(c for s in specs for c in s.costs+tuple(c for _,c in s.overrides))
        assert sum(w for _, w in self.worlds) == 1
        assert all(w > 0 for _, w in self.worlds)
        assert all(len(s.costs) == len(s.collect) and all(c > 0 for c in s.costs) for s in specs)
        assert all(c > 0 for s in specs for _, c in s.overrides)
        assert all(len(paths) == self.n and all(len(path) == len(s.costs) and
                   all(0 <= y <= 1 for y in path) for path, s in zip(paths, specs))
                   for paths, _ in self.worlds)
        self.oracle = Oracle()
        # Per-instance caches avoid retaining previous studies through global keys.
        for name in ('matching', 'nodes', 'potential', 'actions', 'value', 'local_action',
                     'local_value', 'metrics', 'trace'):
            setattr(self, name, lru_cache(None)(getattr(self, name)))

    def matching(self, h):
        hits = [(paths, w) for paths, w in self.worlds
                if all(path[:len(prefix)] == prefix for path, prefix in zip(paths, h))]
        mass = sum(w for _, w in hits)
        assert mass > 0
        return tuple((paths, w/mass) for paths, w in hits)

    def q(self, h):
        return max([self.q0] + [y for i, prefix in enumerate(h) for k, y in enumerate(prefix)
                               if self.specs[i].collect[k]])

    def available(self, h):
        return tuple(i for i in range(self.n) if len(h[i]) < len(self.specs[i].costs))

    def cost(self, h, i):
        return self.specs[i].cost(h[i])

    def children(self, h, i):
        d = defaultdict(F)
        for paths, w in self.matching(h):
            d[paths[i][len(h[i])]] += w
        return tuple((w, h[:i]+(h[i]+(y,),)+h[i+1:], y) for y, w in sorted(d.items()))

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
            return Node(self.specs[i].cost(prefix), self.specs[i].collect[len(prefix)],
                        tuple((w/mass, y, arm(i, prefix+(y,))) for y, w in sorted(d.items())))
        return tuple(arm(i, prefix) for i, prefix in enumerate(h))

    def potential(self, h):
        return self.oracle.covered_value(self.nodes(h), self.q(h))

    def actions(self, h):
        return {i: -self.cost(h, i)+sum(p*self.value(nxt) for p, nxt, _ in self.children(h, i))
                for i in self.available(h)} | {-1: self.q(h)}

    def value(self, h):
        return max(self.actions(h).values())

    def local_action(self, h):
        scores = [(nested(node)[0], -i, i) for i, node in enumerate(self.nodes(h))
                  if node is not None and nested(node)[0] is not None]
        return max(scores)[2] if scores and max(scores)[0] > self.q(h) else -1

    def local_value(self, h):
        i = self.local_action(h)
        return self.q(h) if i < 0 else -self.cost(h, i)+sum(
            p*self.local_value(nxt) for p, nxt, _ in self.children(h, i))

    def marginal(self, h, i):
        d = defaultdict(F)
        for paths, w in self.matching(h):
            d[paths[i][len(h[i]):]] += w
        return dict(d)

    def component_marginal(self, h, i, theta, next_only=False):
        p0, components = self.latent
        hits = [(paths, w) for paths, w in components[theta] if w and
                all(path[:len(prefix)] == prefix for path, prefix in zip(paths, h))]
        mass = sum(w for _, w in hits)
        if not mass:
            return {}, F(0)
        d = defaultdict(F)
        for paths, w in hits:
            path = paths[i][len(h[i]):]
            d[path[:1] if next_only else path] += w/mass
        return dict(d), mass

    def latent_envelope(self, h, i):
        p0, _ = self.latent
        _, m0 = self.component_marginal(h, i, 0)
        _, m1 = self.component_marginal(h, i, 1)
        p = p0*m1/((1-p0)*m0+p0*m1)
        if p in (0, 1):
            return F(0)
        di = tv(self.component_marginal(h, i, 0, True)[0],
                self.component_marginal(h, i, 1, True)[0])
        total = F(0)
        for j in range(self.n):
            if j != i:
                dj = tv(self.component_marginal(h, j, 0)[0],
                        self.component_marginal(h, j, 1)[0])
                total += self.oscillation(h, j)*dj
        return 2*p*(1-p)*di*total

    def oscillation(self, h, j):
        remaining = len(self.specs[j].costs)-len(h[j])
        return 1+remaining*self.cmax if any(self.specs[j].collect[len(h[j]):]) else F(0)

    @staticmethod
    def call(dist, r):
        return sum(p*max(z-r, F(0)) for z, p in dist if z is not None)

    def metrics(self, h, i):
        roots, gamma, call_eps, tv_eps, frozen_expect = self.nodes(h), F(0), F(0), F(0), F(0)
        for p, nxt, y in self.children(h, i):
            child = next(child for _, reading, child in roots[i].branches if reading == y)
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
                bound = self.oscillation(h, j)*tv(self.marginal(h, j), self.marginal(nxt, j))
                assert eta <= bound
                call_eps += p*eta
                tv_eps += p*bound
        d = self.potential(h)+self.cost(h, i)-frozen_expect
        assert d == self.oracle.action_gap(roots, self.q(h), i) >= 0
        assert abs(gamma) <= call_eps <= tv_eps
        assert -self.cost(h, i)+sum(p*self.potential(nxt) for p, nxt, _ in self.children(h, i)) \
            - self.potential(h) == gamma-d
        if self.latent is not None:
            assert self.latent_envelope(h, i) == tv_eps
        return d, gamma, call_eps, tv_eps

    def choose(self, h, policy):
        if policy == 'index':
            return self.local_action(h)
        if policy == 'optimal':
            return max(self.actions(h), key=self.actions(h).get)
        if policy == 'stop':
            return -1
        return next(iter(self.available(h)), -1)  # arbitrary always-advance policy

    def trace(self, h, policy):
        """Utility, sum d, stop gap, Gamma+, Gamma-, refined positive residual."""
        i = self.choose(h, policy)
        if i < 0:
            return self.q(h), F(0), self.potential(h)-self.q(h), F(0), F(0), F(0)
        d, g, _, _ = self.metrics(h, i)
        out = [-self.cost(h, i), d, F(0), max(g,F(0)), max(-g,F(0)), max(g-d,F(0))]
        for p, nxt, _ in self.children(h, i):
            for k, x in enumerate(self.trace(nxt, policy)):
                out[k] += p*x
        assert out[0] == self.potential(h)-out[1]-out[2]+out[3]-out[4]
        if policy == 'index':
            assert out[1] == out[2] == 0
        return tuple(out)

    def verify(self):
        self.value(self.root)  # enumerates every feasible history, including off-policy states
        states = self.all_histories()
        min_gamma, max_call, max_tv, actions = F(0), F(0), F(0), 0
        for h in states:
            for i in self.available(h):
                _, g, ec, et = self.metrics(h, i)
                min_gamma, max_call, max_tv = min(min_gamma, g), max(max_call, ec), max(max_tv, et)
                actions += 1
            for policy in ('index', 'optimal', 'stop', 'advance'):
                self.trace(h, policy)
        ix, opt = self.trace(self.root, 'index'), self.trace(self.root, 'optimal')
        regret = opt[0]-ix[0]
        refined = opt[5]+ix[4]
        signed = opt[3]+ix[4]
        assert opt[0] == self.value(self.root) and ix[0] == self.local_value(self.root)
        assert 0 <= regret <= refined <= signed <= 2*self.horizon*max_call <= 2*self.horizon*max_tv
        return dict(states=len(states), actions=actions, regret=regret, utility=ix[0], optimal=opt[0],
                    refined_bound=refined, signed_bound=signed, call_uniform=2*self.horizon*max_call,
                    tv_uniform=2*self.horizon*max_tv, min_gamma=min_gamma,
                    max_call=max_call, max_tv=max_tv)

    def all_histories(self):
        seen = set()
        def visit(h):
            if h in seen:
                return
            seen.add(h)
            for i in self.available(h):
                for _, nxt, _ in self.children(h, i):
                    visit(nxt)
        visit(self.root)
        return sorted(seen, key=lambda h: (sum(map(len, h)), h))


def latent_model(a, b, reliability=(F(3,4), F(3,4)), costs=((F(19,500),F(1,5)),
                  (F(1,20),F(1,5))), scores=(F(2,5),F(3,5)), collect=(True,True), p=F(4,5)):
    """Two two-stage paths, independent conditional on a binary latent variable."""
    components = []
    for theta in (0, 1):
        worlds = []
        for s0, h0, s1, h1 in product((0,1), repeat=4):
            w = F(1)
            for i, s, h in ((0,s0,h0),(1,s1,h1)):
                ph, r = (a,b)[i][theta], reliability[i]
                w *= (ph if h else 1-ph)*(r if s == h else 1-r)
            paths = ((scores[0]*s0,F(h0)),(scores[1]*s1,F(h1)))
            worlds.append((paths,w))
        components.append(tuple(worlds))
    worlds = tuple((paths,(1-p)*w0+p*w1) for (paths,w0),(_,w1) in zip(*components))
    return Joint(worlds, tuple(Spec(tuple(cs),(co,True)) for cs,co in zip(costs,collect)),
                 latent=(p,tuple(components)))


def shrink(base, p, lam):
    mean = (1-p)*base[0]+p*base[1]
    return tuple(mean+lam*(x-mean) for x in base)
