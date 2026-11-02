#!/usr/bin/env python3
"""Uniform-in-cost certificate for the collection-induced switching witness.

Runs the exact parametric Bellman recursion and its affine certificate over
the field Q(c) of rational functions in the common terminal cost c. Every
order or equality decision taken by the computation is evaluated at one
generic sample cost and recorded. Each recorded rational function is then
proved to have no zero or pole in the open cost interval (exact real-root
counting on numerator and denominator). Then every decision has the same
outcome for every cost in the interval, so the symbolic output is the exact
optimum, and the symbolic Bellman certificate holds, for all such costs.
Endpoint costs are certified separately with the fixed-cost verifier.
"""
import argparse
from fractions import Fraction as F
import json
from pathlib import Path

import sympy
from sympy import QQ
from sympy.polys.fields import field

import p2_switching_scout as scout
from p2_switching_witness import certificate, model

K, C = field("c", QQ)
X = sympy.Symbol("c")


class G:
    """Element of Q(c); comparisons decided at a sample cost and logged."""
    sample = None
    log = None

    def __init__(self, value):
        if isinstance(value, G):
            value = value.v
        elif isinstance(value, (int, F)):
            value = K(QQ(F(value).numerator, F(value).denominator))
        self.v = value

    def const(self):
        if self.v.numer.degree() <= 0 and self.v.denom.degree() <= 0:
            q = self.v.as_expr()
            return F(int(q.p), int(q.q))
        return None

    def __hash__(self):
        q = self.const()
        return hash(q) if q is not None else hash(str(self.v))

    def _o(self, other):
        return other.v if isinstance(other, G) else G(other).v

    def __add__(self, o): return G(self.v+self._o(o))
    __radd__ = __add__
    def __sub__(self, o): return G(self.v-self._o(o))
    def __rsub__(self, o): return G(self._o(o)-self.v)
    def __mul__(self, o): return G(self.v*self._o(o))
    __rmul__ = __mul__
    def __truediv__(self, o): return G(self.v/self._o(o))
    def __rtruediv__(self, o): return G(self._o(o)/self.v)
    def __neg__(self): return G(-self.v)

    def sign(self, other, kind):
        d = G(self.v-self._o(other))
        q = d.const()
        if q is not None:
            return (q > 0)-(q < 0)
        s = d.v.numer(G.sample)*d.v.denom(G.sample)
        s = (s > 0)-(s < 0)
        assert s != 0, "sample cost is not generic"
        G.log.append((d.v, s, kind))
        return s

    def __eq__(self, o):
        if not isinstance(o, (G, int, F)):
            return NotImplemented
        d = G(self.v-self._o(o))
        if d.const() == 0:
            return True
        self.sign(o, "neq")
        return False

    def __ne__(self, o): return not self.__eq__(o)
    def __lt__(self, o): return self.sign(o, "lt") < 0
    def __le__(self, o): return self.sign(o, "le") <= 0
    def __gt__(self, o): return self.sign(o, "gt") > 0
    def __ge__(self, o): return self.sign(o, "ge") >= 0

    def sym(self):
        return sympy.factor(self.v.as_expr().subs(sympy.Symbol("c"), X))


class A:
    """Exact element a+b*sqrt(M) of Q(sqrt(M)), M fixed and not a square."""
    M = None

    def __init__(self, a, b=0):
        self.a, self.b = F(a), F(b)

    def _o(self, o):
        return o if isinstance(o, A) else A(o)

    def __add__(self, o): o = self._o(o); return A(self.a+o.a, self.b+o.b)
    __radd__ = __add__
    def __sub__(self, o): o = self._o(o); return A(self.a-o.a, self.b-o.b)
    def __rsub__(self, o): return self._o(o)-self
    def __mul__(self, o):
        o = self._o(o)
        return A(self.a*o.a+self.b*o.b*A.M, self.a*o.b+self.b*o.a)
    __rmul__ = __mul__
    def __truediv__(self, o):
        o = self._o(o)
        n = o.a*o.a-o.b*o.b*A.M
        return self*A(o.a/n, -o.b/n)
    def __rtruediv__(self, o): return self._o(o)/self
    def __neg__(self): return A(-self.a, -self.b)

    def sgn(self):
        sa, sb = (self.a > 0)-(self.a < 0), (self.b > 0)-(self.b < 0)
        if sa == 0 or sb == 0 or sa == sb:
            return sa or sb
        return sa if self.a*self.a > self.b*self.b*A.M else -sa

    def __eq__(self, o):
        if not isinstance(o, (A, int, F)):
            return NotImplemented
        o = self._o(o)
        return self.a == o.a and self.b == o.b
    def __hash__(self): return hash(self.a) if self.b == 0 else hash((self.a, self.b))
    def __lt__(self, o): return (self-o).sgn() < 0
    def __le__(self, o): return (self-o).sgn() <= 0
    def __gt__(self, o): return (self-o).sgn() > 0
    def __ge__(self, o): return (self-o).sgn() >= 0


def exact(x):
    """Sympy number in Q or Q(sqrt(A.M)) -> F or A."""
    x = sympy.nsimplify(x)
    if x.is_Rational:
        return F(int(x.p), int(x.q))
    a, b = x.as_independent(sympy.sqrt(A.M))[0], x.coeff(sympy.sqrt(A.M))
    assert sympy.simplify(a+b*sympy.sqrt(A.M)-x) == 0
    return A(F(int(a.p), int(a.q)), F(int(b.p), int(b.q)))


def roots_inside(poly, lo, hi, endpoint_minpolys):
    """Exact count of distinct real roots on the open interval (lo, hi)."""
    p = sympy.Poly(poly, X)
    for m in endpoint_minpolys:
        while p.degree() > 0 and sympy.rem(p, m).is_zero:
            p = sympy.quo(p, m)
    if p.degree() <= 0:
        return 0
    seq = sympy.sturm(p)

    def val(q, x):
        r = F(0) if isinstance(x, F) else A(0)
        for co in q.all_coeffs():
            r = r*x+F(int(co.p), int(co.q))
        return r

    def variations(x):
        signs = []
        for q in seq:
            v = val(q, x)
            s = ((v > 0)-(v < 0)) if isinstance(v, F) else v.sgn()
            if s:
                signs.append(s)
        return sum(1 for u, w in zip(signs, signs[1:]) if u != w)
    for e in (lo, hi):
        assert val(p, e) != 0, "endpoint root left after removing its minimal polynomial"
    return variations(lo)-variations(hi)


def verify(lo, hi, minpolys):
    """Each logged decision: no zero or pole of d on the open interval."""
    seen = {}
    for d, s, kind in G.log:
        key = str(d)
        if key in seen:
            assert seen[key][0] == s
            continue
        num, den = sympy.fraction(sympy.together(d.as_expr()))
        bad = roots_inside(num, lo, hi, minpolys)+roots_inside(den, lo, hi, minpolys)
        seen[key] = (s, bad, kind)
    failures = [k for k, (s, bad, kind) in seen.items() if bad]
    return len(G.log), len(seen), failures


def endpoint_minpoly(x):
    return sympy.Poly(sympy.minimal_polynomial(sympy.Rational(x.numerator, x.denominator) if isinstance(x, F)
                                               else x.a+x.b*sympy.sqrt(A.M), X), X)


def strict_check(pa, regions):
    """The region action strictly beats every other root action on each open region.

    All root action values are affine between consecutive points of the union
    of their breakpoints and the region boundaries. On each such piece, the
    difference (winner - other) is >= 0 at both ends and > 0 at the midpoint,
    hence > 0 on the open piece; it is > 0 at every interior breakpoint.
    """
    acts = pa.actions[((), ())]
    bounds = {r[0] for r in regions} | {regions[-1][1]}
    pts = sorted(bounds | {x for fn in acts.values() for seg in fn for x in seg[:2]})
    checks = 0
    for lo, hi, w in regions:
        (win,) = w
        inside = [x for x in pts if lo <= x <= hi]
        for a, b in zip(inside, inside[1:]):
            for name, fn in acts.items():
                if name == win:
                    continue
                for x, strict in ((a, a not in bounds), ((a+b)/2, True), (b, b not in bounds)):
                    d = scout.at(acts[win], x)-scout.at(fn, x)
                    assert d > 0 if strict else d >= 0
                    checks += 1
    return checks


def run_interval(lo, hi, sample, collect=(True, True)):
    """Generic computation at a rational sample, then uniform verification."""
    assert lo < sample < hi
    G.sample = QQ(sample.numerator, sample.denominator)
    G.log = []
    pa = scout.Parametric(model(cost=G(C), collect=collect))
    regions = pa.regions()
    cert = certificate(pa)
    cert["strict_checks"] = strict_check(pa, regions)
    calls, distinct, failures = verify(lo, hi, [endpoint_minpoly(lo), endpoint_minpoly(hi)])
    return dict(regions=regions, states=cert["states"], affine_interval_checks=cert["affine_interval_checks"],
                strict_checks=cert["strict_checks"], decisions=calls, distinct_decision_functions=distinct, failures=failures)


def run_point(cost, collect=(True, True)):
    """Exact fixed-cost computation and Bellman certificate (cost in Q or Q(sqrt M))."""
    pa = scout.Parametric(model(cost=cost, collect=collect))
    regions = pa.regions()
    cert = certificate(pa)
    return dict(regions=regions, states=cert["states"], affine_interval_checks=cert["affine_interval_checks"],
                strict_checks=strict_check(pa, regions))


def tau(c):
    return (F(15, 19), (132-420*c)/(50+35*c), (288-780*c)/(250-535*c))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    A.M = 632689
    lo, hi = F(19, 100), F(1, 5)
    cstar = A(F(2407, 8400), F(-1, 8400))   # smaller root of 21000c^2-12035c+1536
    assert lo < cstar < hi and 21000*cstar*cstar-12035*cstar+1536 == 0
    pieces = [(lo, cstar, F(1915, 10000)), (cstar, hi, F(196, 1000))]
    record = dict(cost_interval=[str(lo), str(hi)], split_point="(2407-sqrt(632689))/8400", intervals=[], points=[])
    for a, b, s in pieces:
        r = run_interval(a, b, s)
        assert not r["failures"], r["failures"][:5]
        # Only B's intermediate reading collectable: same regions and boundaries.
        rb = run_interval(a, b, s, (False, True))
        assert not rb["failures"], rb["failures"][:5]
        assert [(G(x).sym(), G(y).sym(), w) for x, y, w in rb["regions"]] == \
            [(G(x).sym(), G(y).sym(), w) for x, y, w in r["regions"]]
        record.setdefault("intervals_B_only", []).append(dict(
            sample=str(s), states=rb["states"], strict_checks=rb["strict_checks"], decisions=rb["decisions"],
            distinct_decision_functions=rb["distinct_decision_functions"]))
        got = [(G(x).sym(), G(y).sym(), w) for x, y, w in r["regions"]]
        c = sympy.Symbol("c")
        want = [sympy.Rational(15, 19), (132-420*c)/(50+35*c), (288-780*c)/(250-535*c)]
        bounds = [g[1] for g in got[:-1]]
        assert [w for *_, w in got] == [("0",), ("1",), ("0",), ("1",)]
        assert all(sympy.simplify(u-v) == 0 for u, v in zip(bounds, want))
        record["intervals"].append(dict(open_interval=[str(a) if isinstance(a, F) else "c*", str(b) if isinstance(b, F) else "c*"],
                                        sample=str(s), states=r["states"], strict_checks=r["strict_checks"], affine_interval_checks=r["affine_interval_checks"],
                                        decisions=r["decisions"], distinct_decision_functions=r["distinct_decision_functions"],
                                        regions=[dict(lo=str(x), hi=str(y), action="A" if w == ("0",) else "B") for x, y, w in got]))
    for name, c in (("19/100", lo), ("c*", cstar), ("1/5", hi)):
        r = run_point(c)
        t = tau(c)
        assert F(0) < t[0] < t[1] < t[2] < F(1)
        assert r["regions"] == [(F(0), t[0], ("0",)), (t[0], t[1], ("1",)), (t[1], t[2], ("0",)), (t[2], F(1), ("1",))]
        assert run_point(c, (False, True))["regions"] == r["regions"]
        record["points"].append(dict(cost=name, states=r["states"], strict_checks=r["strict_checks"], affine_interval_checks=r["affine_interval_checks"]))
    # Root ordering 0 < tau1 < tau2 < tau3 < 1 on the whole closed interval.
    c = sympy.Symbol("c")
    t1, t2, t3 = sympy.Rational(15, 19), (132-420*c)/(50+35*c), (288-780*c)/(250-535*c)
    for e in (t2-t1, t3-t2, 1-t3, 50+35*c, 250-535*c):
        num, den = sympy.fraction(sympy.together(e))
        for q in (num, den):
            P = sympy.Poly(q, c)
            assert P.degree() <= 0 or P.count_roots(sympy.Rational(19, 100), sympy.Rational(1, 5)) == 0
        assert e.subs(c, sympy.Rational(39, 200)) > 0
    record["root_ordering_on_closed_interval"] = True
    # The split at c* is a branch change of the computation only: the symbolic
    # value and action functions of every state agree on the two sides.
    sides = []
    for s in (F(1915, 10000), F(196, 1000)):
        G.sample = QQ(s.numerator, s.denominator); G.log = []
        pa = scout.Parametric(model(cost=G(C))); pa.value(((), ()))
        sides.append({h: {k: tuple(tuple(str(G(v).sym()) for v in seg) for seg in fn) for k, fn in acts.items()}
                      | {"value": tuple(tuple(str(G(v).sym()) for v in seg) for seg in pa.value(h))}
                      for h, acts in pa.actions.items()})
    assert sides[0] == sides[1]
    record["symbolic_outputs_identical_across_split"] = True
    # Contrast regimes: A is the unique optimal first action for every prior.
    for name, collect in (("terminal_only", (False, False)), ("A_intermediate_only", (True, False))):
        r = run_interval(lo, hi, F(1955, 10000), collect)
        assert not r["failures"], (name, r["failures"][:5])
        assert [w for *_, w in r["regions"]] == [("0",)], (name, r["regions"])
        for c in (lo, hi):
            assert run_point(c, collect)["regions"] == [(F(0), F(1), ("0",))]
        record["contrast_"+name] = dict(decisions=r["decisions"], distinct_decision_functions=r["distinct_decision_functions"],
                                        states=r["states"], strict_checks=r["strict_checks"], unique_first_action="A on (0,1)")
    record["theorem_proved"] = True
    print(json.dumps(record, indent=2))
    if args.out:
        args.out.write_text(json.dumps(record, indent=2)+"\n")


if __name__ == "__main__":
    main()
