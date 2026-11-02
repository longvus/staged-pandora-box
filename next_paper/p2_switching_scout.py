#!/usr/bin/env python3
"""Exact acquisition-region scout for collection-induced switching.

Both arms have a signal stage followed by a binary
collectable terminal reward. A binary latent variable couples their laws.
The parametric Bellman recursion uses unnormalized conditional-history
likelihoods, avoiding posterior division when computing affine envelopes.
"""
import argparse
from dataclasses import dataclass, asdict
from fractions import Fraction as F
from functools import lru_cache
import hashlib
from itertools import product
import json
from pathlib import Path
import random

from staged_pandora_exact import Node, Oracle, nested, clear_caches


@dataclass(frozen=True)
class Model:
    a: tuple
    b: tuple
    reliability: tuple
    signal_cost: tuple
    terminal_cost: tuple
    signal_score: tuple = (F(2, 5), F(3, 5))
    q0: F = F(0)
    collect: tuple = (False, False)

    def json(self):
        return {key: [str(x) for x in value] if isinstance(value, tuple) and key != "collect"
                else value if key == "collect" else str(value)
                for key, value in asdict(self).items()}

    def cost(self, arm, prefix):
        return self.signal_cost[arm] if not prefix else self.terminal_cost[arm]

    def incumbent(self, history):
        q = self.q0
        for i, prefix in enumerate(history):
            if prefix and self.collect[i]:
                q = max(q, self.signal_score[i]*prefix[0])
            if len(prefix) == 2:
                q = max(q, F(prefix[1]))
        return q


def scenarios(model):
    result = []
    for s0, h0, s1, h1 in product((0, 1), repeat=4):
        weights = []
        for theta in (0, 1):
            prob = F(1)
            for i, s, h in ((0, s0, h0), (1, s1, h1)):
                ph = (model.a, model.b)[i][theta]
                r = model.reliability[i]
                prob *= (ph if h else 1-ph)*(r if s == h else 1-r)
            weights.append(prob)
        result.append((((s0, h0), (s1, h1)), tuple(weights)))
    assert all(sum(w[t] for _, w in result) == 1 for t in (0, 1))
    return tuple(result)


# A piecewise affine function is ((lo, hi, intercept, slope), ...).
def affine(intercept, slope):
    return ((F(0), F(1), intercept, slope),)


def line_at(fn, p):
    return next((a, b) for lo, hi, a, b in fn if lo <= p <= hi)


def at(fn, p):
    a, b = line_at(fn, p)
    return a+b*p


def combine(functions, operation):
    points = sorted({x for fn in functions for seg in fn for x in seg[:2]})
    result = []
    for lo, hi in zip(points, points[1:]):
        lines = [line_at(fn, (lo+hi)/2) for fn in functions]
        if operation == "sum":
            pieces = [(lo, hi, sum(a for a, _ in lines), sum(b for _, b in lines))]
        else:
            breaks = {lo, hi}
            for (a, b), (c, d) in product(lines, repeat=2):
                if b != d and lo < (c-a)/(b-d) < hi:
                    breaks.add((c-a)/(b-d))
            bounds = sorted(breaks)
            pieces = []
            for left, right in zip(bounds, bounds[1:]):
                a, b = max(lines, key=lambda line: line[0]+line[1]*(left+right)/2)
                pieces.append((left, right, a, b))
        for seg in pieces:
            if result and result[-1][2:] == seg[2:]:
                result[-1] = (result[-1][0], seg[1], seg[2], seg[3])
            else:
                result.append(seg)
    return tuple(result)


class Parametric:
    def __init__(self, model):
        self.model = model
        self.worlds = scenarios(model)
        self.actions = {}

        @lru_cache(None)
        def likelihood(history):
            return tuple(sum(w[t] for paths, w in self.worlds
                             if all(path[:len(prefix)] == prefix for path, prefix in zip(paths, history)))
                         for t in (0, 1))

        @lru_cache(None)
        def value(history):
            l0, l1 = likelihood(history)
            q = model.incumbent(history)
            actions = {"stop": affine(q*l0, q*(l1-l0))}
            for i, prefix in enumerate(history):
                if len(prefix) == 2:
                    continue
                children = []
                for y in (0, 1):
                    nxt = history[:i]+(prefix+(y,),)+history[i+1:]
                    if any(likelihood(nxt)):
                        children.append(value(nxt))
                c = model.cost(i, prefix)
                actions[str(i)] = combine(children+[affine(-c*l0, -c*(l1-l0))], "sum")
            self.actions[history] = actions
            return combine(list(actions.values()), "max")

        self.value = value
        self.likelihood = likelihood

    def regions(self):
        self.value(((), ()))
        actions = self.actions[((), ())]
        points = {F(0), F(1)}
        points.update(x for fn in actions.values() for seg in fn for x in seg[:2])
        initial = sorted(points)
        for lo, hi in zip(initial, initial[1:]):
            lines = [line_at(fn, (lo+hi)/2) for fn in actions.values()]
            for (a, b), (c, d) in product(lines, repeat=2):
                if b != d and lo < (c-a)/(b-d) < hi:
                    points.add((c-a)/(b-d))
        regions = []
        for lo, hi in zip(sorted(points), sorted(points)[1:]):
            mid = (lo+hi)/2
            best = max(at(fn, mid) for fn in actions.values())
            winners = tuple(name for name, fn in actions.items() if at(fn, mid) == best)
            if regions and regions[-1][2] == winners:
                regions[-1] = (regions[-1][0], hi, winners)
            else:
                regions.append((lo, hi, winners))
        return regions


class Scalar:
    """Separate conditional-scenario Bellman recursion for a specified prior."""
    def __init__(self, model, p):
        self.model = model
        self.worlds = tuple((paths, (1-p)*weights[0]+p*weights[1]) for paths, weights in scenarios(model))

        @lru_cache(None)
        def matching(history):
            return tuple((paths, w) for paths, w in self.worlds if w and
                         all(path[:len(prefix)] == prefix for path, prefix in zip(paths, history)))

        @lru_cache(None)
        def actions(history):
            worlds = matching(history)
            mass = sum(w for _, w in worlds)
            assert mass > 0
            result = {"stop": model.incumbent(history)}
            for i, prefix in enumerate(history):
                if len(prefix) == 2:
                    continue
                total = -model.cost(i, prefix)
                for y in (0, 1):
                    prob = sum(w for paths, w in worlds if paths[i][len(prefix)] == y)/mass
                    if prob:
                        nxt = history[:i]+(prefix+(y,),)+history[i+1:]
                        total += prob*value(nxt)
                result[str(i)] = total
            return result

        @lru_cache(None)
        def value(history):
            return max(actions(history).values())

        @lru_cache(None)
        def local_action(history):
            worlds = matching(history)
            total = sum(w for _, w in worlds)

            def arm(i, prefix):
                if len(prefix) == 2:
                    return None
                mass = sum(w for paths, w in worlds if paths[i][:len(prefix)] == prefix)
                branches = []
                for y in (0, 1):
                    weight = sum(w for paths, w in worlds if paths[i][:len(prefix)+1] == prefix+(y,))
                    if weight:
                        reading = F(y) if len(prefix) else model.signal_score[i]*y
                        branches.append((weight/mass, reading, arm(i, prefix+(y,))))
                collect = bool(prefix) or model.collect[i]
                return Node(model.cost(i, prefix), collect, tuple(branches))

            gs = [(nested(arm(i, prefix))[0], -i, str(i)) for i, prefix in enumerate(history) if len(prefix) < 2]
            return max(gs)[2] if gs and max(gs)[0] > model.incumbent(history) else "stop"

        @lru_cache(None)
        def local_value(history):
            action = local_action(history)
            if action == "stop":
                return model.incumbent(history)
            i = int(action)
            prefix = history[i]
            worlds = matching(history)
            mass = sum(w for _, w in worlds)
            result = -model.cost(i, prefix)
            for y in (0, 1):
                prob = sum(w for paths, w in worlds if paths[i][len(prefix)] == y)/mass
                if prob:
                    nxt = history[:i]+(prefix+(y,),)+history[i+1:]
                    result += prob*local_value(nxt)
            return result

        self.value, self.actions = value, actions
        self.local_action, self.local_value = local_action, local_value


def scout_model(rng, independent=False):
    kernel = [F(i, 10) for i in (1, 2, 3, 5, 7, 8, 9)]
    a = tuple(rng.sample(kernel, 2))
    b = tuple(rng.sample(kernel, 2))
    if independent:
        a, b = (a[0], a[0]), (b[0], b[0])
    return Model(a, b,
                 tuple(rng.choice((F(3, 5), F(3, 4), F(9, 10))) for _ in range(2)),
                 tuple(rng.choice((F(1, 100), F(1, 25), F(2, 25))) for _ in range(2)),
                 tuple(rng.choice((F(1, 20), F(1, 5), F(2, 5), F(13, 20))) for _ in range(2)),
                 q0=rng.choice((F(0), F(1, 4), F(1, 2))))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--models", type=int, default=100)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    rng = random.Random(20261026)
    records, checked = [], 0
    for seed in range(args.models+10):
        base = scout_model(rng, independent=seed >= args.models)
        for collect in ((False, False), (True, False), (True, True)):
            model = Model(**{**asdict(base), "collect": collect})
            parametric = Parametric(model)
            regions = parametric.regions()
            probes = {F(0), F(1)}
            for lo, hi, _ in regions:
                probes.update((lo, (lo+hi)/2, hi))
            values = []
            for p in sorted(probes):
                scalar = Scalar(model, p)
                actual = scalar.actions(((), ()))
                represented = {key: at(fn, p) for key, fn in parametric.actions[((), ())].items()}
                assert actual == represented
                local = scalar.local_value(((), ()))
                optimum = max(actual.values())
                assert optimum >= local
                if seed >= args.models:
                    assert optimum == local
                values.append(dict(prior=str(p), value=str(optimum), local_value=str(local),
                                   local_action=scalar.local_action(((), ())), gap=str(optimum-local)))
                checked += 1
                clear_caches()
            strict = [winners[0] for _, _, winners in regions if len(winners) == 1]
            reentry = any(strict[i] == strict[k] and strict[i] != strict[j]
                          for i in range(len(strict)) for j in range(i+1, len(strict)) for k in range(j+1, len(strict)))
            record = dict(model_id=seed, independent_control=seed >= args.models,
                          model=model.json(), regions=[dict(lo=str(lo), hi=str(hi), actions=w) for lo, hi, w in regions],
                          strict_action_sequence=strict, reentry=reentry, scalar_checks=values)
            records.append(record)
    output = args.out/"records.jsonl"
    output.write_text("".join(json.dumps(r)+"\n" for r in records))
    rs = [r for r in records if not r["independent_control"]]
    best = max(rs, key=lambda r: len(r["strict_action_sequence"]))
    largest = max(rs, key=lambda r: max(F(v["gap"]) for v in r["scalar_checks"]))
    summary = dict(seed=20261026, dependent_models=args.models,
                   collection_variants=3, independent_control_models=10,
                   cells=len(records), scalar_action_value_checks=checked,
                   exact_failures=0, reentry_cells=sum(r["reentry"] for r in rs),
                   largest_strict_sequence=len(best["strict_action_sequence"]),
                   switching_witness=best, loss_witness=largest,
                   source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   records_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    (args.out/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps({k:v for k,v in summary.items() if not k.endswith("witness")}, indent=2))
    print("Switching witness:", json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
