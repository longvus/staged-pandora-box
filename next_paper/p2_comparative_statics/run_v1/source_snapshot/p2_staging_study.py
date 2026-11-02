#!/usr/bin/env python3
"""Run the full known-law grid in P2_STAGING_STUDY_PROTOCOL.md."""
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction as F
from functools import lru_cache
import hashlib
from itertools import product
import json
from pathlib import Path
import platform
import sys
import time

from staged_pandora_exact import Node, Oracle, clear_caches, nested, successor

HERE = Path(__file__).resolve().parent
ACCURACIES = tuple(map(F, ("1/2", "5/8", "3/4", "7/8", "1")))
SIGNAL_COSTS = tuple(map(F, ("1/200", "1/50", "1/20", "1/10")))
TERMINAL_COSTS = tuple(map(F, ("1/20", "1/5", "2/5", "3/5", "4/5")))
COLLECTIONS = {"terminal": (3,), "mixed": (1, 3), "all": (1, 2, 3)}
FAMILIES = ("constant_cost", "history_cost", "deterministic")
POLICIES = ("index", "stop_after_one", "lowest_index_first", "one_extra_at_stop")
MULTIPLIERS = (F(1), F(5, 4))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def posterior(accuracy, history):
    weights = []
    for h in (0, 1):
        w = F(1, 2)
        for s in history:
            w *= accuracy if s == h else 1 - accuracy
        weights.append(w)
    assert sum(weights) > 0
    return weights[1] / sum(weights)


def signal_arm(accuracy, signal_cost, terminal_cost, collectable,
               multiplier=F(1), history_cost=False):
    """Return arm and metadata; stages reveal scaled signals then H."""
    metadata = {}

    def build(history):
        stage = len(history) + 1
        ph = posterior(accuracy, history)
        if stage == 3:
            cost = terminal_cost
            if history_cost and history[0] != history[1]:
                cost *= F(3, 2)
            branches = tuple((p, F(h), None) for h, p in ((0, 1-ph), (1, ph)) if p)
        else:
            cost = signal_cost
            if history_cost and stage == 2:
                cost *= 1 + F(history[0], 2)
            ps1 = ph * accuracy + (1-ph) * (1-accuracy)
            branches = tuple((p, F(stage+1, 5)*s, build(history+(s,)))
                             for s, p in ((0, 1-ps1), (1, ps1)) if p)
        node = Node(multiplier*cost, stage in collectable, branches)
        metadata[node] = (stage, history)
        return node

    return build(()), metadata


def deterministic_arm(readings, signal_cost, terminal_cost, collectable, multiplier=F(1)):
    child, metadata = None, {}
    for stage in (3, 2, 1):
        cost = terminal_cost if stage == 3 else signal_cost
        node = Node(multiplier*cost, stage in collectable, ((F(1), readings[stage-1], child),))
        metadata[node] = (stage, tuple(readings[:stage-1]))
        child = node
    return child, metadata


def make_case(family, accuracy, signal_cost, terminal_cost, collectable):
    arms, metadata = [], {}
    for i, multiplier in enumerate(MULTIPLIERS):
        if family == "deterministic":
            readings = (F(0), F(1, 2), F(1)) if i == 0 else (F(0), F(3, 8), F(3, 4))
            root, md = deterministic_arm(readings, signal_cost, terminal_cost, collectable, multiplier)
        else:
            root, md = signal_arm(accuracy, signal_cost, terminal_cost, collectable,
                                  multiplier, family == "history_cost")
        arms.append(root)
        metadata.update(md)
    return tuple(arms), metadata


def index_action(states, q):
    scores = [(nested(s)[0], -i, i) for i, s in enumerate(states) if s is not None]
    scores = [entry for entry in scores if entry[0] is not None]
    return max(scores)[2] if scores and max(scores)[0] > q else None


@dataclass(frozen=True)
class Stats:
    reward: F = F(0)
    cost: F = F(0)
    stages: tuple = (F(0), F(0), F(0))
    stop_loss: F = F(0)
    advance_loss: F = F(0)

    @property
    def utility(self):
        return self.reward - self.cost

    def json(self):
        return dict(utility=str(self.utility), reward=str(self.reward), cost=str(self.cost),
                    stage_advances=[str(x) for x in self.stages],
                    advances=str(sum(self.stages)),
                    opened_unfinished=str(self.stages[0]-self.stages[2]),
                    stop_loss=str(self.stop_loss), advance_loss=str(self.advance_loss))


def evaluate(arms, metadata, oracle, policy):
    assert policy in POLICIES

    @lru_cache(None)
    def run(states, q, phase):
        i = index_action(states, q)
        next_phase = phase
        if policy == "stop_after_one" and phase == 1:
            i = None
        elif policy == "lowest_index_first" and phase == 0:
            candidates = [(nested(s)[0], j) for j, s in enumerate(states) if s is not None]
            i = min(candidates)[1] if candidates else None
        elif policy == "one_extra_at_stop" and phase == 0 and i is None:
            candidates = [(s.cost, j) for j, s in enumerate(states) if s is not None]
            i = min(candidates)[1] if candidates else None
            next_phase = 1
        if policy in ("stop_after_one", "lowest_index_first"):
            next_phase = 1
        if i is None:
            return Stats(reward=q, stop_loss=oracle.value(states, q)-q)
        node = states[i]
        reward, cost, stop, advance = F(0), node.cost, F(0), oracle.action_gap(states, q, i)
        stages = [F(int(j+1 == metadata[node][0])) for j in range(3)]
        for p, y, child in node.branches:
            ss, qq = successor(states, q, i, y, child)
            st = run(ss, qq, next_phase)
            reward += p * st.reward
            cost += p * st.cost
            stop += p * st.stop_loss
            advance += p * st.advance_loss
            stages = [x + p*z for x, z in zip(stages, st.stages)]
        result = Stats(reward, cost, tuple(stages), stop, advance)
        assert oracle.value(states, q)-result.utility == stop+advance >= 0
        return result

    result = run(arms, F(0), 0)
    assert oracle.value(arms, F(0))-result.utility == result.stop_loss+result.advance_loss
    return result


def design():
    return dict(accuracies=[str(x) for x in ACCURACIES],
                signal_costs=[str(x) for x in SIGNAL_COSTS],
                terminal_costs=[str(x) for x in TERMINAL_COSTS],
                collections=COLLECTIONS, families=FAMILIES, policies=POLICIES,
                multipliers=[str(x) for x in MULTIPLIERS], expected_cells=660,
                figure_signal_cost="1/50", time_cap_seconds=1800)


def cell_grid():
    for family in FAMILIES:
        accuracies = (None,) if family == "deterministic" else ACCURACIES
        for regime, a, k, c in product(COLLECTIONS, accuracies, SIGNAL_COSTS, TERMINAL_COSTS):
            yield family, regime, a, k, c


def summarize(rows):
    result = dict(cells=len(rows), exact_failures=0,
                  bellman_state_evaluations=sum(r["bellman_states"] for r in rows),
                  advance_action_checks=sum(r["advance_actions"] for r in rows), groups={})
    for family, regime in product(FAMILIES, COLLECTIONS):
        rs = [r for r in rows if r["family"] == family and r["regime"] == regime]
        group = dict(cells=len(rs), root_stop_cells=sum(r["root_action"] is None for r in rs),
                     policies={})
        for policy in POLICIES:
            fields = ("utility", "reward", "cost", "advances", "opened_unfinished", "stop_loss", "advance_loss")
            means = {key: str(sum(F(r["policies"][policy][key]) for r in rs)/len(rs)) for key in fields}
            means["loss"] = str(F(means["stop_loss"])+F(means["advance_loss"]))
            means["positive_loss_cells"] = sum(F(r["policies"][policy]["stop_loss"])+
                                               F(r["policies"][policy]["advance_loss"]) > 0 for r in rs)
            group["policies"][policy] = means
        result["groups"][family+"/"+regime] = group
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=HERE/"p2_staging_study"/"run_v1")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), HERE/"staged_pandora_exact.py", HERE/"P2_STAGING_STUDY_PROTOCOL.md"]
    freeze = dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(), design=design(),
                  sources={str(p.relative_to(HERE)): sha(p) for p in sources},
                  python=sys.version, executable=sys.executable, platform=platform.platform())
    (args.out/"design_freeze.json").write_text(json.dumps(freeze, indent=2)+"\n")
    print("Frozen design:", sha(args.out/"design_freeze.json"), flush=True)
    rows = []
    started = time.monotonic()
    try:
        with (args.out/"records.jsonl").open("x") as stream:
            for number, (family, regime, a, k, c) in enumerate(cell_grid(), 1):
                if time.monotonic()-started > 1800:
                    raise TimeoutError("Declared 30-minute study limit exceeded")
                arms, metadata = make_case(family, a, k, c, COLLECTIONS[regime])
                oracle = Oracle()
                states, actions = oracle.verify(arms, F(0))
                results = {p: evaluate(arms, metadata, oracle, p).json() for p in POLICIES}
                optimum = oracle.value(arms, F(0))
                assert F(results["index"]["utility"]) == optimum
                assert F(results["index"]["stop_loss"]) == F(results["index"]["advance_loss"]) == 0
                row = dict(cell=number, family=family, regime=regime,
                           accuracy=None if a is None else str(a), signal_cost=str(k), terminal_cost=str(c),
                           vstar=str(optimum), initial_indices=[str(nested(s)[0]) for s in arms],
                           root_action=index_action(arms, F(0)),
                           bellman_states=states, advance_actions=actions, policies=results)
                stream.write(json.dumps(row)+"\n")
                stream.flush()
                rows.append(row)
                clear_caches()
                if number % 20 == 0:
                    print(f"{number}/660 {family}/{regime}; {time.monotonic()-started:.1f}s", flush=True)
        assert len(rows) == design()["expected_cells"]
        summary = summarize(rows)
        summary["elapsed_seconds"] = time.monotonic()-started
        summary["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        summary["source_hashes_unchanged"] = all(sha(p) == freeze["sources"][str(p.relative_to(HERE))] for p in sources)
        assert summary["source_hashes_unchanged"]
        (args.out/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        print(json.dumps({k: v for k, v in summary.items() if k != "groups"}), flush=True)
    except BaseException as exc:
        (args.out/"failure.json").write_text(json.dumps(dict(type=type(exc).__name__, message=str(exc),
                                                          completed_cells=len(rows)), indent=2)+"\n")
        raise


if __name__ == "__main__":
    main()
