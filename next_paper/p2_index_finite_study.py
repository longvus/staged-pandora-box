#!/usr/bin/env python3
"""Execute P2_INDEX_STUDY_PROTOCOL.md; exact arithmetic and no benchmark inputs."""
import argparse
from fractions import Fraction as F
import json
from pathlib import Path
import random
import time

from staged_pandora_exact import Node, Oracle, clear_caches, nested, random_arm, successor

HERE = Path(__file__).resolve().parent
REGIMES = {"all": (3, (1, 2, 3)), "terminal": (4, (4,)),
           "mixed13": (3, (1, 3)), "mixed24": (4, (2, 4))}


def sampled_state(arms, rng):
    states, q = arms, F(0)
    for _ in range(3):
        av = [i for i, s in enumerate(states) if s is not None]
        if not av:
            break
        i = rng.choice(av)
        u = F(rng.randrange(4), 4)
        total = F(0)
        for p, y, child in states[i].branches:
            total += p
            if u < total:
                states, q = successor(states, q, i, y, child)
                break
    return states, q


def history_example():
    bad = Node(F(1, 10), True, ((F(1), F(0), None),))
    good = Node(F(1, 10), True, ((F(1), F(1), None),))
    second_bad = Node(F(1, 100), True, ((F(1), F(1, 2), bad),))
    second_good = Node(F(1, 100), True, ((F(1), F(1, 2), good),))
    root = Node(F(1, 100), False,
                ((F(1, 2), F(0), second_bad), (F(1, 2), F(1), second_good)))
    Oracle().verify((root,), F(0))
    assert nested(bad)[0] == -F(1, 10) and nested(good)[0] == F(9, 10)
    return {"histories": [[0, "1/2"], [1, "1/2"]], "incumbent": "1/2",
            "indices": [str(nested(bad)[0]), str(nested(good)[0])], "collectable": [2, 3]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=HERE / "p2_index_study")
    args = ap.parse_args()
    args.out.mkdir(exist_ok=True)
    records = []
    for regime, (depth, collectable) in REGIMES.items():
        for seed in range(2026102400, 2026102424):
            rng = random.Random(seed)
            arms = tuple(random_arm(rng, depth, collectable) for _ in range(2))
            sampled = sampled_state(arms, rng)
            for point, (states, q) in (("root", (arms, F(0))), ("history", sampled)):
                t0 = time.perf_counter()
                oracle = Oracle()
                nodes, actions = oracle.verify(states, q)
                result = {}
                for policy in ("index", "committed", "recomputed"):
                    v, st, ad = oracle.policy(states, q, policy)
                    result[policy] = {"value": str(v), "stop_loss": str(st), "advance_loss": str(ad)}
                    assert oracle.value(states, q) - v == st + ad
                assert oracle.value(states, q) == F(result["index"]["value"])
                records.append(dict(regime=regime, collectable=collectable, depth=depth, seed=seed,
                                    point=point, q=str(q), vstar=str(oracle.value(states, q)),
                                    nodes=nodes, actions=actions, policies=result,
                                    seconds=time.perf_counter() - t0))
            clear_caches()
        print(regime, "complete", flush=True)
    summary = {"instances": len(records), "nodes": sum(r["nodes"] for r in records),
               "actions": sum(r["actions"] for r in records), "exact_failures": 0,
               "history_example": history_example(), "regimes": {}}
    for regime in REGIMES:
        rs = [r for r in records if r["regime"] == regime]
        gain = sum(F(r["vstar"]) - F(r["q"]) for r in rs)
        group = dict(instances=len(rs), nodes=sum(r["nodes"] for r in rs),
                     actions=sum(r["actions"] for r in rs), optimal_gain=str(gain), policies={})
        for p in ("index", "committed", "recomputed"):
            stop = sum(F(r["policies"][p]["stop_loss"]) for r in rs)
            advance = sum(F(r["policies"][p]["advance_loss"]) for r in rs)
            group["policies"][p] = dict(loss=str(stop + advance), stop_loss=str(stop),
                                        advance_loss=str(advance),
                                        loss_share=float((stop + advance) / gain) if gain else None)
        summary["regimes"][regime] = group
    (args.out / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "regimes"}))


if __name__ == "__main__":
    main()
