#!/usr/bin/env python3
"""Reproduce the exact switching certificate of Proposition 6 and Appendix D.2 and compare it with the archived one."""

import argparse
from fractions import Fraction as F
import json
from pathlib import Path
import platform
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "next_paper"
ARCHIVE = ROOT / "reports" / "p2_switching_certificate"


def mathematical_summary(record):
    """Exclude incidental computation-order counters from the comparison."""
    return {
        "cost_interval": record["cost_interval"],
        "split_point": record["split_point"],
        "intervals": [
            {key: interval[key] for key in ("open_interval", "regions")}
            for interval in record["intervals"]
        ],
        "endpoint_costs": [point["cost"] for point in record["points"]],
        "root_ordering_on_closed_interval": record["root_ordering_on_closed_interval"],
        "symbolic_outputs_identical_across_split": record["symbolic_outputs_identical_across_split"],
        "terminal_only_action": record["contrast_terminal_only"]["unique_first_action"],
        "A_only_action": record["contrast_A_intermediate_only"]["unique_first_action"],
        "theorem_proved": record["theorem_proved"],
    }


def main():
    if not __debug__:
        raise SystemExit("Run without -O or PYTHONOPTIMIZE: assertions are proof checks.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "verification")
    args = parser.parse_args()
    output = args.out.resolve()
    if output == ROOT or output.is_relative_to(SRC) or output.is_relative_to(ROOT / "reports"):
        parser.error("Choose an output directory outside next_paper/ and reports/.")
    output.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    print("Checking every prior and the full closed terminal-cost interval...", flush=True)
    reproduced = output / "certificate.json"
    run = subprocess.run(
        [sys.executable, "-B", str(SRC / "p2_switching_certificate.py"), "--out", str(reproduced)],
        cwd=ROOT, capture_output=True, text=True,
    )
    (output / "certificate_stdout.txt").write_text(run.stdout)
    if run.returncode:
        (output / "certificate_stderr.txt").write_text(run.stderr)
        raise SystemExit("Symbolic certificate failed; inspect the output logs.")
    got = json.loads(reproduced.read_text())
    expected = json.loads((ARCHIVE / "certificate.json").read_text())
    if mathematical_summary(got) != mathematical_summary(expected):
        raise SystemExit("Reproduced mathematical conclusions differ from the archived certificate.")
    assert got["theorem_proved"] and got["root_ordering_on_closed_interval"]
    assert got["symbolic_outputs_identical_across_split"]

    sys.path.insert(0, str(SRC))
    import sympy
    from p2_switching_certificate import strict_check
    from p2_switching_scout import Parametric, Scalar, at
    from p2_switching_witness import certificate, difference, model

    print("Checking all four collection regimes and the displayed root calculation...", flush=True)
    fixed = []
    scalar_comparisons = 0
    root_identity = False
    for cost in (F(19, 100), F(39, 200), F(1, 5)):
        thresholds = (F(15, 19), (132-420*cost)/(50+35*cost), (288-780*cost)/(250-535*cost))
        switching = [(F(0), thresholds[0], ("0",)),
                     (thresholds[0], thresholds[1], ("1",)),
                     (thresholds[1], thresholds[2], ("0",)),
                     (thresholds[2], F(1), ("1",))]
        for collect in ((False, False), (True, False), (False, True), (True, True)):
            solver = Parametric(model(cost=cost, collect=collect))
            cert = certificate(solver)
            regions = solver.regions()
            assert regions == (switching if collect[1] else [(F(0), F(1), ("0",))])
            assert cert["states"] == 49
            cert["strict_checks"] = strict_check(solver, regions)
            actions = solver.actions[((), ())]
            points = sorted({x for fn in actions.values() for seg in fn for x in seg[:2]})
            probes = set(points) | {(a+b)/2 for a, b in zip(points, points[1:])}
            for prior in probes:
                scalar = Scalar(solver.model, prior)
                assert scalar.actions(((), ())) == {name: at(fn, prior) for name, fn in actions.items()}
                scalar_comparisons += len(actions)
            if cost == F(1, 5) and collect == (True, True):
                lo, hi, root = F(9, 13), F(63, 76), F(15, 19)
                assert (lo, hi, F(3, 400), -F(19, 2000)) in difference(solver)
                assert at(actions["0"], root) == at(actions["1"], root) > at(actions["stop"], root)
                root_identity = True
            fixed.append({
                "terminal_cost": str(cost), "intermediate_collection": collect,
                "regions": [{"lo": str(a), "hi": str(b), "actions": w} for a, b, w in regions],
                "certificate": cert,
            })
    assert root_identity
    if json.loads(json.dumps(fixed)) != json.loads((ARCHIVE / "fixed_cost_certificates.json").read_text()):
        raise SystemExit("Reproduced fixed-cost state certificates differ from the archive.")
    (output / "fixed_cost_certificates.json").write_text(json.dumps(fixed, indent=2)+"\n")
    summary = {
        "verified": True,
        "full_closed_cost_interval": ["19/100", "1/5"],
        "all_prior_symbolic_certificate_passed": True,
        "archived_mathematical_conclusions_reproduced": True,
        "fixed_cost_certificates_passed": len(fixed),
        "archived_fixed_cost_state_certificates_reproduced": True,
        "histories_per_fixed_cost_certificate": 49,
        "independent_scalar_root_action_comparisons": scalar_comparisons,
        "displayed_root_difference_verified": True,
        "environment": {"python": platform.python_version(), "sympy": sympy.__version__},
        "elapsed_seconds": round(time.perf_counter()-started, 3),
    }
    (output / "verification.json").write_text(json.dumps(summary, indent=2)+"\n")
    print("VERIFIED: full cost interval, all priors, all collection regimes, and displayed root calculation.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
