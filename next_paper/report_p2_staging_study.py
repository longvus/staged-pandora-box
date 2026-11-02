#!/usr/bin/env python3
"""Audit and render the complete controlled staging study, with no refitting."""
import argparse
from fractions import Fraction as F
import hashlib
from itertools import product
import json
from pathlib import Path

from p2_staging_study import (ACCURACIES, COLLECTIONS, FAMILIES, POLICIES,
                              SIGNAL_COSTS, TERMINAL_COSTS, cell_grid)

LABELS = {"terminal": "Terminal only: {3}", "mixed": "Mixed: {1,3}", "all": "Every stage: {1,2,3}"}
FAMILY_LABELS = {"constant_cost": "Constant costs", "history_cost": "History-dependent costs",
                 "deterministic": "Deterministic controls"}
POLICY_LABELS = {"index": "Exact index", "stop_after_one": "Stop after one advance",
                 "lowest_index_first": "Force lowest index first", "one_extra_at_stop": "One extra at stop"}
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(run):
    freeze = json.loads((run/"design_freeze.json").read_text())
    summary = json.loads((run/"summary.json").read_text())
    rows = [json.loads(line) for line in (run/"records.jsonl").read_text().splitlines()]
    assert not (run/"failure.json").exists()
    assert all(sha(HERE/name) == digest for name, digest in freeze["sources"].items())
    expected = {(family, regime, None if a is None else str(a), str(k), str(c))
                for family, regime, a, k, c in cell_grid()}
    keys = [(r["family"], r["regime"], r["accuracy"], r["signal_cost"], r["terminal_cost"]) for r in rows]
    assert len(rows) == len(set(keys)) == len(expected) == 660
    assert set(keys) == expected
    assert summary["cells"] == len(rows)
    for r in rows:
        assert set(r["policies"]) == set(POLICIES)
        v = F(r["vstar"])
        assert v >= 0
        gs = list(map(F, r["initial_indices"]))
        assert r["root_action"] == (gs.index(max(gs)) if max(gs) > 0 else None)
        for policy, st in r["policies"].items():
            u, reward, cost, stop, adv = (F(st[k]) for k in ("utility", "reward", "cost", "stop_loss", "advance_loss"))
            assert u == reward-cost
            assert v-u == stop+adv >= 0 and min(stop, adv, cost) >= 0
            assert 0 <= reward <= 1
            first, second, third = map(F, st["stage_advances"])
            assert 2 >= first >= second >= third >= 0
            assert F(st["advances"]) == first+second+third
            assert F(st["opened_unfinished"]) == first-third
            if policy == "index":
                assert u == v and stop == adv == 0
            elif policy == "stop_after_one":
                assert adv == 0
            else:
                assert stop == 0
    for key, group in summary["groups"].items():
        rs = [r for r in rows if r["family"]+"/"+r["regime"] == key]
        assert group["cells"] == len(rs)
        assert group["root_stop_cells"] == sum(r["root_action"] is None for r in rs)
        for policy in POLICIES:
            for metric in ("utility", "reward", "cost", "advances", "opened_unfinished", "stop_loss", "advance_loss"):
                assert F(group["policies"][policy][metric]) == sum(F(r["policies"][policy][metric]) for r in rs)/len(rs)
    assert sum(r["bellman_states"] for r in rows) == summary["bellman_state_evaluations"] == 48840
    assert sum(r["advance_actions"] for r in rows) == summary["advance_action_checks"] == 74160
    assert summary["exact_failures"] == 0
    by_key = {(r["family"], r["accuracy"], r["signal_cost"], r["terminal_cost"], r["regime"]): F(r["vstar"]) for r in rows}
    for r in rows:
        key = (r["family"], r["accuracy"], r["signal_cost"], r["terminal_cost"])
        assert by_key[key+("all",)] >= by_key[key+("mixed",)] >= by_key[key+("terminal",)]
    return rows, summary


def figures(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    lookup = {(r["family"], r["regime"], r["accuracy"], r["signal_cost"], r["terminal_cost"]): r for r in rows}
    def row(family, regime, a, k, c):
        return lookup[family, regime, None if a is None else str(a), str(k), str(c)]
    x = list(map(float, ACCURACIES))
    colors = plt.get_cmap("viridis")(np.linspace(.1, .9, len(TERMINAL_COSTS)))
    names = []
    def save(fig, name):
        fig.savefig(output/(name+".pdf"), bbox_inches="tight", metadata={"CreationDate": None})
        fig.savefig(output/(name+".png"), dpi=180, bbox_inches="tight")
        names.append(name)
        plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(10.2, 8.0), sharex=True, sharey="row")
    fig.subplots_adjust(top=.86, bottom=.08, left=.08, right=.99, hspace=.16, wspace=.10)
    for j, regime in enumerate(COLLECTIONS):
        for c, color in zip(TERMINAL_COSTS, colors):
            rs = [row("constant_cost", regime, a, F(1, 50), c) for a in ACCURACIES]
            utilities = [F(r["vstar"]) for r in rs]
            axes[0, j].plot(x, list(map(float, utilities)), marker="o", ms=3, color=color, label=f"c={float(c):g}")
            axes[1, j].plot(x, [float(v-utilities[0]) for v in utilities], marker="o", ms=3, color=color)
            axes[2, j].plot(x, [float(F(r["policies"]["index"]["stage_advances"][2])) for r in rs], marker="o", ms=3, color=color)
        axes[0, j].set_title(LABELS[regime])
        for i in range(3):
            axes[i, j].grid(alpha=.2)
            axes[i, j].set_xticks(x, [".5", ".625", ".75", ".875", "1"])
        axes[1, j].axhline(0, color="black", lw=.6)
        axes[2, j].set_xlabel("Signal accuracy a")
    axes[0, 0].set_ylabel("Exact expected utility")
    axes[1, 0].set_ylabel("Utility minus a=.5")
    axes[2, 0].set_ylabel("Expected terminal advances")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .95), ncol=5, frameon=False)
    fig.suptitle("Known-law staging: constant costs, k=.02", y=.99)
    save(fig, "staging_signal_utility")

    fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.0), layout="constrained")
    all_values = [float(F(r["vstar"])) for r in rows if r["family"] != "deterministic" and r["terminal_cost"] == "3/5"]
    for i, family in enumerate(("constant_cost", "history_cost")):
        for j, regime in enumerate(COLLECTIONS):
            z = [[float(F(row(family, regime, a, k, F(3, 5))["vstar"])) for a in ACCURACIES] for k in SIGNAL_COSTS]
            im = axes[i, j].imshow(z, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=max(all_values))
            axes[i, j].set_xticks(range(len(ACCURACIES)), [".5", ".625", ".75", ".875", "1"])
            axes[i, j].set_yticks(range(len(SIGNAL_COSTS)), [f"{float(k):g}" for k in SIGNAL_COSTS])
            if i == 0:
                axes[i, j].set_title(LABELS[regime])
            if j == 0:
                axes[i, j].set_ylabel(FAMILY_LABELS[family]+"\nSignal cost k")
            if i == 1:
                axes[i, j].set_xlabel("Signal accuracy a")
            for ik, ia in product(range(len(SIGNAL_COSTS)), range(len(ACCURACIES))):
                axes[i, j].text(ia, ik, f"{z[ik][ia]:.3f}", ha="center", va="center", fontsize=7,
                                color="white" if z[ik][ia] < max(all_values)*.55 else "black")
    fig.colorbar(im, ax=axes, shrink=.8, label="Exact expected utility")
    fig.suptitle("Cost and information grid: terminal base cost c=.6")
    save(fig, "staging_cost_grid")

    fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.8), sharex=True, sharey="row")
    fig.subplots_adjust(top=.82, bottom=.10, left=.08, right=.99, hspace=.18, wspace=.10)
    for j, regime in enumerate(COLLECTIONS):
        for c, color in zip(TERMINAL_COSTS, colors):
            for i, family in enumerate(("constant_cost", "history_cost")):
                rs = [row(family, regime, a, F(1, 50), c) for a in ACCURACIES]
                axes[i, j].plot(x, [float(F(r["initial_indices"][0])) for r in rs], marker="o", ms=3, color=color)
        axes[0, j].set_title(LABELS[regime])
        axes[1, j].set_xlabel("Signal accuracy a")
        for i in range(2):
            axes[i, j].axhline(0, color="black", lw=.6)
            axes[i, j].grid(alpha=.2)
            axes[i, j].set_xticks(x, [".5", ".625", ".75", ".875", "1"])
    axes[0, 0].set_ylabel("Constant costs\nInitial arm-1 index")
    axes[1, 0].set_ylabel("History costs\nInitial arm-1 index")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .94), ncol=5, frameon=False)
    fig.suptitle("Reservation values under both cost laws: k=.02", y=.99)
    save(fig, "staging_initial_indices")

    fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.8), sharex=True, sharey="row")
    fig.subplots_adjust(top=.82, bottom=.10, left=.08, right=.99, hspace=.18, wspace=.10)
    for j, regime in enumerate(COLLECTIONS):
        for k, color in zip(SIGNAL_COSTS, plt.get_cmap("plasma")(np.linspace(.1, .8, 4))):
            rs = [row("deterministic", regime, None, k, c) for c in TERMINAL_COSTS]
            axes[0, j].plot(list(map(float, TERMINAL_COSTS)), [float(F(r["vstar"])) for r in rs],
                            marker="o", ms=3, color=color, label=f"k={float(k):g}")
            axes[1, j].plot(list(map(float, TERMINAL_COSTS)),
                            [float(F(r["policies"]["index"]["stage_advances"][2])) for r in rs], marker="o", ms=3, color=color)
        axes[0, j].set_title(LABELS[regime])
        axes[1, j].set_xlabel("Terminal base cost c")
        for i in range(2):
            axes[i, j].grid(alpha=.2)
    axes[0, 0].set_ylabel("Exact expected utility")
    axes[1, 0].set_ylabel("Expected terminal advances")
    handles2, labels2 = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles2, labels2, loc="upper center", bbox_to_anchor=(.5, .94), ncol=4, frameon=False)
    fig.suptitle("Later collection without learning: deterministic controls", y=.99)
    save(fig, "staging_deterministic_controls")
    return names


def make_report(rows, summary, output):
    lookup = {(r["family"], r["regime"], r["accuracy"], r["signal_cost"], r["terminal_cost"]): r for r in rows}
    contrast = {}
    for family, regime in product(FAMILIES[:2], COLLECTIONS):
        pairs, end = [], []
        for k, c in product(SIGNAL_COSTS, TERMINAL_COSTS):
            rs = [lookup[family, regime, str(a), str(k), str(c)] for a in ACCURACIES]
            us = [F(r["vstar"]) for r in rs]
            pairs.extend(v-u for u, v in zip(us, us[1:]))
            end.append(us[-1]-us[0])
        contrast[family+"/"+regime] = dict(adjacent_increases=sum(x > 0 for x in pairs),
                                          adjacent_ties=sum(x == 0 for x in pairs),
                                          adjacent_decreases=sum(x < 0 for x in pairs),
                                          endpoint_min=str(min(end)), endpoint_max=str(max(end)),
                                          endpoint_mean=str(sum(end)/len(end)))
    assert contrast["constant_cost/terminal"]["adjacent_decreases"] == 0
    selected = {}
    for family, regime, a, k, c in (
            ("constant_cost", "terminal", "1/2", "1/50", "3/5"),
            ("constant_cost", "terminal", "1", "1/50", "3/5"),
            ("constant_cost", "all", "1/2", "1/50", "3/5"),
            ("constant_cost", "all", "1", "1/50", "3/5"),
            ("deterministic", "terminal", None, "1/50", "4/5"),
            ("deterministic", "all", None, "1/50", "4/5")):
        r = lookup[family, regime, a, k, c]
        selected[f"{family}/{regime}/a={a}/k={k}/c={c}"] = {
            "utility": r["vstar"], "initial_indices": r["initial_indices"],
            "stage_advances": r["policies"]["index"]["stage_advances"],
            "opened_unfinished": r["policies"]["index"]["opened_unfinished"]}
    analysis = dict(signal_accuracy_contrasts=contrast, illustrative_cells=selected)
    (output/"analysis.json").write_text(json.dumps(analysis, indent=2)+"\n")

    lines = ["# Staging grid: the value of staging", "",
             "Completed the 660-cell full grid using known independent-arm laws and exact rational arithmetic. "
             "There is no learned surrogate and no recorded-curve input. The planning and observation laws coincide.", "",
             f"The staged index agrees exactly with direct joint Bellman optimization across "
             f"{summary['bellman_state_evaluations']:,} state evaluations and {summary['advance_action_checks']:,} "
             "advance-action checks. All maximal-index actions and action-loss identities pass. "
             "These are computational checks of specified finite laws, not a proof or an empirical comparison with another optimizer.", "",
             "## Design and interpretation", "",
             "Two three-stage arms reveal two noisy signals followed by a Bernoulli terminal reward. "
             "Signal accuracy has five levels; signal and terminal costs have four and five levels. "
             "Each of the three collection regimes has 100 constant-cost cells, 100 history-cost cells, "
             "and 20 deterministic controls. The terminal marginal is fixed at Bernoulli(1/2) in both stochastic families. "
             "All cells are retained; there are no Monte Carlo estimates or confidence intervals. "
             "State counts include separate solves on related grid points and are not independent observations.", "",
             "For terminal-only collection and constant costs, the accuracy intervention isolates information. "
             "For intermediate collection, it also changes dependence between eligible rewards. In the history-cost "
             "family it changes the distribution of the mismatch surcharge. Comparisons between collection sets "
             "change which observations are eligible for the objective; they are mechanism interventions, not fair "
             "optimizer comparisons.", "",
             "## Main findings", "",
             "1. Under terminal-only collection and constant costs, utility never decreases as signals become "
             "more accurate: 62 of 80 adjacent comparisons increase and 18 tie. The mean perfect-minus-uninformative "
             "utility difference over the 20 cost pairs is 49/320 = 0.153125. This is a descriptive average over "
             "the chosen grid, not an estimate for practical tasks.",
             "2. At k=.02 and c=.6, terminal-only utility rises from zero with uninformative signals to "
             "171/800 = 0.213750 with perfect signals. The latter policy opens 1.5 arms in expectation but "
             "completes only 0.75, illustrating adaptive abandonment. Terminal usage can increase with information "
             "when it makes an otherwise unprofitable inspection worthwhile; it need not always decrease.",
             "3. With every stage collectable and constant costs, utility decreases in 66 of the 80 adjacent "
             "accuracy comparisons. Here early readings are rewards as well as information: changing accuracy "
             "changes their dependence and the opportunities to retain a high early score. This intervention "
             "does not isolate information alone. The index still matches Bellman exactly in every case.",
             "4. Deterministic controls show that later collection matters even without learning. At k=.02 and "
             "c=.8, terminal-only utility is 0.16, whereas every-stage collection yields 0.46 by collecting "
             "the intermediate 0.5 reward and stopping before the expensive terminal stage. These are different "
             "collection objectives applied to the same underlying process.",
             "5. History-dependent costs preserve exact index/Bellman agreement throughout all 300 cells. "
             "Expanding the collectable set never lowers optimal utility at a fixed law and cost setting; "
             "the audit checks this across all three regimes.", "",
             "## Signal-accuracy effects over the complete cost grid", "",
             "Counts below describe 80 adjacent-accuracy comparisons per row (20 cost pairs times four increments). "
             "Endpoint contrasts compare a=1 with a=1/2 over the same 20 cost pairs.", "",
             "| Cost law | Collection | Increases / ties / decreases | Mean endpoint utility change | Endpoint range |",
             "|---|---|---:|---:|---:|"]
    for family, regime in product(FAMILIES[:2], COLLECTIONS):
        ct = contrast[family+"/"+regime]
        lines.append(f"| {FAMILY_LABELS[family]} | {LABELS[regime]} | {ct['adjacent_increases']} / {ct['adjacent_ties']} / {ct['adjacent_decreases']} | "
                     f"{float(F(ct['endpoint_mean'])):.6f} | [{float(F(ct['endpoint_min'])):.6f}, {float(F(ct['endpoint_max'])):.6f}] |")
    lines += ["", "## Illustrative cells", "",
              "These transparent examples accompany the complete-grid summaries; they are not a selected performance benchmark.", "",
              "| Cell | Exact utility | Expected advances by stage | Expected opened but unfinished arms |",
              "|---|---:|---|---:|"]
    for name, ex in selected.items():
        lines.append(f"| {name} | {ex['utility']} ({float(F(ex['utility'])):.6f}) | {', '.join(ex['stage_advances'])} | {ex['opened_unfinished']} |")
    lines += ["", "## Action-loss diagnostics", "",
              "Each row is a uniform mean over its declared family/collection grid. Stop after one advance "
              "uses the optimal initial action, then stops. Force lowest index first forces one initial action "
              "and resumes the optimal rule. One extra at stop forces the cheapest remaining advance at the "
              "first optimal stopping point, then resumes the optimal rule. These are constructed diagnostic "
              "deviations, not published acquisition methods. All four policies, including zero-loss cases, are recorded.", "",
              "| Family | Collection | Diagnostic | Mean utility loss | Stop loss | Advance loss | Positive loss cells |",
              "|---|---|---|---:|---:|---:|---:|"]
    csv_lines = ["family,collection,policy,cells,mean_utility_loss,mean_stop_loss,mean_advance_loss,positive_loss_cells"]
    tex = [r"% Generated by report_p2_staging_study.py; all declared cells included.",
           r"\begin{table}[htb]\centering\footnotesize",
           r"\caption{Controlled known-law action-loss diagnostics. Entries are uniform mean utility losses over 100 cells per stochastic family and collection regime; these are descriptive grid averages. The exact index has zero loss in every cell. The three deviations are defined in the accompanying protocol.}\label{tab:staging-loss}",
           r"\begin{tabular}{llrrr}\toprule", r"cost law & collection & stop after one & lowest first & extra at stop \\ \midrule"]
    for family, regime in product(FAMILIES, COLLECTIONS):
        g = summary["groups"][family+"/"+regime]
        for policy in POLICIES:
            p = g["policies"][policy]
            values = [float(F(p[k])) for k in ("loss", "stop_loss", "advance_loss")]
            lines.append(f"| {FAMILY_LABELS[family]} | {LABELS[regime]} | {POLICY_LABELS[policy]} | "
                         f"{values[0]:.6f} | {values[1]:.6f} | {values[2]:.6f} | {p['positive_loss_cells']}/{g['cells']} |")
            csv_lines.append(f"{family},{regime},{policy},{g['cells']},{p['loss']},{p['stop_loss']},{p['advance_loss']},{p['positive_loss_cells']}")
        if family != "deterministic":
            values = [float(F(g["policies"][p]["loss"])) for p in POLICIES[1:]]
            tex.append(f"{FAMILY_LABELS[family]} & {regime} & {values[0]:.4f} & {values[1]:.4f} & {values[2]:.4f}"+r" \\")
    tex += [r"\bottomrule\end{tabular}\end{table}"]
    (output/"action_loss_table.tex").write_text("\n".join(tex)+"\n")
    (output/"action_loss_table.csv").write_text("\n".join(csv_lines)+"\n")
    lines += ["", "## What this strengthens", "",
              "The study adds a controlled explanation of signal value, adaptive abandonment, designated collection, "
              "and history-dependent costs. It also quantifies the stop-versus-advance loss accounting on specified deviations. "
              "It does not establish computational scaling, robustness to model misspecification, "
              "or performance on practical learning curves. The roughly seven-second local execution time is "
              "not a scaling result; every arm here has only three binary stages.", "",
              "## Figures", "",
              "`staging_signal_utility.pdf`: utility, change relative to uninformative signals, and terminal usage "
              "for constant costs at the protocol-specified slice k=.02; every terminal cost and accuracy is shown. "
              "`staging_initial_indices.pdf` uses the same slice for both cost laws. "
              "`staging_cost_grid.pdf` adds a c=.6 heatmap, and `staging_deterministic_controls.pdf` shows all "
              "deterministic cells. The c=.6 display slice and the illustrative table are for exposition; "
              "the complete-grid analysis and records are the evidence.", "",
              "## Reproduction", "", "From the project root:", "", "```sh",
              "python -m pytest tests/test_p2_staging_study.py tests/test_staged_pandora_exact.py -q",
              "python next_paper/p2_staging_study.py --out next_paper/p2_staging_study/rerun",
              "python next_paper/report_p2_staging_study.py --run next_paper/p2_staging_study/rerun",
              "```", "", "The runner refuses an existing output directory. Source and protocol hashes "
              "are frozen before evaluation, records are checkpointed cell by cell, and the report verifies the full grid and loss accounting. "
              "See `design_freeze.json`, `summary.json`, `audit.json`, and `artifact_manifest.json` for the run's design, sources and hashes."]
    (output/"FINDINGS.md").write_text("\n".join(lines)+"\n")
    return analysis


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args()
    rows, summary = audit(args.run)
    analysis = make_report(rows, summary, args.run)
    names = figures(rows, args.run)
    audit_result = dict(passed=True, cells=len(rows), exact_failures=0,
                        complete_declared_grid=True, frozen_sources_unchanged=True,
                        rational_policy_accounting=True, collection_expansion_monotone=True, figures=names,
                        reporter_sha256=sha(Path(__file__)),
                        records_sha256=sha(args.run/"records.jsonl"))
    (args.run/"audit.json").write_text(json.dumps(audit_result, indent=2)+"\n")
    snapshot = args.run/"source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for path in (HERE/"p2_staging_study.py", HERE/"staged_pandora_exact.py",
                 HERE/"P2_STAGING_STUDY_PROTOCOL.md", Path(__file__),
                 HERE.parent/"tests"/"test_p2_staging_study.py"):
        (snapshot/path.name).write_bytes(path.read_bytes())
    manifest = {str(p.relative_to(args.run)): sha(p) for p in sorted(args.run.rglob("*"))
                if p.is_file() and p.name != "artifact_manifest.json"}
    (args.run/"artifact_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(audit_result, indent=2))
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
