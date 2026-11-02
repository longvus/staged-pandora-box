"""Analytic checks of the controlled known-law family and loss diagnostics."""
from fractions import Fraction as F
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"next_paper"))
from p2_staging_study import (COLLECTIONS, POLICIES, evaluate, make_case,
                              posterior, signal_arm)
from staged_pandora_exact import Node, Oracle, nested, reservation


def test_bayes_uses_both_signals_and_preserves_terminal_marginal():
    assert posterior(F(3, 4), (1, 0)) == F(1, 2)
    assert posterior(F(3, 4), (0, 0)) == F(1, 10)
    root, _ = signal_arm(F(3, 4), F(1, 100), F(1, 5), (3,))
    def expected_terminal(node):
        return sum(p*(y if child is None else expected_terminal(child)) for p, y, child in node.branches)
    assert expected_terminal(root) == F(1, 2)


def test_no_information_and_perfect_information_analytic_utilities():
    k, c = F(1, 50), F(3, 5)
    for a, expected in ((F(1, 2), F(0)), (F(1), (1-c-k)/2-k)):
        root, md = signal_arm(a, k, c, (3,))
        oracle = Oracle()
        oracle.verify((root,), F(0))
        assert evaluate((root,), md, oracle, "index").utility == expected
    root, md = signal_arm(F(1, 2), k, F(1, 5), (3,))
    assert Oracle().value((root,), F(0)) == F(1, 2)-2*k-F(1, 5)


def test_history_costs_are_known_before_the_advance():
    root, md = signal_arm(F(3, 4), F(1, 50), F(1, 5), (3,), history_cost=True)
    by_history = {history: node for node, (_, history) in md.items()}
    assert by_history[(0,)].cost == F(1, 50)
    assert by_history[(1,)].cost == F(3, 100)
    assert by_history[(1, 0)].cost == F(3, 10)
    assert by_history[(0, 0)].cost == F(1, 5)
    Oracle().verify((root,), F(0))


def test_classical_one_stage_reservation_recovery():
    node = Node(F(1, 10), True, ((F(1, 2), F(0), None), (F(1, 2), F(1), None)))
    assert nested(node)[0] == reservation(((F(0), F(1, 2)), (F(1), F(1, 2))), F(1, 10)) == F(4, 5)
    Oracle().verify((node,), F(0))


def test_intermediate_collection_can_be_best_without_information():
    arms, md = make_case("deterministic", None, F(1, 50), F(4, 5), COLLECTIONS["all"])
    oracle = Oracle()
    oracle.verify(arms, F(0))
    st = evaluate(arms, md, oracle, "index")
    assert st.utility == F(23, 50)
    assert st.stages == (F(1), F(1), F(0))


def test_diagnostic_losses_and_resource_accounting_across_collection_regimes():
    for regime in COLLECTIONS:
        arms, md = make_case("constant_cost", F(3, 4), F(1, 50), F(3, 5), COLLECTIONS[regime])
        oracle = Oracle()
        oracle.verify(arms, F(0))
        for policy in POLICIES:
            st = evaluate(arms, md, oracle, policy)
            assert oracle.value(arms, F(0))-st.utility == st.stop_loss+st.advance_loss >= 0
            assert st.stages[0] >= st.stages[1] >= st.stages[2] >= 0
            assert F(0) <= st.reward <= F(1)
            assert st.cost >= 0
        assert evaluate(arms, md, oracle, "index").stop_loss == 0
        assert evaluate(arms, md, oracle, "lowest_index_first").stop_loss == 0
        assert evaluate(arms, md, oracle, "one_extra_at_stop").stop_loss == 0
        assert evaluate(arms, md, oracle, "stop_after_one").advance_loss == 0
