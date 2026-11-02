"""Analytic examples for staged reservation values and policy loss accounting."""
from fractions import Fraction as F

from next_paper.staged_pandora_exact import Node, Oracle, completion, nested, reservation


def terminal(y, cost):
    return Node(cost, True, ((F(1), y, None),))


def test_later_deterministic_collectable_reward_needs_no_information():
    arm = Node(F(1, 10), True, ((F(1), F(0), terminal(F(1), F(1, 10))),))
    oracle = Oracle()
    oracle.verify((arm,), F(0))
    assert nested(arm)[0] == F(4, 5)
    assert oracle.value((arm,), F(0)) == F(4, 5)
    assert oracle.policy((arm,), F(0), "index") == (F(4, 5), F(0), F(0))
    assert oracle.policy((arm,), F(0), "recomputed") == (F(0), F(4, 5), F(0))


def test_informative_inspection_changes_terminal_only_reservation_value():
    arm = Node(F(1, 100), False, (
        (F(1, 2), F(0), terminal(F(0), F(3, 5))),
        (F(1, 2), F(1), terminal(F(1), F(3, 5))),
    ))
    oracle = Oracle()
    oracle.verify((arm,), F(0))
    d, cost = completion(arm)
    assert cost == F(61, 100)
    assert reservation(d, cost) == -F(11, 100)
    assert nested(arm)[0] == F(19, 50)
    assert oracle.policy((arm,), F(0), "index")[0] == F(19, 100)
    assert oracle.policy((arm,), F(0), "committed") == (F(0), F(19, 100), F(0))


def test_committed_policy_pays_for_an_unprofitable_terminal_inspection():
    arm = Node(F(1, 100), False, (
        (F(1, 2), F(0), terminal(F(0), F(1, 10))),
        (F(1, 2), F(1), terminal(F(1), F(1, 10))),
    ))
    oracle = Oracle()
    oracle.verify((arm,), F(0))
    assert oracle.policy((arm,), F(0), "committed") == (F(39, 100), F(0), F(1, 20))
    assert oracle.policy((arm,), F(0), "recomputed") == (F(11, 25), F(0), F(0))


def test_history_dependent_cost_is_averaged_for_a_committed_completion():
    arm = Node(F(1, 100), False, (
        (F(1, 2), F(0), terminal(F(0), F(1, 5))),
        (F(1, 2), F(1), terminal(F(1), F(1, 10))),
    ))
    oracle = Oracle()
    oracle.verify((arm,), F(0))
    dist, cost = completion(arm)
    assert cost == F(4, 25)
    assert reservation(dist, cost) == F(17, 25)
    assert oracle.policy((arm,), F(0), "committed") == (F(17, 50), F(0), F(1, 10))
    assert oracle.policy((arm,), F(0), "index")[0] == F(11, 25)


def test_wrong_first_arm_has_an_exact_positive_advance_loss():
    a, b = terminal(F(4, 5), F(1, 10)), terminal(F(7, 10), F(1, 10))
    oracle = Oracle()
    oracle.verify((a, b), F(1, 2))
    assert oracle.value((a, b), F(1, 2)) == F(7, 10)
    assert oracle.action_gap((a, b), F(1, 2), 0) == 0
    assert oracle.action_gap((a, b), F(1, 2), 1) == F(1, 10)


def test_no_collectable_reward_has_minus_infinity_index():
    arm = Node(F(1, 10), False, ((F(1, 2), F(0), None), (F(1, 2), F(1), None)))
    oracle = Oracle()
    oracle.verify((arm,), F(3, 10))
    assert nested(arm)[0] is None
    assert oracle.value((arm,), F(3, 10)) == F(3, 10)
    assert oracle.action_gap((arm,), F(3, 10), 0) == F(1, 10)


def test_atom_at_reservation_boundary_allows_stopping_or_advancing():
    arm = Node(F(1, 8), True, (
        (F(1, 4), F(0), None),
        (F(1, 2), F(1, 2), None),
        (F(1, 4), F(1), None),
    ))
    oracle = Oracle()
    oracle.verify((arm,), F(1, 2))
    assert nested(arm)[0] == F(1, 2)
    assert oracle.value((arm,), F(1, 2)) == F(1, 2)
    assert oracle.action_gap((arm,), F(1, 2), 0) == 0
    assert oracle.policy((arm,), F(1, 2), "index") == (F(1, 2), F(0), F(0))


def test_continuous_dependent_readings_match_direct_stopping_problem():
    from math import sqrt
    from scipy.integrate import quad
    from scipy.optimize import brentq

    def integrate(fn, points):
        return quad(fn, 0, 1, points=sorted({p for p in points if 0 < p < 1}),
                    epsabs=1e-12, epsrel=1e-12)[0]

    # Direct single-arm Bellman gain: first collect U, then decide whether to
    # buy the known reward 1-U. This computation does not use covered calls.
    def forced_gain(r):
        def payoff(u):
            immediate = max(u-r, 0)
            continuation = max(1-u-max(r, u)-.25, 0)
            return immediate + continuation - .125
        return integrate(payoff, [r, .75-r, .375])

    root = brentq(forced_gain, 0, 1, xtol=1e-14)
    expected_root = (7-sqrt(7))/8
    assert abs(root-expected_root) < 1e-12
    assert abs(forced_gain(.5)-1/32) < 1e-12

    def index_policy_payoff(u):
        incumbent = max(.5, u)
        reward_after_cost = 1-u-.25 if .75-u > incumbent else incumbent
        return reward_after_cost - .125

    policy_value = integrate(index_policy_payoff, [.25, .5])
    covered_value = integrate(lambda u: max(.5, min(root, max(u, .75-u))),
                              [.25, .375, .5, root, .75-root])
    assert abs(policy_value-17/32) < 1e-12
    assert abs(covered_value-policy_value) < 1e-12
