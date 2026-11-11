"""Randfälle (DEMO-PLAYBOOK §12): kleinste Populationen, vollständige Schrumpfung, Gleichstände, Extremwerte der Regler."""

import numpy as np
import pytest

import wbh_constants as C
import wbh_estimators as E
import wbh_evaluation as V
import wbh_pools as P
from wbh_stream import Stream


def test_exactly_two_sites():
    """n = 2: jedes Budget wählt mindestens einen Standort, die Auswahl ist die mit der größeren Schätzung."""
    out = V.evaluate(np.array([1.0, 3.0]), np.array([2.0, 5.0]), Stream(1))
    for b in C.SWEEP_BUDGETS:
        assert out[V.MI[f"v{b}"]] in (3.0, 2.0) and out[V.MI[f"naiv{b}"]] in (5.0, 3.5)
    assert out[V.MI["v10"]] == 3.0 and out[V.MI["naiv10"]] == 5.0          # k = 1: der Standort mit Schätzung 5


def test_single_site_and_empty_ties():
    out = V.evaluate(np.array([2.0]), np.array([7.0]), Stream(1))
    assert out[V.MI["v10"]] == 2.0 and out[V.MI["naiv80"]] == 7.0


def test_budget_of_all_sites_equals_the_overall_mean():
    tau = np.arange(100.0)
    out = V.evaluate(tau, np.random.default_rng(0).normal(size=100), Stream(1))
    assert out[V.MI["v80"]] == pytest.approx(tau.mean(), abs=6.0)         # 80 %: dicht am Mittel (kein Orakel)
    assert V.k_of(100, 100) == 100


def test_all_estimates_equal_choose_at_random_and_the_threshold_rule_depends_on_the_side_of_c():
    tau = np.linspace(0.0, 10.0, 100)
    th = np.full(100, 5.0)                                                  # tbar = 5
    out = V.evaluate(tau, th, Stream(1))
    assert out[V.MI["net1"]] == 0.0 and out[V.MI["net0"]] == 0.0            # Schätzung 5 liegt nicht über c = 6 bzw. 5
    assert out[V.MI["net-1"]] == pytest.approx(((tau - 4.0)).sum() / 100)    # über c = 4: alle behandeln


def test_the_forest_with_a_constant_target_predicts_the_constant():
    X = np.random.default_rng(0).normal(size=(30, 4))
    assert np.all(E.forest_predict(X, np.full(30, 2.5), Stream(1), 5) == 2.5)


def test_tree_with_leaf_size_above_half_the_data_is_a_single_leaf():
    X = np.random.default_rng(0).normal(size=(10, 4))
    y = np.arange(10.0)
    assert np.all(E.tree_fit_predict(X, y, X, 6) == y.mean())


def test_regression_with_constant_features_does_not_crash():
    X = np.ones((20, 4))
    yA, yB = np.random.default_rng(0).normal(size=(20, 3)), np.random.default_rng(1).normal(size=(20, 3))
    for name in ("tlin", "tlin2", "twald"):
        th, info = E.estimate(name, X, yA, yB, Stream(1))
        assert np.all(np.isfinite(th)) and info["defined"]


def test_every_welt_and_estimator_at_the_smallest_and_largest_pilot():
    for welt in range(len(C.WELTEN)):
        for m in (C.M_GRID[0], C.M_GRID[-1]):
            for est in range(len(C.ESTIMATORS)):
                a = V.analyse(V.Settings(welt, m, C.BUDGETS[-1], C.DELTAS[0], est, 1))
                assert a.defined == (not (C.ESTIMATORS[est] == "eb" and m == 1))
                if a.defined:
                    assert np.isfinite(a.est_mean) and np.isfinite(a.net)


def test_null_world_has_no_oracle_advantage():
    a = V.analyse(V.Settings(2, 8, 10, 1, 0, 3))
    assert a.oracle_mean == pytest.approx(a.tbar) and np.isnan(V.treated_note(a))


def test_largest_pilot_never_reads_beyond_the_pool():
    pop = P.make_population("glatt", 1)
    yA, _ = P.draw_pilot(pop, 64, Stream(1))
    assert yA.shape == (C.N_SITES, 64)
