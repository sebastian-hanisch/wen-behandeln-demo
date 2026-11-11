"""Wertmaße an handgesetzten Zahlen, Referenzgrößen, Fang-Anteil, Messzelle und Live-Auswertung."""

import numpy as np
import pytest

import wbh_constants as C
import wbh_evaluation as V
from wbh_stream import Stream

TAU = np.arange(10.0, 0.0, -1.0)                                      # 10, 9, ..., 1
TH = np.array([9.5, 10.5, 8.5, 7.5, 6.5, 5.5, 4.5, 3.5, 2.5, 1.5])    # Rangfolge nach der Schätzung: Index 1, 0, 2, 3, ...


def test_k_of_rounds_the_budget_share_to_whole_sites_with_at_least_one():
    assert [V.k_of(b, 100) for b in (10, 30, 50, 80)] == [10, 30, 50, 80]
    assert V.k_of(10, 10) == 1 and V.k_of(30, 10) == 3 and V.k_of(10, 5) == 1


def test_evaluate_by_hand():
    out = V.evaluate(TAU, TH, Stream(1))
    # Rangfolge: Index 1, 0, 2, 3, 4, 5, 6, 7 (Schätzungen 10,5 9,5 8,5 7,5 6,5 5,5 4,5 3,5)
    assert out[V.MI["v10"]] == 9.0 and out[V.MI["naiv10"]] == 10.5                      # k = 1
    assert out[V.MI["v30"]] == pytest.approx(9.0) and out[V.MI["naiv30"]] == pytest.approx(9.5)   # k = 3: wahr 9, 10, 8 / geschätzt 10,5 9,5 8,5
    assert out[V.MI["v50"]] == pytest.approx(8.0) and out[V.MI["naiv50"]] == pytest.approx(8.5)   # k = 5: wahr 9 10 8 7 6
    assert out[V.MI["v80"]] == pytest.approx(6.5) and out[V.MI["naiv80"]] == pytest.approx(7.0)   # k = 8
    # Schwellenregel: Mittel der wahren Effekte 5,5; c = 5,5 + delta; behandelt wird, wo die Schätzung über c liegt
    assert out[V.MI["net1"]] == pytest.approx(0.8)      # c = 6,5: Index 0..3 (Schätzung 6,5 nicht über 6,5): (3,5 + 2,5 + 1,5 + 0,5) / 10
    assert out[V.MI["net0"]] == pytest.approx(1.25)     # c = 5,5: Index 0..4: (4,5 + 3,5 + 2,5 + 1,5 + 0,5) / 10
    assert out[V.MI["net-1"]] == pytest.approx(1.8)     # c = 4,5: Index 0..5: (5,5 + 4,5 + 3,5 + 2,5 + 1,5 + 0,5) / 10


def test_overestimation_is_estimate_minus_truth_of_the_chosen():
    out = V.evaluate(TAU, TH, Stream(1))
    assert out[V.MI["naiv30"]] - out[V.MI["v30"]] == pytest.approx(0.5)


def test_reference_by_hand():
    ref = V.reference(TAU)
    assert ref["tbar"] == 5.5
    assert ref["orakel"] == {10: 10.0, 30: 9.0, 50: 8.0, 80: 6.5}
    assert ref["alle"] == {-1: 1.0, 0: 0.0, 1: -1.0}
    assert ref["beste_feste"] == {-1: 1.0, 0: 0.0, 1: 0.0}           # delta = +1: "keinen" ist die beste feste Regel, delta = -1: "alle"


def test_capture_share():
    assert V.capture_share(8.0, 5.5, 8.0) == 1.0
    assert V.capture_share(5.5, 5.5, 8.0) == 0.0
    assert V.capture_share(7.0, 5.0, 9.0) == pytest.approx(0.5)
    assert V.capture_share(4.0, 5.0, 9.0) == pytest.approx(-0.25)    # schlechter als Zufall: negativ
    assert np.isnan(V.capture_share(3.4, 3.4, 3.4))                    # Null-Welt: das Orakel liegt nicht über dem Zufall


def test_run_cell_returns_none_for_the_undefined_shrinkage_and_a_vector_otherwise():
    import wbh_pools as P
    pop = P.make_population("glatt", 1)
    res, infos = V.run_cell(pop, 1, C.ESTIMATORS, (1, 2, 3))
    assert res["eb"] is None and infos["eb"]["defined"] is False
    assert all(res[e].shape == (len(V.METRICS),) for e in C.ESTIMATORS if e != "eb")
    res2, _ = V.run_cell(pop, 2, C.ESTIMATORS, (1, 2, 3))
    assert res2["eb"] is not None


def test_branch_test_every_estimator_differs_from_its_neighbour_on_at_least_one_pilot():
    """Keine Nullspalte, kein toter Zweig (DEMO-PLAYBOOK §0): jede Variante weicht von der Nachbarvariante auf mindestens einem Piloten ab."""
    import wbh_pools as P
    pop = P.make_population("wechselwirkung", 2)
    cols = {e: [] for e in C.ESTIMATORS}
    for r in range(25):
        res, _ = V.run_cell(pop, 4, C.ESTIMATORS, (9, r))
        for e in C.ESTIMATORS:
            cols[e].append(res[e][V.MI["v30"]])
    for a, b in zip(C.ESTIMATORS[:-1], C.ESTIMATORS[1:]):
        assert not np.array_equal(cols[a], cols[b]), (a, b)
    for e in C.ESTIMATORS:
        assert np.all(np.isfinite(cols[e])) and np.ptp(cols[e]) > 0, e


# ---------------------------------------------------------------------------------------------------------------------------------------
# Live-Auswertung
# ---------------------------------------------------------------------------------------------------------------------------------------


def test_analysis_fields_are_consistent_with_the_population_and_the_pilot():
    a = V.analyse(V.Settings(0, 4, 30, 1, 0, 35))
    assert a.k == 30 and len(a.chosen) == 30 and len(set(a.chosen)) == 30
    assert a.true_mean == pytest.approx(a.pop.tau[a.chosen].mean())
    assert a.est_mean == pytest.approx(a.th[a.chosen].mean())
    assert a.tbar == pytest.approx(a.pop.tau.mean())
    assert a.oracle_mean == pytest.approx(np.sort(a.pop.tau)[::-1][:30].mean()) and a.oracle_mean >= a.true_mean
    assert a.n_treated == int((a.th > a.tbar + 1).sum())
    assert a.reps["n"] == C.LIVE_REPS and a.reps["over"] == pytest.approx(a.reps["est_mean"] - a.reps["true_mean"])


def test_analysis_is_deterministic():
    s = V.Settings(1, 8, 10, -1, 2, 77)
    a, b = V.analyse(s), V.analyse(s)
    assert np.array_equal(a.th, b.th) and a.reps == b.reps and a.net == b.net


def test_the_shrinkage_is_undefined_at_m_one_and_the_analysis_says_so():
    a = V.analyse(V.Settings(0, 1, 10, 1, 1, 35))
    assert a.defined is False and a.reps == {"n": 0} and len(a.chosen) == 0 and np.isnan(a.est_mean)
    assert "diff" in a.bridge and "eb" not in a.bridge


def test_every_control_changes_the_result_no_dead_controls():
    base = V.Settings(0, 8, 30, 1, 2, 35)
    a0 = V.analyse(base)
    changed = {
        "welt": V.Settings(3, 8, 30, 1, 2, 35),
        "m": V.Settings(0, 16, 30, 1, 2, 35),
        "budget": V.Settings(0, 8, 50, 1, 2, 35),
        "delta": V.Settings(0, 8, 30, -1, 2, 35),
        "est": V.Settings(0, 8, 30, 1, 0, 35),
        "seed": V.Settings(0, 8, 30, 1, 2, 36),
    }
    for name, s in changed.items():
        a = V.analyse(s)
        sig0 = (a0.est_mean, a0.true_mean, a0.net, a0.n_treated, a0.k, a0.reps["over"], a0.reps["net_diff"])
        sig = (a.est_mean, a.true_mean, a.net, a.n_treated, a.k, a.reps["over"], a.reps["net_diff"])
        assert sig != sig0, name


def test_collapsed_shrinkage_is_flagged_and_chooses_at_random():
    """In der Null-Welt ist die Prior-Varianz bei m = 2 in etwa der Hälfte der Piloten null: das Kennzeichen ist gesetzt, alle Schätzungen sind gleich."""
    flags = []
    for seed in range(20):
        a = V.analyse(V.Settings(2, 2, 10, 1, 1, seed))
        flags.append(a.collapsed)
        if a.collapsed:
            assert np.all(a.th == a.th[0])
    assert 4 <= sum(flags) <= 16
