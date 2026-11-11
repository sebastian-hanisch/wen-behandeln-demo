"""Die vorgerechnete Messreihe `precomputed_sweeps.json`: Aufbau, innere Widerspruchsfreiheit und Zugriffsfunktionen."""

import json

import pytest

import wbh_constants as C
import wbh_sweeps as W

DATA = W.data()


def test_meta_matches_the_constants():
    m = DATA["meta"]
    assert m["n_pop"] == C.N_POP and m["n_rep"] == C.N_REP and m["m_grid"] == list(C.M_GRID)
    assert m["budgets"] == list(C.SWEEP_BUDGETS) and m["deltas"] == list(C.DELTAS) and m["estimators"] == list(C.ESTIMATORS)
    assert m["leaf_by_m"] == {str(k): v for k, v in C.LEAF_BY_M.items()}      # die Messreihe stammt von der Blattgröße, die die App benutzt


def test_every_curve_has_one_band_per_pilot_size_with_ordered_bounds():
    for welt in C.WELTEN:
        for est in C.ESTIMATORS:
            for kind, keys in (("fluch", C.SWEEP_BUDGETS), ("faktor", C.SWEEP_BUDGETS), ("fang", C.SWEEP_BUDGETS), ("netdiff", C.DELTAS)):
                for key in keys:
                    rows = W.curve(welt, est, kind, key)
                    assert len(rows) == len(C.M_GRID)
                    for r in rows:
                        assert r is None or (r[1] <= r[0] <= r[2]), (welt, est, kind, key)


def test_undefined_cells_are_none_not_zero():
    assert W.curve("glatt", "eb", "fluch", 10)[0] is None and W.curve("glatt", "eb", "fang", 10)[0] is None     # Schrumpfung bei m = 1
    assert all(r is None for r in W.curve("homogen", "diff", "fang", 10))                                       # Null-Welt: Orakel = Zufall
    assert W.eb_collapse("glatt", 1) is None


def test_no_all_zero_column_in_the_heterogeneous_worlds():
    for welt in ("glatt", "wechselwirkung", "falle"):
        for est in C.ESTIMATORS:
            vals = [r[0] for r in W.curve(welt, est, "fang", 30) if r is not None]
            assert len(set(round(v, 6) for v in vals)) > 3 and any(abs(v) > 1 for v in vals), (welt, est)


def test_kipp_records_are_consistent():
    for welt in C.WELTEN:
        for est in C.ESTIMATORS:
            for d in C.DELTAS:
                k = W.kipp(welt, est, d)
                assert k["min"] <= k["median"] <= k["max"] and k["min"] in C.M_GRID + (128,) and k["max"] in C.M_GRID + (128,)
                assert 0 <= k["n_bis_64"] <= C.N_POP
                assert (k["n_bis_64"] == 0) == (k["min"] == 128)


def test_kipp_text():
    assert W.kipp_text({"median": 1.5, "min": 1, "max": 4, "n_bis_64": 20}) == "zwischen m = 1 und m = 2 (Spanne m = 1 bis m = 4)"
    assert W.kipp_text({"median": 16.0, "min": 8, "max": 32, "n_bis_64": 20}) == "m = 16 (Spanne m = 8 bis m = 32)"
    assert W.kipp_text({"median": 1.0, "min": 1, "max": 1, "n_bis_64": 20}) == "m = 1"
    assert W.kipp_text({"median": 64.0, "min": 32, "max": 128, "n_bis_64": 17}) == "m = 64 (Spanne m = 32 bis über m = 64)"
    assert W.kipp_text({"median": 6.5, "min": 4, "max": 8, "n_bis_64": 20}) == "zwischen m = 4 und m = 8 (Spanne m = 4 bis m = 8)"


def test_rank_at_is_sorted_and_skips_undefined_estimators():
    rows = W.rank_at("glatt", 1, 10)
    assert [r[0] for r in rows] != [] and "eb" not in [r[0] for r in rows]
    assert all(a[1] >= b[1] for a, b in zip(rows, rows[1:]))
    assert len(W.rank_at("glatt", 8, 10)) == len(C.ESTIMATORS)


def test_pair_counts_are_within_the_number_of_populations():
    for welt in ("glatt", "wechselwirkung", "falle"):
        for budget in C.BUDGETS:
            for m in C.M_GRID:
                med, vorn, n = W.pair(welt, "twald", "tlin", budget, m)
                assert n == C.N_POP and 0 <= vorn <= n


def test_json_is_plain_data_without_nan_literals():
    text = W.PATH.read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text)["meta"]["n_pop"] == C.N_POP


@pytest.mark.parametrize("welt", C.WELTEN)
def test_eb_collapse_rates_are_shares_and_vanish_for_large_m_except_in_the_null_world(welt):
    r = DATA["eb_kollaps"][welt]
    assert r[0] is None and all(0.0 <= x <= 1.0 for x in r[1:])
    if welt != "homogen":
        assert r[-1] == 0.0 and r[1] > r[3] > r[5] - 1e-12
