"""Befunde: jede Zahl der README steht hier (Zahl -> Test, siehe README „Befunde“).

Zwei Arten von Prüfung:
  * **Neu gemessen** (`test_remeasure_*`): Welten glatt und Falle, Schätzer Rohdifferenz und T-linear, alle 20 Populationen x 7 Pilotumfänge x 50 Piloten
    mit den Seeds der Messreihe; die Kurven und Kipppunkte stimmen mit `precomputed_sweeps.json` überein (Toleranz 1e-3, die Datei ist auf 4 Stellen gerundet).
  * **Gegen die Datei** (`test_readme_*`): Zahlen der Schrumpfung, des Waldes, der Kopplung und aller Welten stehen als Messung der Datei, nicht neu gemessen;
    hier ist geprüft, dass der Text der README genau diese Werte nennt und dass die gezählten Aussagen („in 20 von 20 Populationen“) in der Datei stimmen.
Wald-Zahlen (T-Wald) sind nur gegen die Datei geprüft, nicht neu gemessen."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

import wbh_constants as C
import wbh_evaluation as V
import wbh_pools as P
import wbh_sweeps as W

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import messreihe as MR  # noqa: E402

README = (ROOT / "README.md").read_text(encoding="utf-8")


def fm(welt, est, kind, key, m, d=1):
    """'Median [Minimum bis Maximum]' einer Kurve beim Pilotumfang m, Dezimalkomma."""
    r = W.curve(welt, est, kind, key)[C.M_GRID.index(m)]
    f = lambda x: f"{x:.{d}f}".replace(".", ",")
    return f"{f(r[0])} [{f(r[1])} bis {f(r[2])}]"


def med(welt, est, kind, key, m):
    return W.curve(welt, est, kind, key)[C.M_GRID.index(m)][0]


# ---------------------------------------------------------------------------------------------------------------------------------------
# Neu gemessen
# ---------------------------------------------------------------------------------------------------------------------------------------


def _remeasure(welt, ests):
    out = {e: {"fluch": [], "faktor": [], "fang30": [], "kipp": {d: [] for d in C.DELTAS}, "net": {d: [] for d in C.DELTAS}} for e in ests}
    for k in range(C.N_POP):
        pop = P.make_population(welt, MR.POP_SEED(welt, k))
        ref = V.reference(pop.tau)
        arr = np.full((len(C.M_GRID), C.N_REP, len(ests), len(V.METRICS)), np.nan)
        for mi, m in enumerate(C.M_GRID):
            for r in range(C.N_REP):
                res, _ = V.run_cell(pop, m, ests, MR.REP_PARTS(welt, m, k, r))
                for ei, e in enumerate(ests):
                    arr[mi, r, ei] = res[e]
        for ei, e in enumerate(ests):
            v, naiv = arr[:, :, ei, V.MI["v10"]], arr[:, :, ei, V.MI["naiv10"]]
            out[e]["fluch"].append((naiv - v).mean(axis=1))
            out[e]["faktor"].append(naiv.mean(axis=1) / v.mean(axis=1))
            v30 = arr[:, :, ei, V.MI["v30"]]
            out[e]["fang30"].append(100 * ((v30 - ref["tbar"]) / (ref["orakel"][30] - ref["tbar"])).mean(axis=1))
            for d in C.DELTAS:
                diff = arr[:, :, ei, V.MI[f"net{d}"]] - ref["beste_feste"][d]
                out[e]["net"][d].append(diff.mean(axis=1))
                out[e]["kipp"][d].append(V.kipp_point(diff))
    return out


@pytest.mark.parametrize("welt", ["glatt", "falle", "wechselwirkung"])
def test_remeasure_rohdifferenz_and_t_linear_reproduce_the_stored_curves_and_kipppunkte(welt):
    res = _remeasure(welt, ("diff", "tlin"))
    for e in ("diff", "tlin"):
        for kind, key, store in (("fluch", 10, "fluch"), ("faktor", 10, "faktor"), ("fang", 30, "fang30")):
            got = np.median(np.array(res[e][store]), axis=0)
            want = [r[0] for r in W.curve(welt, e, kind, key)]
            assert got == pytest.approx(want, abs=1e-3), (welt, e, kind)
        for d in C.DELTAS:
            got = np.median(np.array(res[e]["net"][d]), axis=0)
            assert got == pytest.approx([r[0] for r in W.curve(welt, e, "netdiff", d)], abs=1e-3), (welt, e, d)
            ks = np.array(res[e]["kipp"][d])
            k = W.kipp(welt, e, d)
            assert (float(np.median(ks)), int(ks.min()), int(ks.max()), int((ks <= 64).sum())) == (k["median"], k["min"], k["max"], k["n_bis_64"]), (welt, e, d)
    if welt == "wechselwirkung":                                               # der Median „m = 6“ ist kein gemessener Pilotumfang: 10 Populationen bei 4, 10 bei 8
        ks = res["diff"]["kipp"][1]
        assert sorted(ks) == [4] * 10 + [8] * 10


def test_remeasure_the_correlation_of_effect_and_dispersion_in_every_heterogeneous_world():
    for welt in ("glatt", "wechselwirkung", "falle"):
        rho = np.array([P.make_population(welt, MR.POP_SEED(welt, k)).rho for k in range(C.N_POP)])
        band = W.rho(welt)[0]
        assert [round(float(np.median(rho)), 4), round(float(rho.min()), 4), round(float(rho.max()), 4)] == pytest.approx(band, abs=1e-4), welt
        assert int((np.abs(rho) >= C.RHO_LIMIT).sum()) == W.rho(welt)[1], welt


# ---------------------------------------------------------------------------------------------------------------------------------------
# Gewinner-Fluch
# ---------------------------------------------------------------------------------------------------------------------------------------


def test_winners_curse_factor_of_the_ten_chosen_with_the_raw_difference():
    assert fm("glatt", "diff", "faktor", 10, 1) == "4,9 [4,3 bis 5,7]"
    assert fm("glatt", "diff", "faktor", 10, 8) == "2,0 [1,8 bis 2,2]"
    assert fm("glatt", "diff", "faktor", 10, 64) == "1,2 [1,1 bis 1,2]"
    assert fm("falle", "diff", "faktor", 10, 1) == "4,4 [4,3 bis 4,6]"
    assert fm("falle", "diff", "faktor", 10, 8) == "2,1 [2,0 bis 2,2]"
    assert fm("falle", "diff", "faktor", 10, 64) == "1,3 [1,3 bis 1,4]"
    assert fm("wechselwirkung", "diff", "faktor", 10, 1) == "7,5 [5,4 bis 9,1]"


def test_winners_curse_of_the_raw_difference_at_m_2_in_percentage_points():
    vals = [med(w, "diff", "fluch", 10, 2) for w in C.WELTEN]
    assert round(min(vals), 1) == 8.1 and round(max(vals), 1) == 9.5


def test_the_true_effect_has_a_standard_error_of_0_24_at_7_days_and_0_05_at_224_days():
    """Der wahre Effekt ist die Differenz zweier Poolmittel (Regel B mit z = 1,4 minus Regel A): Standardfehler aus beiden Pool-Varianzen."""
    b = P.bank()

    def se(L):
        va, vb = (b.var[b.index[P.key_name(P.key_std(z, L))]] for z in (C.Z_A, C.Z_B))
        return float(np.sqrt((va + vb) / C.POOL_SIZE))
    assert round(se(7), 2) == 0.24 and round(se(224), 2) == 0.05


def test_winners_curse_in_percentage_points_and_in_the_null_world():
    assert fm("glatt", "diff", "fluch", 10, 1) == "12,7 [11,3 bis 14,1]"
    assert fm("glatt", "diff", "fluch", 10, 8) == "3,8 [3,4 bis 4,3]"
    assert fm("glatt", "diff", "fluch", 10, 64) == "0,9 [0,7 bis 1,0]"
    assert fm("homogen", "diff", "fluch", 10, 1) == "11,7 [11,3 bis 12,2]"
    for k in range(C.N_POP):                                                   # Null-Welt: alle Standorte haben denselben wahren Effekt (3,4)
        tau = P.make_population("homogen", MR.POP_SEED("homogen", k)).tau
        assert np.all(tau == tau[0]) and round(float(tau[0]), 1) == 3.4


def test_the_winners_curse_shrinks_monotonically_with_the_pilot_size_for_the_raw_difference_in_every_world():
    for welt in C.WELTEN:
        v = [r[0] for r in W.curve(welt, "diff", "fluch", 10)]
        assert all(a > b for a, b in zip(v, v[1:])), welt


def test_the_shrinkage_keeps_the_median_overestimation_within_plus_minus_0_6_for_m_from_2():
    vals = [r[0] for w in C.WELTEN for r in W.curve(w, "eb", "fluch", 10)[1:]]
    assert max(vals) <= 0.6 + 0.05 and min(vals) >= -0.6 - 0.05
    assert round(max(vals), 1) == 0.6 and round(min(vals), 1) == -0.6          # Rand: Falle/Null-Welt bei m = 2 (+0,6), Wechselwirkung bei m = 8 (-0,6)
    assert med("wechselwirkung", "eb", "fluch", 10, 8) < 0                      # die Schrumpfung schießt in der Wechselwirkung über das Ziel


# ---------------------------------------------------------------------------------------------------------------------------------------
# Kipppunkt
# ---------------------------------------------------------------------------------------------------------------------------------------


def test_kipppunkt_of_the_raw_difference_at_threshold_plus_one():
    assert W.kipp_text(W.kipp("glatt", "diff", 1)) == "m = 16 (Spanne m = 8 bis m = 32)"
    assert W.kipp_text(W.kipp("wechselwirkung", "diff", 1)) == "zwischen m = 4 und m = 8 (Spanne m = 4 bis m = 8)"     # Median aus 10 x m = 4 und 10 x m = 8
    assert W.kipp_text(W.kipp("falle", "diff", 1)) == "m = 64 (Spanne m = 32 bis über m = 64)" and W.kipp("falle", "diff", 1)["n_bis_64"] == 17
    assert W.kipp_text(W.kipp("glatt", "diff", -1)) == "m = 8 (Spanne m = 4 bis m = 16)"
    assert W.kipp_text(W.kipp("falle", "diff", -1)) == "m = 64 (Spanne m = 64 bis über m = 64)" and W.kipp("falle", "diff", -1)["n_bis_64"] == 14


def test_threshold_zero_favours_the_raw_difference_from_m_1_and_threshold_plus_one_does_not():
    for w, up, down in (("glatt", 0.21, -0.22), ("wechselwirkung", 0.24, -0.17), ("falle", 0.10, -0.25)):
        assert round(W.curve(w, "diff", "netdiff", 0)[0][0], 2) == up, w
        assert W.kipp(w, "diff", 0) == {"median": 1.0, "min": 1, "max": 1, "n_bis_64": C.N_POP}, w
        assert round(W.curve(w, "diff", "netdiff", 1)[0][0], 2) == down, w
    for w in ("glatt", "wechselwirkung"):                                    # „kippt in glatt und Wechselwirkung schon bei m = 1“ (T-linear, Median)
        assert W.kipp(w, "tlin", 1)["median"] == 1.0


def test_kipppunkt_of_t_linear_is_at_most_16_in_all_sixty_populations_of_the_three_heterogeneous_worlds():
    assert W.kipp_text(W.kipp("glatt", "tlin", 1)) == "m = 1 (Spanne m = 1 bis m = 2)"
    assert W.kipp_text(W.kipp("wechselwirkung", "tlin", 1)) == "m = 1"
    assert W.kipp_text(W.kipp("falle", "tlin", 1)) == "m = 8 (Spanne m = 4 bis m = 16)"
    for d in (-1, 1):
        for w in ("glatt", "wechselwirkung", "falle"):
            k = W.kipp(w, "tlin", d)
            assert k["max"] <= 16 and k["n_bis_64"] == C.N_POP, (w, d)


def test_t_linear_tips_earlier_than_the_raw_difference_at_the_thresholds_plus_minus_one_in_the_three_heterogeneous_worlds():
    for w in ("glatt", "wechselwirkung", "falle"):
        for d in (-1, 1):
            assert W.kipp(w, "tlin", d)["median"] < W.kipp(w, "diff", d)["median"], (w, d)


def test_in_the_null_world_no_threshold_rule_beats_the_best_fixed_rule_in_any_population():
    for e in C.ESTIMATORS:
        for d in C.DELTAS:
            k = W.kipp("homogen", e, d)
            assert k["n_bis_64"] == 0 and k["min"] == 128, (e, d)


# ---------------------------------------------------------------------------------------------------------------------------------------
# Schätzer (bedingter Befund) und Schrumpfung
# ---------------------------------------------------------------------------------------------------------------------------------------


def test_capture_share_by_estimator_in_the_smooth_world_at_30_percent_and_m_8():
    assert fm("glatt", "tlin", "fang", 30, 8) == "95,6 [93,5 bis 97,4]"
    assert fm("glatt", "twald", "fang", 30, 8) == "87,1 [83,3 bis 91,4]"
    assert fm("glatt", "diff", "fang", 30, 8) == "59,7 [49,6 bis 65,2]"
    assert fm("glatt", "eb", "fang", 30, 8) == "67,5 [56,6 bis 75,5]"
    assert W.pair("glatt", "twald", "tlin", 30, 8)[1] == 0                      # T-Wald in keiner der 20 Populationen vor T-linear


def test_in_the_interaction_world_the_forest_wins_at_30_percent_budget_and_the_line_plateaus():
    assert fm("wechselwirkung", "tlin", "fang", 30, 64) == "93,1 [89,7 bis 96,0]"
    assert fm("wechselwirkung", "twald", "fang", 30, 64) == "98,5 [97,7 bis 99,8]"
    for m in (4, 8, 16, 32, 64):
        assert W.pair("wechselwirkung", "twald", "tlin", 30, m)[1] >= 18, m
    assert [W.pair("wechselwirkung", "twald", "tlin", 30, m)[1] for m in (8, 16, 32, 64)] == [20, 20, 20, 20]


def test_the_ranking_of_forest_and_line_in_the_interaction_world_depends_on_the_budget():
    """Korrektur gegenüber der Vormessung (nur 30 % Budget gemessen): bei 10 % liegt T-linear bis m = 16 im Median vor T-Wald."""
    for m in (1, 2, 4, 8, 16):
        assert W.pair("wechselwirkung", "twald", "tlin", 10, m)[0] < 0, m
    assert abs(W.pair("wechselwirkung", "twald", "tlin", 10, 32)[0]) < 0.05 and W.pair("wechselwirkung", "twald", "tlin", 10, 64)[0] > 0     # 10 %: gleichauf bei m = 32, vorn bei m = 64
    assert W.pair("wechselwirkung", "twald", "tlin", 30, 2)[0] < 0 < W.pair("wechselwirkung", "twald", "tlin", 30, 4)[0]                 # 30 %: ab m = 4
    med, vorn, n = W.pair("wechselwirkung", "twald", "tlin2", 30, 8)                                                                     # gegen die Gerade mit Produkten klein
    assert round(med, 1) == 2.1 and vorn == 17 and n == 20
    assert W.pair("wechselwirkung", "twald", "tlin", 10, 64)[1] == 16
    assert W.pair("wechselwirkung", "twald", "tlin", 50, 16)[1] == 15 and W.pair("wechselwirkung", "twald", "tlin", 50, 64)[1] == 20
    assert W.pair("wechselwirkung", "twald", "tlin", 50, 8)[0] < 0 < W.pair("wechselwirkung", "twald", "tlin", 50, 16)[0]


def test_shrinkage_beats_the_raw_difference_from_m_8_without_coupling_and_never_in_the_trap():
    assert W.pair("glatt", "eb", "diff", 30, 8)[1] == 20 and W.pair("glatt", "eb", "diff", 30, 8)[0] == pytest.approx(8.4, abs=0.05)
    assert W.pair("wechselwirkung", "eb", "diff", 30, 8)[1] == 19
    for m in C.M_GRID[1:]:
        assert W.pair("falle", "eb", "diff", 30, m)[1] == 0, m
    assert W.pair("falle", "eb", "diff", 30, 8)[0] == pytest.approx(-17.3, abs=0.05)
    assert W.pair("falle", "eb", "diff", 10, 8)[0] == pytest.approx(-29.5, abs=0.05)


def test_the_trap_reverses_at_50_percent_budget_for_m_from_16():
    assert [W.pair("falle", "eb", "diff", 50, m)[1] for m in (16, 32, 64)] == [16, 20, 20]
    assert all(W.pair("falle", "eb", "diff", 50, m)[0] > 0 for m in (16, 32, 64))
    for bu in (10, 30):
        assert all(W.pair("falle", "eb", "diff", bu, m)[1] == 0 for m in C.M_GRID[1:])


def test_t_linear_lowers_the_winners_curse_at_m_8_in_the_smooth_world():
    assert fm("glatt", "tlin", "fluch", 10, 8) == "0,6 [0,5 bis 0,8]"
    for w in C.WELTEN:                                                          # die Rohdifferenz überschätzt bei jedem Budget und Pilotumfang
        for bu in C.BUDGETS:
            assert all(r[0] > 0 for r in W.curve(w, "diff", "fluch", bu)), (w, bu)


def test_shrinkage_collapse_rate_of_the_prior_variance():
    r = lambda w, m: round(100 * W.eb_collapse(w, m))
    assert [r(w, 4) for w in ("wechselwirkung", "glatt", "falle")] == [13, 15, 35]
    assert [r("wechselwirkung", 2), r("glatt", 2), r("falle", 2), r("homogen", 2)] == [31, 33, 43, 53]
    assert [r(w, 16) for w in ("glatt", "wechselwirkung", "falle")] == [0, 0, 6]
    assert 0.5 < min(W.eb_collapse("homogen", m) for m in C.M_GRID[1:]) < 0.6   # Null-Welt: in etwa der Hälfte der Piloten, für jedes m


# ---------------------------------------------------------------------------------------------------------------------------------------
# Kopplung von Effekt und Streuung
# ---------------------------------------------------------------------------------------------------------------------------------------


def test_effect_and_dispersion_are_weakly_negatively_coupled_in_all_forty_populations_of_glatt_and_wechselwirkung():
    for w in ("glatt", "wechselwirkung"):
        assert W.rho(w)[0][2] < 0, w                                         # Maximum der 20 Korrelationen ist negativ
        assert W.rho(w)[0][0] < -0.15


def test_coupling_of_effect_and_dispersion():
    assert W.rho("falle") == ([0.9272, 0.9109, 0.9542], 20)
    assert W.rho("glatt") == ([-0.2336, -0.4361, -0.0468], 6)
    assert W.rho("wechselwirkung") == ([-0.1652, -0.3364, -0.0464], 2)


def test_budget_of_80_percent_is_not_offered_because_the_raw_difference_is_worse_than_random_in_one_world():
    assert 80 in C.SWEEP_BUDGETS and 80 not in C.BUDGETS
    assert fm("falle", "diff", "fang", 80, 1) == "-16,4 [-25,1 bis -12,6]"
    assert all(r[0] > 0 for r in W.curve("glatt", "diff", "fang", 80)) and all(r[0] > 0 for r in W.curve("wechselwirkung", "diff", "fang", 80))


def test_construction_numbers_named_in_the_readme():
    import json
    assert len(P.all_keys()) == 342 and C.N_POP * len(C.WELTEN) * len(C.M_GRID) * C.N_REP == 28000
    assert C.LEAF_BY_M == {1: 10, 2: 10, 4: 10, 8: 10, 16: 10, 32: 10, 64: 5}
    tuned = json.loads((ROOT / "tools" / "tuning.json").read_text(encoding="utf-8"))["LEAF_BY_M"]
    assert {int(k): v for k, v in tuned.items()} == C.LEAF_BY_M                      # die Konstante ist das Ergebnis der Abstimmung
    assert sum(1 for v in C.LEAF_BY_M.values() if v in (min(C.LEAF_GRID), max(C.LEAF_GRID))) == 6      # 6 von 7 am Rand des Gitters
    assert (C.N_TREES, C.EB_PRIOR_DF, C.LIVE_REPS, C.LIVE_REPS_WALD) == (40, 4.0, 40, 10)
    mean_effect = {w: W.data()["welt"][w]["tau_mittel"][0] for w in ("glatt", "wechselwirkung")}
    assert mean_effect["wechselwirkung"] < mean_effect["glatt"]                        # „kleinerer mittlerer Effekt“ der Welt Wechselwirkung


def test_shrinkage_at_m_2_collapsed_pilots_have_no_overestimation_and_the_others_0_7_to_1_2_points():
    """Neu gemessen: Schrumpfung bei m = 2, alle vier Welten, 20 Populationen x 50 Piloten, getrennt nach Kollaps der Prior-Varianz."""
    nc_med, co_med, shares = {}, {}, []
    for w in C.WELTEN:
        nc, co, sh = [], [], []
        for k in range(C.N_POP):
            pop = P.make_population(w, MR.POP_SEED(w, k))
            over, col = [], []
            for r in range(C.N_REP):
                res, inf = V.run_cell(pop, 2, ("eb",), MR.REP_PARTS(w, 2, k, r))
                over.append(res["eb"][V.MI["naiv10"]] - res["eb"][V.MI["v10"]])
                col.append(inf["eb"]["collapsed"])
            over, col = np.array(over), np.array(col)
            sh.append(col.mean())
            nc.append(over[~col].mean())
            co.append(over[col].mean())
        nc_med[w], co_med[w] = float(np.median(nc)), float(np.median(co))
        shares.append(float(np.mean(sh)))
    assert [round(nc_med[w], 1) for w in C.WELTEN] == [1.0, 0.7, 1.2, 1.1]      # glatt, Wechselwirkung, Null-Welt, Falle
    assert all(abs(co_med[w]) < 0.15 for w in C.WELTEN)                         # kollabierte Piloten: Auswahl Zufall, Überschätzung um null
    assert round(100 * min(shares)) == 31 and round(100 * max(shares)) == 53    # 31 bis 53 % der Piloten kollabieren


def test_rule_b_equals_rule_a_at_some_sites_in_glatt_and_wechselwirkung():
    share = {}
    for w in ("glatt", "wechselwirkung"):
        z = np.concatenate([P.make_population(w, MR.POP_SEED(w, k)).zB for k in range(C.N_POP)])
        tau = np.concatenate([P.make_population(w, MR.POP_SEED(w, k)).tau for k in range(C.N_POP)])
        share[w] = float((z == C.Z_A).mean())
        assert np.all(tau[z == C.Z_A] == 0.0) and np.all(tau[z != C.Z_A] != 0.0)     # gleiche Regel: derselbe Pool, wahrer Effekt exakt null
    assert share["glatt"] == pytest.approx(0.1185, abs=1e-9) and share["wechselwirkung"] == pytest.approx(0.4675, abs=1e-9)     # gerundet 11,9 % und 46,8 %


def test_the_difference_of_the_medians_differs_from_the_median_of_the_differences():
    """8,4 (Median der Unterschiede je Population) gegen 67,5 − 59,7 = 7,8 (Differenz der Mediane), Welt glatt, 30 % Budget, m = 8."""
    assert round(W.pair("glatt", "eb", "diff", 30, 8)[0], 1) == 8.4
    assert round(med("glatt", "eb", "fang", 30, 8) - med("glatt", "diff", "fang", 30, 8), 1) == 7.8


def test_the_readme_names_the_numbers_of_this_file():
    """Jede Zahl, die die README im Befund-Teil nennt, steht in den Tests oben; hier die Zeichenketten, die sie wörtlich nennt."""
    for s in ("4,9 [4,3 bis 5,7]", "2,0 [1,8 bis 2,2]", "1,2 [1,1 bis 1,2]", "4,4 [4,3 bis 4,6]", "2,1 [2,0 bis 2,2]", "1,3 [1,3 bis 1,4]", "7,5 [5,4 bis 9,1]",
              "12,7 [11,3 bis 14,1]", "11,7 [11,3 bis 12,2]", "95,6 [93,5 bis 97,4]", "87,1 [83,3 bis 91,4]", "59,7 [49,6 bis 65,2]", "67,5 [56,6 bis 75,5]", "93,1 [89,7 bis 96,0]", "98,5 [97,7 bis 99,8]",
              "-16,4 [-25,1 bis -12,6]"):
        assert s in README, s
    for s in ("m = 16 (Spanne m = 8 bis m = 32)", "m = 4 oder m = 8** (je 10 von 20 Populationen)", "(Median der Unterschiede je Population, 20 von 20; die Differenz der beiden Mediane 67,5 − 59,7 ist 7,8)", "28 000 Piloten", "11,9 % der Standorte", "an 46,8 %", "zwischen −0,1 und +0,1", "bei 0,7 bis 1,2 Prozentpunkten", "(−0,22 / −0,17 / −0,25)", "+0,21 / +0,24 / +0,10", "m = 64 (Spanne m = 32 bis über m = 64)", "17 von 20", "14 von 20", "0,6 [0,5 bis 0,8]", "(16 von 20 Populationen, bei m = 32 und 64 je 20 von 20)"):
        assert s in README, s
    assert os.path.exists(ROOT / "precomputed_sweeps.json")
