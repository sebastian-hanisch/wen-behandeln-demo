"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Schwelle und jede genannte Zahl steht hier. Ein Verdikt ist eine reine Funktion (Art, Text)."""

from types import SimpleNamespace

import numpy as np

import wbh_constants as C
import wbh_evaluation as V
import wbh_pools as P


def _analysis(est_mean=6.0, true_mean=4.0, net_diff=0.1, se=0.01, defined=True, delta=1, n=40):
    pop = P.make_population("glatt", 1)
    reps = ({"n": n, "est_mean": est_mean, "true_mean": true_mean, "over": est_mean - true_mean,
             "factor": est_mean / true_mean if true_mean > 0.05 else float("nan"), "net_diff": net_diff, "net_diff_se": se} if defined else {"n": 0})
    return V.Analysis(V.Settings(0, 1, 10, delta, 0, 35), pop, defined, False, np.zeros(pop.n), np.arange(10), 10, est_mean, true_mean, 5.0, 3.0, 0.0, -float(delta), 0, {}, reps)


# --- Gewinner-Fluch (Live) ---------------------------------------------------------------------------------------------------------------


def test_verdict_fluch_warns_from_factor_1_5_inclusive_and_names_the_numbers():
    kind, text = V.verdict_fluch(_analysis(6.0, 4.0))                       # Faktor genau 1,5
    assert kind == "warning"
    assert "**10 Gewählten**" in text and "**6,0 Prozentpunkten**" in text and "**4,0 Prozentpunkte**" in text
    assert "Überschätzung **2,0 Prozentpunkte**" in text and "Faktor **1,5**" in text and "40 Piloten" in text


def test_verdict_fluch_info_between_1_1_and_1_5_and_success_below():
    assert V.verdict_fluch(_analysis(4.4, 4.0))[0] == "info"                # Faktor genau 1,1
    kind, text = V.verdict_fluch(_analysis(4.39, 4.0))
    assert kind == "success" and "nahezu unverzerrt" in text


def test_verdict_fluch_says_underestimation_when_the_estimate_is_below_the_truth():
    kind, text = V.verdict_fluch(_analysis(3.0, 4.0))
    assert kind == "info" and "unterschätzt" in text and "1,0 Prozentpunkte" in text


def test_verdict_fluch_without_a_positive_true_mean_gives_no_factor():
    kind, text = V.verdict_fluch(_analysis(2.0, 0.0))
    assert kind == "info" and "Faktor" not in text and "2,0 Prozentpunkte" in text


def test_verdict_fluch_for_the_undefined_shrinkage():
    kind, text = V.verdict_fluch(_analysis(defined=False))
    assert kind == "info" and "m = 1" in text


# --- Kipppunkt (Live) --------------------------------------------------------------------------------------------------------------------


def test_verdict_kipp_thresholds_are_two_standard_errors_strict():
    assert V.verdict_kipp(_analysis(net_diff=0.1, se=0.01))[0] == "success"
    assert V.verdict_kipp(_analysis(net_diff=-0.1, se=0.01))[0] == "warning"
    assert V.verdict_kipp(_analysis(net_diff=0.02, se=0.01))[0] == "info"        # genau 2 Standardfehler: noch kein Unterschied
    assert V.verdict_kipp(_analysis(net_diff=0.0201, se=0.01))[0] == "success"
    assert V.verdict_kipp(_analysis(net_diff=1e-12, se=0.0))[0] == "info"        # Rundungsrauschen ist kein Gewinn


def test_verdict_kipp_names_the_best_fixed_rule_and_the_numbers():
    text = V.verdict_kipp(_analysis(net_diff=0.25, se=0.05, delta=1))[1]
    assert "„keinen behandeln“" in text and "+0,25 Prozentpunkte" in text
    assert "0,050 Standardfehler" in text and "40 Piloten" in text
    assert "„alle behandeln“" in V.verdict_kipp(_analysis(delta=-1))[1]


def test_verdict_kipp_calls_a_tiny_difference_small_and_a_larger_one_not():
    assert "Der Betrag ist klein" in V.verdict_kipp(_analysis(net_diff=-0.01, se=0.001))[1]
    assert "Der Betrag ist klein" in V.verdict_kipp(_analysis(net_diff=0.049, se=0.001))[1]
    assert "Der Betrag ist klein" not in V.verdict_kipp(_analysis(net_diff=0.05, se=0.001))[1]
    assert "Der Betrag ist klein" not in V.verdict_kipp(_analysis(net_diff=-0.3, se=0.001))[1]


# --- Tabellen-Verdikte aus der vorgerechneten Messreihe ----------------------------------------------------------------------------------


def test_verdict_kipp_table_names_median_and_range():
    kind, text = V.verdict_kipp_table("glatt", "diff", 1)
    assert kind == "info" and "m = 16" in text and "Spanne m = 8 bis m = 32" in text and "20 von 20" in text


def test_verdict_kipp_table_in_the_null_world_is_never():
    kind, text = V.verdict_kipp_table("homogen", "diff", 1)
    assert kind == "warning" and "in keiner der 20 Populationen" in text


def test_verdict_fluch_table_uses_the_median_and_the_factor():
    kind, text = V.verdict_fluch_table("glatt", "diff", 10, 1)
    assert kind == "warning" and "**12,7 Prozentpunkte**" in text and "Faktor geschätzt/wahr **4,9**" in text and "11,3 bis 14,1" in text
    assert V.verdict_fluch_table("glatt", "diff", 10, 64)[0] == "info"                 # Faktor 1,2
    assert V.verdict_fluch_table("glatt", "eb", 10, 8)[0] == "success"                  # Faktor 1,0
    assert "nicht definiert" in V.verdict_fluch_table("glatt", "eb", 10, 1)[1]
    kind, text = V.verdict_fluch_table("wechselwirkung", "eb", 10, 8)
    assert kind == "info" and "unterschätzt" in text and "0,6 Prozentpunkte" in text    # Überschätzung -0,6: die Korrektur schießt über das Ziel


def test_verdict_zoo():
    kind, text = V.verdict_zoo("glatt", 8, 30)
    assert kind == "info" and "**T-linear** vorn" in text and "96 %" in text and "T-linear liegt in 20 von 20 Populationen vor T-Wald" in text
    text = V.verdict_zoo("wechselwirkung", 8, 30)[1]
    assert "**T-Wald** vorn" in text and "T-Wald liegt in 20 von 20 Populationen vor T-linear" in text
    assert "nicht definiert" in V.verdict_zoo("homogen", 8, 30)[1]


def test_verdict_kollaps():
    assert "nicht definiert" in V.verdict_kollaps("glatt", 1)[1]
    kind, text = V.verdict_kollaps("falle", 2)
    assert kind == "warning" and "**43 %**" in text and "Zufall" in text
    kind, text = V.verdict_kollaps("glatt", 16)
    assert kind == "info" and "**0 %**" in text


# --- Kopplung von Effekt und Streuung ----------------------------------------------------------------------------------------------------


def test_verdict_falle_thresholds():
    pop = lambda rho: SimpleNamespace(n=100, rho=rho)
    assert V.verdict_falle(pop(0.8))[0] == "warning" and "eng zusammen" in V.verdict_falle(pop(0.8))[1]
    assert V.verdict_falle(pop(-0.8))[0] == "warning"
    kind, text = V.verdict_falle(pop(0.5))
    assert kind == "info" and "Mäßig" in text and "**0,50**" in text
    assert "nur schwach gekoppelt" in V.verdict_falle(pop(0.29))[1]
    assert "denselben Effekt" in V.verdict_falle(pop(float("nan")))[1]
    assert C.RHO_LIMIT == 0.3 and C.RHO_STRONG == 0.8
