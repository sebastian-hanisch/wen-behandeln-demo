"""Orakel für den Kern der Demo (DEMO-PLAYBOOK §12): Auswahl, Gewinner-Fluch, Schrumpfung und Bäume gegen einen anderen Rechenweg.

* Gewinner-Fluch im Normalmodell gegen die geschlossene Formel (Erwartungswerte der Ordnungsstatistiken per Quadratur, scipy) und die bedingte Erwartung
  E[tau | tau_hat] = mu + B (tau_hat - mu).
* Schrumpfung: Hand-Rechnung; mit bekannter Varianz je Standort (Mittel und Streuung der wahren Effekte geschätzt) sinkt die Überschätzung der Gewählten unter 0,1.
* Auswahl unter Budget: top-k gegen Aufzählung aller Teilmengen und gegen `scipy.optimize.milp`.
* Regressionsbaum gegen `sklearn.tree.DecisionTreeRegressor` (Instanzen ohne Gleichstände), Regression gegen exakt lineare Wahrheit.
* Kipppunkt an handgesetzten Reihen."""

import itertools
import math

import numpy as np
import pytest

import wbh_constants as C
import wbh_estimators as E
import wbh_evaluation as V
from wbh_stream import Stream

# ---------------------------------------------------------------------------------------------------------------------------------------
# Gewinner-Fluch im Normalmodell
# ---------------------------------------------------------------------------------------------------------------------------------------


def _expected_top_k_sum(n, k, scale):
    """E[Summe der k größten von n unabhängigen N(0, scale^2)]: Summe der Erwartungswerte der Ordnungsstatistiken, jede per Quadratur."""
    stats = pytest.importorskip("scipy.stats")
    x = np.linspace(-9, 9, 100001)
    phi, Phi = stats.norm.pdf(x), stats.norm.cdf(x)
    total = 0.0
    for j in range(1, k + 1):
        r = n - j + 1                                           # j-größte = r-kleinste
        density = r * math.comb(n, r) * phi * Phi ** (r - 1) * (1 - Phi) ** (n - r)
        total += np.trapezoid(x * density, x)
    return total * scale


def test_winners_curse_matches_the_closed_form_in_the_normal_model():
    n, k, mu, sigma, v = 100, 10, 3.0, 1.0, 2.0
    B = sigma**2 / (sigma**2 + v**2)
    mean_est = mu + _expected_top_k_sum(n, k, math.sqrt(sigma**2 + v**2)) / k
    expected_over = (1 - B) * (mean_est - mu)                    # E[tau_hat - tau | gewählt] = (1 - B) (tau_hat - mu)
    st = Stream(11)
    over, true = [], []
    for _ in range(2500):
        tau = mu + sigma * st.normal(n)
        row = V.evaluate(tau, tau + v * st.normal(n), st)
        over.append(row[V.MI["naiv10"]] - row[V.MI["v10"]])
        true.append(row[V.MI["v10"]])
    assert expected_over == pytest.approx(3.094, abs=0.001)      # Zahl der Handrechnung (Quadratur mit feinerem Gitter nachgerechnet)
    assert np.mean(over) == pytest.approx(expected_over, abs=0.06)
    assert np.mean(true) == pytest.approx(mean_est - expected_over, abs=0.06)   # wahrer Mittelwert der Gewählten = E[tau | tau_hat] gemittelt


def test_the_mean_of_the_k_largest_estimates_exceeds_the_mean_for_every_k_below_n_but_not_every_chosen_site_does():
    """Die Überschätzung ist für jedes k < N positiv (Mittel der k größten über dem Mittel); jeder Gewählte hat eine Schätzung über dem Mittel nur für k bis etwa N/2."""
    for k in (10, 50, 80):
        assert _expected_top_k_sum(100, k, 1.0) / k > 0.2
    assert _expected_top_k_sum(100, 100, 1.0) == pytest.approx(0.0, abs=1e-6)       # k = N: alle, Überschätzung null
    assert V.k_of(80) == 80


def test_shrinking_with_the_known_variance_removes_the_winners_curse():
    """Mit bekannter Varianz je Standort, aber geschätztem Mittel und geschätzter Streuung der wahren Effekte: die Überschätzung der Gewählten liegt unter 0,1 (ohne Schrumpfung 3,1)."""
    n, mu, sigma, v = 100, 3.0, 1.0, 2.0
    st = Stream(12)
    over = []
    for _ in range(1500):
        tau = mu + sigma * st.normal(n)
        th, _ = E.eb_shrink(tau + v * st.normal(n), np.full(n, v**2))
        row = V.evaluate(tau, th, st)
        over.append(row[V.MI["naiv10"]] - row[V.MI["v10"]])
    assert abs(np.mean(over)) < 0.1                              # ohne Schrumpfung: 3,1 (siehe oben); Rest durch geschätztes mu und sigma^2


# ---------------------------------------------------------------------------------------------------------------------------------------
# Schrumpfung (Empirical Bayes) von Hand
# ---------------------------------------------------------------------------------------------------------------------------------------


def test_eb_shrink_by_hand():
    D = np.array([1.0, 2.0, 6.0])
    shrunk, tau2 = E.eb_shrink(D, np.ones(3))
    # mu = 3, Var(D, ddof=1) = (4 + 1 + 9) / 2 = 7, tau2 = 7 - 1 = 6, B = 6 / 7
    assert tau2 == pytest.approx(6.0)
    assert shrunk == pytest.approx(3 + 6 / 7 * (D - 3))


def test_eb_collapses_to_the_grand_mean_when_the_prior_variance_is_zero():
    D = np.array([1.0, 2.0, 3.0])                                # Var(D) = 1, mittlere se2 = 4: tau2 = max(0, -3) = 0
    shrunk, tau2 = E.eb_shrink(D, np.full(3, 4.0))
    assert tau2 == 0.0 and np.all(shrunk == 2.0)


def test_moderated_variance_by_hand():
    yA = np.array([[0.0, 2.0], [1.0, 1.0]])
    yB = np.array([[0.0, 4.0], [2.0, 2.0]])
    # Fensterstreuung (ddof=1) je Standort: A: [2, 0], B: [8, 0], Summe [10, 0], Mittel 5; d0 = 4, m = 2:
    # s2m = (2 * 1 * s2 + 4 * 5) / (2 * 1 + 4) = [40/6, 20/6]; Varianz der Differenz = s2m / m
    assert E.moderated_se2(yA, yB, d0=4.0) == pytest.approx([40 / 6 / 2, 20 / 6 / 2])


def test_eb_is_undefined_at_m_equal_one_and_defined_from_two():
    st = Stream(1)
    X = st.normal((20, 4))
    theta, info = E.estimate("eb", X, st.normal((20, 1)), st.normal((20, 1)), st)
    assert np.all(np.isnan(theta)) and info["defined"] is False
    theta, info = E.estimate("eb", X, st.normal((20, 2)), st.normal((20, 2)), st)
    assert info["defined"] is True and not np.any(np.isnan(theta))


# ---------------------------------------------------------------------------------------------------------------------------------------
# Auswahl unter Budget: top-k ist das Optimum des 0-1-Problems mit Kardinalitätsschranke
# ---------------------------------------------------------------------------------------------------------------------------------------


def _brute_force_best(th, k):
    return max(sum(th[list(c)]) for c in itertools.combinations(range(len(th)), k))


def test_top_k_equals_enumeration_of_all_subsets():
    rng = np.random.default_rng(0)
    for t in range(40):
        n = int(rng.integers(3, 10))
        k = int(rng.integers(1, n))
        th = np.round(rng.normal(size=n), 1)                    # gerundet: Gleichstände kommen vor
        chosen = V.rank_order(th, Stream(t))[:k]
        assert len(set(chosen)) == k
        assert th[chosen].sum() == pytest.approx(_brute_force_best(th, k)), (t, th, k)


def test_top_k_equals_the_milp_optimum():
    opt = pytest.importorskip("scipy.optimize")
    rng = np.random.default_rng(1)
    for t in range(30):
        n, k = int(rng.integers(5, 40)), int(rng.integers(1, 5))
        th = rng.normal(size=n)
        res = opt.milp(c=-th, constraints=opt.LinearConstraint(np.ones((1, n)), k, k), integrality=np.ones(n), bounds=opt.Bounds(0, 1))
        assert th[V.rank_order(th, Stream(t))[:k]].sum() == pytest.approx(-res.fun)


def test_ties_are_broken_at_random_not_by_position():
    th = np.zeros(50)
    firsts = {int(V.rank_order(th, Stream(s))[0]) for s in range(30)}
    assert len(firsts) > 10


# ---------------------------------------------------------------------------------------------------------------------------------------
# Bäume und Regression
# ---------------------------------------------------------------------------------------------------------------------------------------


def test_regression_tree_equals_sklearn_on_instances_without_ties():
    sk = pytest.importorskip("sklearn.tree")
    rng = np.random.default_rng(0)
    for t in range(60):
        n = int(rng.integers(6, 60))
        X, y = rng.normal(size=(n, 4)), rng.normal(size=n)
        X[:, 0] += y
        ml = int(rng.choice([1, 2, 3, 5]))
        mine = E.tree_fit_predict(X, y, X, ml)
        ref = sk.DecisionTreeRegressor(min_samples_leaf=ml, random_state=0).fit(X, y).predict(X)
        assert mine == pytest.approx(ref), (t, n, ml)


def test_tree_predicts_new_rows_by_the_same_splits():
    sk = pytest.importorskip("sklearn.tree")
    rng = np.random.default_rng(3)
    # große Blätter: in kleinen Blättern (z. B. 6 Zeilen bei Mindestgröße 3) liefern verschiedene Merkmale dieselbe Teilung der Trainingszeilen, und sklearn wählt
    # unter ihnen zufällig; für neue Zeilen sind die Teilungen dann verschieden.
    X, y = rng.normal(size=(200, 4)), rng.normal(size=200)
    X[:, 1] += y
    Xn = rng.normal(size=(60, 4))
    ref = sk.DecisionTreeRegressor(min_samples_leaf=20, random_state=0).fit(X, y).predict(Xn)
    assert E.tree_fit_predict(X, y, Xn, 20) == pytest.approx(ref)


def test_forest_recovers_a_noise_free_step_function_and_is_deterministic():
    rng = np.random.default_rng(4)
    X = rng.uniform(size=(100, 4))
    y = (X[:, 0] > 0.5).astype(float)
    p1 = E.forest_predict(X, y, Stream(5), 2)
    assert np.mean((p1 - y) ** 2) < 0.03
    assert np.array_equal(p1, E.forest_predict(X, y, Stream(5), 2))
    assert not np.array_equal(p1, E.forest_predict(X, y, Stream(6), 2))


def test_linear_t_learner_recovers_an_exactly_linear_effect():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(30, 4))
    base = 90 + X @ np.array([1.0, -2.0, 0.5, 0.0])
    effect = 2.0 + X @ np.array([0.7, 0.0, -0.3, 1.1])
    theta, _ = E.estimate("tlin", X, base[:, None], (base + effect)[:, None], Stream(1))
    assert theta == pytest.approx(effect, abs=1e-8)


def test_product_terms_recover_an_interaction_that_the_plain_regression_misses():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(60, 4))
    Xc = (X - X.mean(0)) / X.std(0)
    effect = 1.5 + 2.0 * Xc[:, 0] * Xc[:, 1]
    base = np.zeros(60)
    th2, _ = E.estimate("tlin2", X, base[:, None], effect[:, None], Stream(1))
    th1, _ = E.estimate("tlin", X, base[:, None], effect[:, None], Stream(1))
    assert th2 == pytest.approx(effect, abs=1e-8)
    assert np.abs(th1 - effect).max() > 1.0


# ---------------------------------------------------------------------------------------------------------------------------------------
# Kipppunkt
# ---------------------------------------------------------------------------------------------------------------------------------------


def _rows(positive_from):
    """(7, 50): die Zeilen ab Index `positive_from` haben Mittel 1 und kleine Streuung, die davor Mittel 0."""
    base = np.tile(np.linspace(-0.01, 0.01, 50), (7, 1))
    base[positive_from:] += 1.0
    return base


def test_kipp_point_by_hand():
    assert V.kipp_point(_rows(3)) == C.M_GRID[3] == 8
    assert V.kipp_point(_rows(0)) == 1
    assert V.kipp_point(_rows(7)) == 128                         # nie
    x = _rows(2)
    x[3] = np.linspace(-0.01, 0.01, 50)                         # bei m = 8 wieder nicht besser: dauerhaft erst ab m = 16
    assert V.kipp_point(x) == 16


def test_kipp_point_ignores_rounding_noise_in_the_null_world():
    x = np.full((7, 50), 3e-16)                                  # Rundungsrauschen, kein Gewinn
    assert V.kipp_point(x) == 128
