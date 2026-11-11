"""Welten und Populationen: Aufbau aus dem Zufalls-Seed, wahrer Effekt = Poolmittel, Struktur je Welt, Pilot aus den Pools."""

import numpy as np
import pytest

import wbh_constants as C
import wbh_pools as P
from wbh_stream import Stream


def test_population_is_a_pure_function_of_world_and_seed():
    for welt in C.WELTEN:
        a, b = P.make_population(welt, 5), P.make_population(welt, 5)
        assert np.array_equal(a.X, b.X) and np.array_equal(a.tau, b.tau)
        assert not np.array_equal(a.X, P.make_population(welt, 6).X)


def test_shapes_and_four_observed_features_in_every_world():
    for welt in C.WELTEN:
        pop = P.make_population(welt, 1)
        assert pop.n == C.N_SITES and pop.X.shape == (C.N_SITES, 4)
        assert pop.tau.shape == pop.sd.shape == pop.rowA.shape == (C.N_SITES,)


def test_true_effect_is_the_difference_of_the_pool_means_recomputed_from_the_bank():
    b = P.bank()
    for welt in C.WELTEN:
        pop = P.make_population(welt, 3)
        for s in (0, 17, 99):
            assert pop.tau[s] == pytest.approx(b.values[pop.rowB[s]].mean() - b.values[pop.rowA[s]].mean())
            assert pop.sd[s] == pytest.approx(np.sqrt(b.values[pop.rowA[s]].var() + b.values[pop.rowB[s]].var()))


def test_world_glatt_effect_follows_the_observed_dose_and_window_length_is_independent_of_it():
    pop = P.make_population("glatt", 2)
    assert set(np.unique(pop.zB)) <= set(C.Z_GRID) and set(np.unique(pop.length)) <= set(C.LENGTHS)
    assert np.array_equal(pop.X[:, 0], pop.zB)                      # die Dosis ist beobachtet
    zero = pop.zB == 1.0
    assert zero.any() and np.all(pop.tau[zero] == 0.0)             # Regel B = Regel A: derselbe Pool, Effekt exakt null
    # gleiche (Dosis, Länge) teilen sich den Pool und haben denselben wahren Effekt
    for z in np.unique(pop.zB):
        for L in np.unique(pop.length[pop.zB == z]):
            t = pop.tau[(pop.zB == z) & (pop.length == L)]
            assert np.all(t == t[0])
    corr = [abs(np.corrcoef(q.zB, q.length)[0, 1]) for q in (P.make_population("glatt", 100 + k) for k in range(20))]
    assert np.mean(corr) < 0.15                                      # Länge hängt nicht an der Dosis (20 Populationen, mittlerer Betrag der Korrelation)


def test_world_wechselwirkung_has_an_effect_only_when_both_features_are_high():
    pop = P.make_population("wechselwirkung", 4)
    u, v = pop.X[:, 0], pop.X[:, 1]
    sig = lambda x: 1 / (1 + np.exp(-10 * (x - 0.5)))
    assert np.array_equal(pop.zB, np.round((1.0 + 0.8 * sig(u) * sig(v)) * 10) / 10)
    low = np.minimum(u, v) < 0.3
    assert low.any() and np.all(pop.zB[low] <= 1.1)                  # s(0,3) = 0,119: z_B höchstens 1,1
    assert pop.tau[(u > 0.7) & (v > 0.7)].mean() > pop.tau[low].mean() + 2.0


def test_world_homogen_has_one_effect_for_every_site():
    pop = P.make_population("homogen", 7)
    assert np.all(pop.tau == pop.tau[0]) and pop.tau[0] > 3.0
    assert np.isnan(pop.rho)
    assert np.all(pop.length == C.WINDOW_DAYS) and np.all(pop.zB == C.Z_B)


def test_world_falle_couples_effect_and_dispersion_and_the_others_do_not_in_the_median():
    rho = {w: [P.make_population(w, 100 + k).rho for k in range(20)] for w in ("falle", "glatt", "wechselwirkung")}
    assert min(rho["falle"]) > 0.8                                    # 20 von 20 Populationen
    assert np.median(np.abs(rho["glatt"])) < C.RHO_LIMIT and np.median(np.abs(rho["wechselwirkung"])) < C.RHO_LIMIT


def test_world_falle_features_are_the_vehicle_parameters():
    pop = P.make_population("falle", 3)
    assert set(np.unique(pop.X[:, 0])) <= set(map(float, C.FALLE_LEADS))
    assert set(np.round(np.exp(np.unique(pop.X[:, 1])))) <= set(map(float, C.FALLE_DISPS))
    assert set(np.unique(pop.X[:, 2])) <= set(C.FALLE_MEANS)
    assert np.all(pop.length == C.WINDOW_DAYS)


def test_pilot_draws_pool_values_with_replacement_and_is_reproducible():
    pop = P.make_population("glatt", 1)
    b = P.bank()
    yA, yB = P.draw_pilot(pop, 5, Stream(3))
    assert yA.shape == yB.shape == (C.N_SITES, 5)
    for s in (0, 50, 99):
        assert set(yA[s]) <= set(b.values[pop.rowA[s]]) and set(yB[s]) <= set(b.values[pop.rowB[s]])
    again = P.draw_pilot(pop, 5, Stream(3))
    assert np.array_equal(yA, again[0]) and np.array_equal(yB, again[1])


def test_the_pilot_mean_approaches_the_pool_mean_for_a_large_pilot():
    pop = P.make_population("glatt", 1)
    yA, yB = P.draw_pilot(pop, 3000, Stream(4))
    se = np.sqrt(pop.sd**2 / 3000)                                    # Standardfehler der Differenz der Pilotmittel
    z = ((yB.mean(1) - yA.mean(1)) - pop.tau) / se
    assert np.abs(z).max() < 4.5 and abs(z.mean()) < 0.5
