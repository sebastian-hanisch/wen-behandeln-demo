"""Orakel für das Vehikel der Pools (DEMO-PLAYBOOK §12): die vektorisierte Fassung `wbh_sim` gegen (a) die skalare, aus ab-test-demo kopierte `wbh_vehicle`
(Bit-Gleichheit), (b) eine unabhängig geschriebene skalare Referenz mit allen Parametern (Mittel, Dispersion, Wiederbeschaffungszeit, Fensterlänge),
(c) die exakte Formel des Gleichgewichts und `scipy.stats.poisson.ppf` für die Poisson-Inversion; dazu die Bank der vorgerechneten Pools."""

import math

import numpy as np
import pytest

import wbh_constants as C
import wbh_pools as P
import wbh_sim as S
import wbh_vehicle as V
from wbh_rng import SplitMix64


def test_vectorized_pool_equals_the_scalar_vehicle_of_ab_test_demo_bit_for_bit():
    for z in (1.0, 1.4, 2.0):
        for seed in (1, 35, 987654321):
            ref = np.array(V.run_units(z, 25, seed))
            assert np.array_equal(S.pool(seed, z, C.BASE_MEAN, C.DISP, C.LEAD, 25), ref), (z, seed)


def _scalar_reference(seed, z, mean, disp, lead, count, length, burn=28):
    """Unabhängige skalare Referenz: Tag für Tag, ein Strom, je Tag `disp` Exponentialzahlen und eine Gleichverteilte für die Poisson-Inversion."""
    rng = SplitMix64(seed)
    sd = math.sqrt(lead * (mean + mean * mean / disp))
    level = round(lead * mean + z * sd)
    out = []
    for _ in range(count):
        net = round(lead * mean) + round(z * sd)
        pipe = [0] * lead
        dem = short = 0
        for day in range(burn + length):
            g = 0.0
            for j in range(disp):
                g = (g + rng.exponential()) if j else rng.exponential()
            lam = mean * g / disp
            u = rng.uniform()
            p = math.exp(-lam)
            cdf, d = p, 0
            while u > cdf and d < 10_000:
                d += 1
                p *= lam / d
                cdf += p
            net += pipe[0]
            served = min(max(net, 0), d)
            net -= d
            order = max(0, level - (net + sum(pipe[1:])))
            pipe = pipe[1:] + [order]
            if day >= burn:
                dem += d
                short += d - served
        out.append(1.0 - short / dem if dem > 0 else 1.0)
    return np.array(out)


@pytest.mark.parametrize("mean,disp,lead,length", [(8.0, 2, 1, 7), (40.0, 40, 4, 28), (14.0, 3, 3, 14), (26.0, 8, 2, 56), (20.0, 5, 2, 28)])
def test_vectorized_pool_equals_an_independent_scalar_reference_for_all_parameters(mean, disp, lead, length):
    for z in (1.0, 1.4):
        ref = _scalar_reference(4711, z, mean, disp, lead, 8, length)
        assert np.allclose(S.pool(4711, z, mean, disp, lead, 8, length=length), ref, rtol=0, atol=1e-12)


def test_poisson_inversion_equals_scipy_ppf_on_20000_draws():
    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(1)
    lam, u = rng.uniform(0.5, 60, 20000), rng.random(20000)
    assert np.array_equal(S.poisson_inversion(u, lam), stats.poisson.ppf(u, lam).astype(int))


def test_demand_moments_match_the_gamma_poisson_model():
    d = S.draw_demands(5, 4000, 30, 20.0, 5)
    assert abs(d.mean() - 20.0) < 0.15
    assert abs(d.var() - (20.0 + 20.0**2 / 5)) < 4.0                    # Varianz mean + mean^2/disp = 100


def test_the_bank_pool_means_are_close_to_the_exact_equilibrium_formula():
    """Quote der Summen im Gleichgewicht (Formel) gegen Mittel der Fenster-Quoten (Pool): andere Einheit, deshalb ein Band, keine Gleichheit."""
    import wbh_formulas as F
    b = P.bank()
    for z in (1.0, 1.4):
        row = b.index[P.key_name(P.key_std(z, 28))]
        assert abs(b.mean[row] - 100 * F.fill_rate(z)) < 0.5, z


def test_bank_has_every_key_the_populations_can_ask_for_and_values_in_range():
    b = P.bank()
    assert set(b.index) == {P.key_name(k) for k in P.all_keys()}
    assert b.values.shape == (len(P.all_keys()), C.POOL_SIZE)
    assert 0.0 <= b.values.min() and b.values.max() <= 100.0


def test_a_bank_pool_is_reproducible_from_its_key():
    b = P.bank()
    for key in (P.key_std(1.4, 28), P.key_std(1.0, 7), P.key_falle(3, 8, 26, 1.4)):
        again = P.quantize(P.generate_pool(key)).astype(np.float64) / P.SCALE * 100.0
        assert np.array_equal(again, b.values[b.index[P.key_name(key)]]), key


def test_higher_safety_factor_raises_the_service_level_in_every_standard_pool():
    """Ab Fensterlänge 14 Tage steigt das Poolmittel mit jedem Schritt von z; bei 7 Tagen (Standardfehler des Poolmittels 0,14 Prozentpunkte) darf ein Schritt
    um höchstens 0,3 zurückfallen, über die ganze Spanne von z = 1,0 bis 1,8 steigt es in jeder Länge."""
    b = P.bank()
    for L in C.LENGTHS:
        means = np.array([b.mean[b.index[P.key_name(P.key_std(z, L))]] for z in C.Z_GRID])
        assert means[-1] > means[0] + 3.0, L
        if L >= 14:
            assert np.all(np.diff(means) > 0), L
        else:
            assert np.diff(means).min() > -0.3, L
