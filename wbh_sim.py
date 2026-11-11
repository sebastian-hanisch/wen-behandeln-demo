"""Vektorisierte Fassung des Vehikels für die vorgerechneten Pools (die skalare Referenz ist die kopierte `wbh_vehicle.py`).

Gleiche Modellgleichungen wie `wbh_vehicle.py`, aber je Pool wählbar: Tagesmittel, Dispersion (ganzzahlig), Wiederbeschaffungszeit und Länge des Auswertefensters.
Die Welten (`wbh_pools.py`) nutzen davon nur: glatt und Wechselwirkung den Sicherheitsfaktor und die Fensterlänge, die Falle Wiederbeschaffungszeit, Dispersion und Tagesmittel.
Alle Fenster eines Pools laufen gleichzeitig durch dieselbe Tagesschleife (Wareneingang, Nachfrage, Bestellung).

Zufall: Ein Pool zieht seine Gleichverteilten aus EINEM SplitMix64-Strom `uniform_block(seed, ...)`, je Tag und Fenster in der Reihenfolge
`disp` Exponentialzahlen (Summe = Gamma(disp)), dann eine Gleichverteilte für die Poisson-Inversion. Das ist genau die Reihenfolge, in der die skalare
`wbh_vehicle.draw_demand` aus demselben Strom zieht (bei disp = 5): `pool(seed, ...)` reproduziert deshalb `wbh_vehicle.run_units(z, count, seed)`
(Test: tests/test_oracle_vehicle.py)."""

import math

import numpy as np

import wbh_constants as C
from wbh_rng import uniform_block


def level_S(z, mean, disp, lead):
    """Bestellniveau S = round(lead*mu + z*sd) mit sd der Schutzzeit-Nachfrage (i.i.d.-Näherung)."""
    sd = math.sqrt(lead * (mean + mean * mean / disp))
    return round(lead * mean + z * sd)


def init_net(z, mean, disp, lead):
    """Startbestand: Bedarf der Schutzzeit plus Sicherheitsbestand."""
    sd = math.sqrt(lead * (mean + mean * mean / disp))
    return round(lead * mean) + round(z * sd)


def poisson_inversion(u, lam, kmax=10_000):
    """Poisson(lam) durch Inversion zu den Gleichverteilten u (gleiche Schleife wie `wbh_vehicle.draw_demand`, nur über ein ganzes Feld)."""
    p = np.exp(-lam)
    cdf = p.copy()
    k = np.zeros(lam.shape, dtype=np.int64)
    active = np.flatnonzero((u > cdf).ravel())
    uf, lf, pf, cf, kf = u.ravel(), lam.ravel(), p.ravel().copy(), cdf.ravel().copy(), k.ravel()
    step = 0
    while active.size and step < kmax:
        step += 1
        kf[active] = step
        pf[active] = pf[active] * lf[active] / step
        cf[active] = cf[active] + pf[active]
        active = active[uf[active] > cf[active]]
    return kf.reshape(lam.shape)


def draw_demands(seed, windows, days, mean, disp):
    """Tagesnachfragen (windows, days), ganzzahlig, aus dem Strom SplitMix64(seed)."""
    disp = int(disp)
    u = uniform_block(seed, windows * days * (disp + 1)).reshape(windows, days, disp + 1)
    g = -np.log(1.0 - u[..., 0])
    for j in range(1, disp):                                  # feste Summationsreihenfolge (wie die skalare Fassung)
        g = g + (-np.log(1.0 - u[..., j]))
    lam = mean * g / disp
    return poisson_inversion(u[..., disp], lam)


def windows_from_demands(z, demands, mean, disp, lead, burn=C.BURN_DAYS):
    """Servicegrad (Typ 2, Quote der Summen im Fenster) je Fenster aus vorgegebenen Tagesnachfragen (W, T)."""
    W, T = demands.shape
    S = level_S(z, mean, disp, lead)
    net = np.full(W, init_net(z, mean, disp, lead), dtype=np.int64)
    pipe = np.zeros((W, lead), dtype=np.int64)                # pipe[:, 0] kommt am nächsten Tag an
    dem_sum = np.zeros(W, dtype=np.int64)
    short_sum = np.zeros(W, dtype=np.int64)
    for day in range(T):
        net = net + pipe[:, 0]
        d = demands[:, day]
        served = np.minimum(np.maximum(net, 0), d)
        net = net - d
        position = net + pipe[:, 1:].sum(axis=1)
        order = np.maximum(0, S - position)
        pipe = np.concatenate([pipe[:, 1:], order[:, None]], axis=1)
        if day >= burn:
            dem_sum += d
            short_sum += d - served
    return np.where(dem_sum > 0, 1.0 - short_sum / np.maximum(dem_sum, 1), 1.0)


def pool(seed, z, mean, disp, lead, windows, length=C.WINDOW_DAYS, burn=C.BURN_DAYS):
    """Ein Pool von `windows` unabhängigen Standort-Fenstern (Servicegrade in [0, 1])."""
    d = draw_demands(seed, windows, burn + length, mean, disp)
    return windows_from_demands(z, d, mean, disp, lead, burn)
