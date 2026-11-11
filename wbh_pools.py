"""Vorgerechnete Pools und Populationen von Standorten.

**Pools.** Je (Standorttyp, Regel, Fensterlänge) gibt es einen Pool von POOL_SIZE unabhängigen Standort-Fenstern (Servicegrad); sie stehen in
`precomputed_pools.npz` (erzeugt von `generate_precomputed.py`, SplitMix64, je Pool ein eigener Strom). Standorte mit gleichen Eigenschaften teilen sich
einen Pool. **Wahrheit = Poolmittel**: der wahre Effekt eines Standorts ist Mittel(B) minus Mittel(A) seiner beiden Pools (Prozentpunkte Servicegrad je
Fenster); der Pilot zieht m Fenster je Standort und Arm mit Zurücklegen aus den Pools.

**Welten** (Populationen von N_SITES Standorten, aus dem Zufalls-Seed erzeugt):

  glatt          Standardstandorte; Regel B fährt je Standort einen Sicherheitsfaktor z_B aus Z_GRID (beobachtet), die Fensterlänge je Standort ist zufällig
                 aus LENGTHS und unabhängig von z_B gezogen. Kurze Fenster haben allerdings einen etwas kleineren wahren Effekt und eine größere Streuung der Einzeldifferenz:
                 Effekt und Streuung sind dadurch nur schwach gekoppelt (Korrelation im Median etwa -0,2, nicht null).
  wechselwirkung z_B = 1,0 + 0,8 * s(u) * s(v) (s = steile Sigmoide um 0,5) mit zwei beobachteten Merkmalen u, v: großer Effekt nur, wenn u UND v hoch
                 sind. **Gesetzte Zuordnung der Regelstärke, kein Ergebnis des Vehikels.** Fensterlänge zufällig wie in "glatt".
  homogen        alle Standorte gleich (Standardstandort, z_B = 1,4, Fenster 28 Tage), alle Merkmale Rotheringe.
  falle          Vehikel pur: Wiederbeschaffungszeit, Dispersion und Tagesmittel je Standort aus Gittern, Fenster 28 Tage; Effekt und Streuung hängen
                 zusammen (gemessen in der README).
Beobachtet wird je Standort X mit 4 Spalten (die Merkmale der Welt, der Rest Rotheringe aus der Standardnormalverteilung)."""

import functools
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import wbh_constants as C
import wbh_sim as S
from wbh_stream import Stream, mix_seed

POOLS_PATH = Path(__file__).resolve().parent / "precomputed_pools.npz"
POP_TAG = 3_100_000
POOL_TAG = 7_000_000
SCALE = 65535.0                                   # Pools liegen als uint16 (Servicegrad * 65535) in der Datei


def key_std(z, length):
    return ("s", int(round(z * 10)), int(length))


def key_falle(lead, disp, mean, z):
    return ("f", int(lead), int(disp), int(round(mean)), int(round(z * 10)))


def key_name(key):
    return "|".join(str(p) for p in key)


def all_keys():
    """Alle Pools der Bank in fester Reihenfolge."""
    keys = [key_std(z, L) for z in C.Z_GRID for L in C.LENGTHS]
    for lead in C.FALLE_LEADS:
        for disp in C.FALLE_DISPS:
            for mean in C.FALLE_MEANS:
                for z in (C.Z_A, C.Z_B):
                    keys.append(key_falle(lead, disp, mean, z))
    return keys


def pool_seed(key):
    kind = 1 if key[0] == "s" else 2
    return mix_seed(POOL_TAG, kind, *key[1:])


def generate_pool(key, windows=C.POOL_SIZE):
    """Ein Pool (Servicegrade in [0, 1]) für einen Schlüssel; der Strom hängt nur vom Schlüssel ab."""
    if key[0] == "s":
        _, z10, length = key
        return S.pool(pool_seed(key), z10 / 10.0, C.BASE_MEAN, C.DISP, C.LEAD, windows, length=length)
    _, lead, disp, mean, z10 = key
    return S.pool(pool_seed(key), z10 / 10.0, float(mean), disp, lead, windows)


def quantize(values):
    return np.round(np.asarray(values) * SCALE).astype(np.uint16)


@dataclass(frozen=True)
class Bank:
    index: dict
    values: np.ndarray      # (Pools, POOL_SIZE) in Prozentpunkten Servicegrad (0 bis 100)
    mean: np.ndarray        # Poolmittel
    var: np.ndarray         # Poolvarianz (ddof = 0)


@functools.lru_cache(maxsize=1)
def bank():
    data = np.load(POOLS_PATH, allow_pickle=False)
    keys = [str(k) for k in data["keys"]]
    values = data["values"].astype(np.float64) / SCALE * 100.0
    return Bank({k: i for i, k in enumerate(keys)}, values, values.mean(axis=1), values.var(axis=1))


@dataclass(frozen=True)
class Population:
    welt: str
    seed: int
    X: np.ndarray            # (n, 4) beobachtete Merkmale
    rowA: np.ndarray         # Pool-Zeile von Regel A je Standort
    rowB: np.ndarray         # Pool-Zeile von Regel B je Standort
    tau: np.ndarray          # wahrer Effekt je Standort (Prozentpunkte)
    sd: np.ndarray           # Streuung der Einzeldifferenz je Fenster-Paar (Prozentpunkte)
    length: np.ndarray       # Fensterlänge je Standort (Tage)
    zB: np.ndarray           # Sicherheitsfaktor der Regel B je Standort

    @property
    def n(self):
        return len(self.tau)

    @property
    def rho(self):
        """Pearson-Korrelation zwischen wahrem Effekt und Streuung der Einzeldifferenz (nan, wenn alle Effekte gleich sind)."""
        if self.tau.std() < 1e-12 or self.sd.std() < 1e-12:
            return float("nan")
        return float(np.corrcoef(self.tau, self.sd)[0, 1])


def _sig(x):
    return 1.0 / (1.0 + np.exp(-10.0 * (x - 0.5)))


def make_population(welt, seed, n=C.N_SITES):
    """N_SITES Standorte der Welt `welt`, ganz aus dem Zufalls-Seed bestimmt."""
    b = bank()
    st = Stream(mix_seed(POP_TAG, C.WELTEN.index(welt), seed))
    noise = st.normal((n, 4))
    lengths = np.full(n, C.WINDOW_DAYS)
    z_b = np.full(n, C.Z_B)
    if welt == "falle":
        lead = 1 + st.integers(len(C.FALLE_LEADS), n)
        disp = np.array(C.FALLE_DISPS)[st.integers(len(C.FALLE_DISPS), n)]
        mean = np.array(C.FALLE_MEANS)[st.integers(len(C.FALLE_MEANS), n)]
        X = np.column_stack([lead.astype(float), np.log(disp), mean, noise[:, 0]])
        rowA = np.array([b.index[key_name(key_falle(lead[s], disp[s], mean[s], C.Z_A))] for s in range(n)])
        rowB = np.array([b.index[key_name(key_falle(lead[s], disp[s], mean[s], C.Z_B))] for s in range(n)])
    else:
        if welt == "glatt":
            z_b = np.array(C.Z_GRID)[st.integers(len(C.Z_GRID), n)]
            lengths = np.array(C.LENGTHS)[st.integers(len(C.LENGTHS), n)]
            X = np.column_stack([z_b, noise[:, :3]])
        elif welt == "wechselwirkung":
            u, v = st.uniform(n), st.uniform(n)
            z_b = np.round((1.0 + 0.8 * _sig(u) * _sig(v)) * 10) / 10
            lengths = np.array(C.LENGTHS)[st.integers(len(C.LENGTHS), n)]
            X = np.column_stack([u, v, noise[:, :2]])
        elif welt == "homogen":
            X = noise
        else:
            raise ValueError(welt)
        rowA = np.array([b.index[key_name(key_std(C.Z_A, lengths[s]))] for s in range(n)])
        rowB = np.array([b.index[key_name(key_std(z_b[s], lengths[s]))] for s in range(n)])
    tau = b.mean[rowB] - b.mean[rowA]
    sd = np.sqrt(b.var[rowA] + b.var[rowB])
    return Population(welt, seed, X, rowA, rowB, tau, sd, lengths, z_b)


def draw_pilot(pop, m, stream):
    """Pilot: m Fenster je Standort und Arm, mit Zurücklegen aus den Pools (Prozentpunkte Servicegrad). Rückgabe (yA, yB), je (n, m)."""
    b = bank()
    u = stream.uniform((2, pop.n, m))
    ia = np.floor(u[0] * C.POOL_SIZE).astype(np.int64)
    ib = np.floor(u[1] * C.POOL_SIZE).astype(np.int64)
    return b.values[pop.rowA[:, None], ia], b.values[pop.rowB[:, None], ib]


def effect_summary(pop):
    """Kennzahlen der wahren Effekte: Mittel, Streuung, Minimum, Maximum (Prozentpunkte)."""
    return {"mean": float(pop.tau.mean()), "sd": float(pop.tau.std()), "min": float(pop.tau.min()), "max": float(pop.tau.max())}
