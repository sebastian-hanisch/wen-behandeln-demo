"""Zugriff auf die vorgerechnete Messreihe `precomputed_sweeps.json` (erzeugt von tools/messreihe.py aggregate).

Jede Kurve ist eine Liste über M_GRID mit [Median, Minimum, Maximum] über die N_POP Populationen der Welt (je Population Mittel über N_REP Piloten);
`None` heißt "nicht definiert" (Schrumpfung bei m = 1, Fang-Anteil in der Null-Welt)."""

import functools
import json
from pathlib import Path

import wbh_constants as C

PATH = Path(__file__).resolve().parent / "precomputed_sweeps.json"


@functools.lru_cache(maxsize=1)
def data():
    return json.loads(PATH.read_text(encoding="utf-8"))


def curve(welt, est, kind, key):
    """kind: 'fluch' (Überschätzung der Gewählten, Prozentpunkte), 'faktor' (geschätzt/wahr), 'fang' (Fang-Anteil in %), 'netdiff' (Nettonutzen der
    Schwellenregel minus beste feste Regel, Prozentpunkte). key: Budget in % bzw. Schwelle delta."""
    return data()["welt"][welt]["est"][est][kind][str(key)]


def kipp(welt, est, delta):
    """{'median', 'min', 'max', 'n_bis_64'}; 128 = bis m = 64 nie."""
    return data()["welt"][welt]["est"][est]["kipp"][str(delta)]


def kipp_text(k):
    """Kipppunkt als Text: 'm = 16 (Spanne m = 8 bis m = 32)'; 128 = 'über m = 64' (bis m = 64 nie erreicht); liegt der Median zwischen zwei Stufen des Gitters,
    'zwischen m = 4 und m = 8' (kein gemessener Pilotumfang)."""
    def one(v):
        return "über m = 64" if v >= 128 else f"m = {int(v)}"
    med = k["median"]
    if float(med).is_integer() and (int(med) in C.M_GRID or med >= 128):
        med_txt = one(med)
    else:                                              # Median zwischen zwei Stufen des Gitters: kein gemessener Pilotumfang
        lo, hi = max(m for m in C.M_GRID if m <= med), min(m for m in C.M_GRID if m >= med)
        med_txt = f"zwischen m = {lo} und m = {hi}"
    if k["min"] == k["max"]:
        return med_txt
    return f"{med_txt} (Spanne {one(k['min'])} bis {one(k['max'])})"


def rank_at(welt, m, budget):
    """Schätzer nach Fang-Anteil (Median über die Populationen) bei Pilotumfang m, absteigend; nicht definierte fehlen."""
    mi = C.M_GRID.index(m)
    rows = []
    for e in C.ESTIMATORS:
        v = curve(welt, e, "fang", budget)[mi]
        if v is not None:
            rows.append((e, v[0], v[1], v[2]))
    return sorted(rows, key=lambda r: -r[1])


def eb_collapse(welt, m):
    """Anteil der Piloten, in denen die Prior-Varianz der Schrumpfung null ist (None bei m = 1)."""
    return data()["eb_kollaps"][welt][C.M_GRID.index(m)]


def pair(welt, a, b, budget, m):
    """(Median des Unterschieds im Fang-Anteil a minus b in Prozentpunkten, Zahl der Populationen mit a vorn, Zahl der Populationen) bei Pilotumfang m."""
    p = data()["paare"][welt][f"{a}-{b}"][str(budget)]
    mi = C.M_GRID.index(m)
    return p["median"][mi], p["vorn"][mi], p["n"]


def rho(welt):
    w = data()["welt"][welt]
    return w["rho"], w["rho_ueber_grenze"]
