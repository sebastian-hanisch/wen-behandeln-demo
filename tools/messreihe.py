"""Die vorgerechnete Messreihe der Demo (Zufallsgenerator der Demo: SplitMix64), gleiche Definitionen wie die Vormessung.

    _venvs/test/Scripts/python.exe tools/messreihe.py tune       # Blattgröße des Waldes je Pilotumfang auf getrennten Populationen und Seeds
    _venvs/test/Scripts/python.exe tools/messreihe.py run        # je Welt N_POP Populationen x M_GRID x N_REP Piloten -> _scratch/cells/*.npz
    _venvs/test/Scripts/python.exe tools/messreihe.py aggregate  # -> precomputed_sweeps.json (Tabellen und Kurven der App)

Zelle = ein Pilot einer Population bei einem Pilotumfang; Population = N_SITES Standorte einer Welt (Zufalls-Seed `POP_SEED(welt, k)`);
Messen und Abstimmen benutzen verschiedene Populationen und Zufalls-Seeds."""

import json
import warnings
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import wbh_constants as C  # noqa: E402
import wbh_evaluation as V  # noqa: E402
import wbh_pools as P  # noqa: E402

CELLS = os.path.join(ROOT, "_scratch", "cells")
OUT = os.path.join(ROOT, "precomputed_sweeps.json")
TUNE_JSON = os.path.join(ROOT, "tools", "tuning.json")
WIDX = {w: i for i, w in enumerate(C.WELTEN)}
POP_SEED = lambda welt, k: 3_000_000 + 1000 * WIDX[welt] + k          # Messen
TUNE_POP_SEED = lambda welt: 880_000 + WIDX[welt]                      # Abstimmen: andere Population
REP_PARTS = lambda welt, m, k, r: (70_000_000, WIDX[welt], m, k, r)    # Messen
TUNE_PARTS = lambda welt, m, r: (555_000, WIDX[welt], m, r)            # Abstimmen
TUNE_REPS = 20
TUNE_WELTEN = ("glatt", "wechselwirkung", "falle")                     # in der Null-Welt gibt das Abstimmen kein Signal


def tune_task(args):
    welt, m = args
    pop = P.make_population(welt, TUNE_POP_SEED(welt))
    out = {}
    for ml in C.LEAF_GRID:
        C.LEAF_BY_M[m] = ml
        vals = [V.run_cell(pop, m, ("twald",), TUNE_PARTS(welt, m, r))[0]["twald"][V.MI["v30"]] for r in range(TUNE_REPS)]
        out[ml] = float(np.mean(vals))
    return welt, m, out


def tune():
    t0 = time.time()
    tasks = [(w, m) for w in TUNE_WELTEN for m in C.M_GRID]
    with Pool(14) as p:
        res = p.map(tune_task, tasks, chunksize=1)
    table = {}
    for w, m, out in res:
        table.setdefault(str(m), {})[w] = out
    best = {}
    for m in C.M_GRID:
        avg = {ml: float(np.mean([table[str(m)][w][ml] for w in TUNE_WELTEN])) for ml in C.LEAF_GRID}
        best[m] = max(C.LEAF_GRID, key=lambda ml: (round(avg[ml], 9), -ml))      # bei Gleichstand die kleinere Blattgröße
        table[str(m)]["mittel"] = avg
    json.dump({"LEAF_BY_M": best, "tabelle": table}, open(TUNE_JSON, "w", encoding="utf-8"), indent=1)
    print("LEAF_BY_M =", best, round(time.time() - t0), "s")


def run_task(args):
    welt, k = args
    pop = P.make_population(welt, POP_SEED(welt, k))
    nm, ne, nx = len(C.M_GRID), len(C.ESTIMATORS), len(V.METRICS)
    arr = np.full((nm, C.N_REP, ne, nx), np.nan, dtype=np.float32)
    collapsed = np.zeros((nm, C.N_REP), dtype=bool)
    for mi, m in enumerate(C.M_GRID):
        for r in range(C.N_REP):
            res, infos = V.run_cell(pop, m, C.ESTIMATORS, REP_PARTS(welt, m, k, r))
            for ei, name in enumerate(C.ESTIMATORS):
                if res[name] is not None:
                    arr[mi, r, ei] = res[name]
            collapsed[mi, r] = infos["eb"].get("collapsed", False)
    os.makedirs(CELLS, exist_ok=True)
    np.savez_compressed(os.path.join(CELLS, f"{welt}_{k:02d}.npz"), arr=arr, collapsed=collapsed, tau=pop.tau, sd=pop.sd, rho=pop.rho)
    return welt, k


def run():
    t0 = time.time()
    tasks = [(w, k) for k in range(C.N_POP) for w in C.WELTEN]
    with Pool(14) as p:
        for w, k in p.imap_unordered(run_task, tasks, chunksize=1):
            print(w, k, "fertig", round(time.time() - t0), flush=True)


# ---------------------------------------------------------------------------------------------------------------------------------------
def _band(x):
    """[Median, Minimum, Maximum] über die Populationen (nan -> None)."""
    x = np.asarray(x, dtype=float)
    if np.all(np.isnan(x)):
        return None
    return [round(float(np.nanmedian(x)), 4), round(float(np.nanmin(x)), 4), round(float(np.nanmax(x)), 4)]


PAIRS = (("twald", "tlin"), ("tlin2", "tlin"), ("twald", "tlin2"), ("eb", "diff"), ("tlin", "diff"))


def aggregate():
    warnings.simplefilter("ignore", RuntimeWarning)
    cells = {w: [np.load(os.path.join(CELLS, f"{w}_{k:02d}.npz")) for k in range(C.N_POP)] for w in C.WELTEN}
    EI = {e: i for i, e in enumerate(C.ESTIMATORS)}
    MI = V.MI
    out = {"meta": {"n_pop": C.N_POP, "n_rep": C.N_REP, "m_grid": list(C.M_GRID), "budgets": list(C.SWEEP_BUDGETS), "deltas": list(C.DELTAS),
                    "pop_seed": "3000000 + 1000 * Weltindex + k", "estimators": list(C.ESTIMATORS), "leaf_by_m": {str(m): v for m, v in C.LEAF_BY_M.items()}},
           "welt": {}, "eb_kollaps": {}, "paare": {}}
    for w in C.WELTEN:
        cs = cells[w]
        refs = [V.reference(c["tau"]) for c in cs]
        rho = np.array([float(c["rho"]) for c in cs])
        wd = {"rho": _band(rho), "rho_ueber_grenze": int(np.sum(np.abs(rho) >= C.RHO_LIMIT)) if not np.all(np.isnan(rho)) else None,
              "tau_sd": _band([c["tau"].std() for c in cs]), "tau_mittel": _band([c["tau"].mean() for c in cs]), "est": {}}
        for e in C.ESTIMATORS:
            ed = {"fluch": {}, "faktor": {}, "fang": {}, "netdiff": {}, "kipp": {}}
            for b in C.SWEEP_BUDGETS:
                fl, fa, fg = [], [], []
                for c, ref in zip(cs, refs):
                    a = c["arr"][:, :, EI[e], :].astype(float)                       # (m, rep, metric)
                    v, naiv = a[:, :, MI[f"v{b}"]], a[:, :, MI[f"naiv{b}"]]
                    fl.append(np.nanmean(naiv - v, axis=1) if not np.all(np.isnan(v)) else np.full(len(C.M_GRID), np.nan))
                    fa.append(np.nanmean(naiv, axis=1) / np.nanmean(v, axis=1) if not np.all(np.isnan(v)) else np.full(len(C.M_GRID), np.nan))
                    tb, orak = ref["tbar"], ref["orakel"][b]
                    fg.append(100 * np.nanmean((v - tb) / (orak - tb), axis=1) if orak - tb > 1e-9 and not np.all(np.isnan(v)) else np.full(len(C.M_GRID), np.nan))
                ed["fluch"][str(b)] = [_band(np.array(fl)[:, i]) for i in range(len(C.M_GRID))]
                ed["faktor"][str(b)] = [_band(np.array(fa)[:, i]) for i in range(len(C.M_GRID))]
                ed["fang"][str(b)] = [_band(np.array(fg)[:, i]) for i in range(len(C.M_GRID))]
            for d in C.DELTAS:
                nd, ks = [], []
                for c, ref in zip(cs, refs):
                    a = c["arr"][:, :, EI[e], MI[f"net{d}"]].astype(float)
                    diff = a - ref["beste_feste"][d]
                    nd.append(np.nanmean(diff, axis=1) if not np.all(np.isnan(a)) else np.full(len(C.M_GRID), np.nan))
                    ks.append(V.kipp_point(diff) if not np.all(np.isnan(a)) else None)
                ed["netdiff"][str(d)] = [_band(np.array(nd)[:, i]) for i in range(len(C.M_GRID))]
                if all(k is not None for k in ks):
                    ks = np.array(ks)
                    ed["kipp"][str(d)] = {"median": float(np.median(ks)), "min": int(ks.min()), "max": int(ks.max()), "n_bis_64": int((ks <= 64).sum())}
                else:                                                               # Schrumpfung: bei m = 1 nicht definiert; Kipppunkt nur über definierte Zellen
                    ks2 = []
                    for c, ref in zip(cs, refs):
                        a = c["arr"][:, :, EI[e], MI[f"net{d}"]].astype(float)
                        diff = a - ref["beste_feste"][d]
                        diff[0] = np.nan
                        ks2.append(V.kipp_point(diff))
                    ks2 = np.array(ks2)
                    ed["kipp"][str(d)] = {"median": float(np.median(ks2)), "min": int(ks2.min()), "max": int(ks2.max()), "n_bis_64": int((ks2 <= 64).sum())}
            wd["est"][e] = ed
        out["welt"][w] = wd
        out["eb_kollaps"][w] = [round(float(np.mean([c["collapsed"][mi] for c in cs])), 4) if m >= 2 else None for mi, m in enumerate(C.M_GRID)]
        # gepaarte Unterschiede im Fang-Anteil (Population, Wiederholung gepaart; je Population Mittel über Wiederholungen)
        pw = {}
        for a_, b_ in PAIRS:
            pw[f"{a_}-{b_}"] = {}
            for b in C.BUDGETS:
                dif = []
                for c, ref in zip(cs, refs):
                    tb, orak = ref["tbar"], ref["orakel"][b]
                    if orak - tb <= 1e-9:
                        continue
                    va = c["arr"][:, :, EI[a_], MI[f"v{b}"]].astype(float)
                    vb = c["arr"][:, :, EI[b_], MI[f"v{b}"]].astype(float)
                    dif.append(100 * np.nanmean((va - vb) / (orak - tb), axis=1))
                if dif:
                    dif = np.array(dif)
                    pw[f"{a_}-{b_}"][str(b)] = {"median": [round(float(np.nanmedian(dif[:, i])), 3) if not np.all(np.isnan(dif[:, i])) else None for i in range(len(C.M_GRID))],
                                                "vorn": [int(np.sum(dif[:, i] > 0)) for i in range(len(C.M_GRID))], "n": int(dif.shape[0])}
        out["paare"][w] = pw
    json.dump(out, open(OUT, "w", encoding="utf-8"), separators=(",", ":"), ensure_ascii=False)
    print("geschrieben", OUT, round(os.path.getsize(OUT) / 1024), "KB")


if __name__ == "__main__":
    {"tune": tune, "run": run, "aggregate": aggregate}[sys.argv[1]]()
