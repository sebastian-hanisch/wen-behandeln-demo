"""Auswahl nach dem Piloten, Wertmaße und die Messzelle (ein Pilot, alle Schätzer).

Zähleinheiten: Standort (Auswahl, N_SITES je Population), Fenster (Pilot: m je Standort und Arm), Wert in Prozentpunkten Servicegrad je behandeltem Standort.

Wertmaße einer Auswahl S (k Standorte) mit wahren Effekten tau_s und Schätzung th_s:
  v      wahrer Mittelwert der Gewählten   mean(tau[S])
  naiv   geschätzter Mittelwert der Gewählten   mean(th[S])   (die Zahl, die der Pilot "ablesen" würde)
  net    Schwellenregel: behandle jeden Standort mit th_s > c, c = mittlerer wahrer Effekt + delta; Nettonutzen (tau_s - c) je Standort,
         verglichen mit der besten festen Regel "alle" (mean(tau) - c) bzw. "keinen" (0)."""

from dataclasses import dataclass

import numpy as np

import wbh_constants as C
import wbh_estimators as E
import wbh_pools as P
from wbh_stream import Stream, mix_seed

METRICS = tuple([f"v{b}" for b in C.SWEEP_BUDGETS] + [f"naiv{b}" for b in C.SWEEP_BUDGETS] + [f"net{d}" for d in C.DELTAS])
MI = {name: i for i, name in enumerate(METRICS)}


def k_of(budget_pct, n=C.N_SITES):
    return max(1, int(round(budget_pct / 100.0 * n)))


def rank_order(th, stream):
    """Absteigende Rangfolge nach dem geschätzten Effekt; Gleichstände (etwa bei vollständiger Schrumpfung) zufällig aufgelöst."""
    jitter = stream.uniform(len(th)) * 1e-9
    return np.argsort(-(th + jitter), kind="stable")


def reference(tau):
    """Größen der Population, die nicht vom Piloten abhängen: Mittel, Orakel je Budget (wahre Rangfolge), Nettonutzen von "alle" je Schwelle."""
    srt = np.sort(tau)[::-1]
    return {
        "tbar": float(tau.mean()),
        "orakel": {b: float(srt[: k_of(b, len(tau))].mean()) for b in C.SWEEP_BUDGETS},
        "alle": {d: float(-d) for d in C.DELTAS},         # mean(tau - c) mit c = mean(tau) + d
        "beste_feste": {d: float(max(-d, 0.0)) for d in C.DELTAS},
    }


def evaluate(tau, th, stream):
    """Vektor der METRICS für eine Schätzung th (n,) bei wahren Effekten tau (n,)."""
    n = len(tau)
    order = rank_order(th, stream)
    out = np.empty(len(METRICS))
    for b in C.SWEEP_BUDGETS:
        sel = order[: k_of(b, n)]
        out[MI[f"v{b}"]] = tau[sel].mean()
        out[MI[f"naiv{b}"]] = th[sel].mean()
    tbar = tau.mean()
    for d in C.DELTAS:
        c = tbar + d
        out[MI[f"net{d}"]] = float(((tau - c) * (th > c)).sum() / n)
    return out


def run_cell(pop, m, estimators, seed_parts):
    """Ein Pilot (m Fenster je Standort und Arm) und alle gewünschten Schätzer. Rückgabe ({name: Vektor der METRICS oder None wenn nicht definiert}, {name: info})."""
    st = Stream(mix_seed(*seed_parts))
    yA, yB = P.draw_pilot(pop, m, st)
    res, infos = {}, {}
    for name in estimators:
        th, info = E.estimate(name, pop.X, yA, yB, st)
        infos[name] = info
        res[name] = evaluate(pop.tau, th, st) if info["defined"] else None
    return res, infos


def capture_share(v, tbar, orakel):
    """Fang-Anteil: (Wert - Zufallsniveau) / (Orakel - Zufallsniveau); nan, wenn das Orakel nicht über dem Zufallsniveau liegt (Null-Welt)."""
    return (v - tbar) / (orakel - tbar) if orakel - tbar > 1e-9 else float("nan")


# ---------------------------------------------------------------------------------------------------------------------------------------
def kipp_point(net_minus_best):
    """Kipppunkt einer Population: kleinster Pilotumfang m, ab dem die Schwellenregel für alle größeren m besser ist als die beste feste Regel
    (Mittel über die Wiederholungen größer als zwei Standardfehler und größer 0); 128 = bis m = 64 nie. Eingabe (m, rep)."""
    better = []
    for mi in range(len(C.M_GRID)):
        x = net_minus_best[mi]
        x = x[~np.isnan(x)]
        if len(x) < 2:
            better.append(False)
            continue
        se = x.std(ddof=1) / np.sqrt(len(x))
        better.append(bool(x.mean() > max(2 * se, 1e-9)))
    for mi in range(len(C.M_GRID)):
        if all(better[mi:]):
            return C.M_GRID[mi]
    return 128


# ---------------------------------------------------------------------------------------------------------------------------------------
# Live-Auswertung für die App: ein Pilot (die Anzeige) und LIVE_REPS weitere Piloten auf denselben Standorten
# ---------------------------------------------------------------------------------------------------------------------------------------
MAIN_TAG = 9_100_000
REP_TAG = 9_200_000


@dataclass(frozen=True)
class Settings:
    welt: int = C.DEFAULT_WELT
    m: int = C.DEFAULT_M
    budget: int = C.DEFAULT_BUDGET
    delta: int = C.DEFAULT_DELTA
    est: int = C.DEFAULT_EST
    seed: int = C.DEFAULT_SEED

    @property
    def welt_name(self):
        return C.WELTEN[self.welt]

    @property
    def est_name(self):
        return C.ESTIMATORS[self.est]


@dataclass(frozen=True)
class Analysis:
    settings: Settings
    pop: P.Population
    defined: bool                 # False: Schrumpfung bei m = 1 (nicht definiert)
    collapsed: bool               # Schrumpfung: Prior-Varianz null, alle Schätzungen gleich
    th: np.ndarray                # Schätzung je Standort (nan, wenn nicht definiert)
    chosen: np.ndarray            # Indizes der Gewählten
    k: int
    est_mean: float               # geschätzter Mittelwert der Gewählten ("was der Pilot abliest")
    true_mean: float              # wahrer Mittelwert der Gewählten
    oracle_mean: float            # wahrer Mittelwert der k besten Standorte
    tbar: float                   # wahrer Mittelwert aller Standorte (Zufallsauswahl)
    net: float                    # Nettonutzen der Schwellenregel (Prozentpunkte je Standort)
    net_all: float                # Nettonutzen von "alle behandeln"
    n_treated: int                # Zahl der Standorte der Schwellenregel
    bridge: dict                  # {"diff"|"eb": (geschätzter Zielwert, wahrer Zielwert) der Auswahl}; "eb" fehlt bei m = 1
    reps: dict                    # Wiederholungen: Mittel von geschätztem und wahrem Wert, Faktor, Nettoabstand, Anzahl


def _mean_se(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def analyse(s: Settings) -> Analysis:
    welt, name = s.welt_name, s.est_name
    pop = P.make_population(welt, s.seed)
    n, k, tau = pop.n, k_of(s.budget, pop.n), pop.tau
    ref = reference(tau)
    st = Stream(mix_seed(MAIN_TAG, s.seed, s.welt, s.m))
    yA, yB = P.draw_pilot(pop, s.m, st)
    th, info = E.estimate(name, pop.X, yA, yB, st)
    bridge = {}
    for other in ("diff", "eb"):
        tho, infoo = (th, info) if other == name else E.estimate(other, pop.X, yA, yB, st)
        if infoo["defined"]:
            sel_o = rank_order(tho, Stream(mix_seed(MAIN_TAG, s.seed, 1, s.m)))[:k]
            bridge[other] = (float(tho[sel_o].sum()), float(tau[sel_o].sum()))
    if not info["defined"]:
        chosen, est_mean, true_mean, net, ntr = np.array([], dtype=int), float("nan"), float("nan"), float("nan"), 0
    else:
        chosen = rank_order(th, Stream(mix_seed(MAIN_TAG, s.seed, 2, s.m)))[:k]
        est_mean, true_mean = float(th[chosen].mean()), float(tau[chosen].mean())
        c = ref["tbar"] + s.delta
        net = float(((tau - c) * (th > c)).sum() / n)
        ntr = int((th > c).sum())
    # Wiederholungen auf denselben Standorten
    R = C.LIVE_REPS_WALD if name == "twald" else C.LIVE_REPS
    est_m, true_m, diffs = [], [], []
    if info["defined"]:
        for r in range(R):
            res, _ = run_cell(pop, s.m, (name,), (REP_TAG, s.seed, s.welt, s.m, r))
            row = res[name]
            est_m.append(row[MI[f"naiv{s.budget}"]] if s.budget in C.SWEEP_BUDGETS else float("nan"))
            true_m.append(row[MI[f"v{s.budget}"]])
            diffs.append(row[MI[f"net{s.delta}"]] - ref["beste_feste"][s.delta])
        em, tm = float(np.mean(est_m)), float(np.mean(true_m))
        dm, dse = _mean_se(diffs)
        reps = {"n": R, "est_mean": em, "true_mean": tm, "over": em - tm, "factor": em / tm if tm > 0.05 else float("nan"),
                "net_diff": dm, "net_diff_se": dse}
    else:
        reps = {"n": 0}
    return Analysis(s, pop, bool(info["defined"]), bool(info.get("collapsed", False)), th, chosen, k, est_mean, true_mean, ref["orakel"].get(s.budget, float("nan")),
                    ref["tbar"], net, ref["alle"][s.delta], ntr, bridge, reps)


def treated_note(a: Analysis):
    """Fang-Anteil der Anzeige-Auswahl (nan bei Null-Welt oder nicht definiert)."""
    return capture_share(a.true_mean, a.tbar, a.oracle_mean) if a.defined else float("nan")


# ---------------------------------------------------------------------------------------------------------------------------------------
# Verdikte: reine Funktionen (Art, Text); die App ruft getattr(st, Art)(Text)
# ---------------------------------------------------------------------------------------------------------------------------------------
FACTOR_WARN = 1.5          # ab diesem Faktor geschätzt/wahr warnt das Verdikt
FACTOR_INFO = 1.1          # darunter gilt die Auswahl als im Mittel nahezu unverzerrt
NET_Z = 2.0                # Nettoabstand der Schwellenregel zur besten festen Regel gilt ab 2 Standardfehlern als Unterschied
NET_MIN = 1e-9             # und nur, wenn er größer als Rundungsrauschen ist (Null-Welt bei delta = 0)
NET_SMALL = 0.05           # unter diesem Betrag (Prozentpunkte je Standort) nennt das Verdikt den Unterschied klein


def best_fixed_name(delta):
    return {-1: "alle behandeln", 0: "alle behandeln (gleich gut wie keinen)", 1: "keinen behandeln"}[delta]


def verdict_fluch(a: Analysis):
    r = a.reps
    if not a.defined:
        return "info", ("Die Schrumpfung ist bei **m = 1** nicht definiert: mit einem Fenster je Standort und Arm gibt es keine Streuung je Standort, die man schätzen könnte. "
                        "Wählen Sie m ≥ 2 oder einen anderen Schätzer.")
    head = (f"Im Piloten erscheinen die **{a.k} Gewählten** mit **{C.fmt_num(r['est_mean'])} Prozentpunkten**, wahr sind es **{C.fmt_pp0(r['true_mean'])}** "
            f"(Mittel über {r['n']} Piloten auf denselben Standorten)")
    if r["true_mean"] <= 0.05 or np.isnan(r["factor"]):
        return "info", head + f": der Pilot liegt um {C.fmt_pp0(r['over'])} über dem wahren Wert."
    f = r["factor"]
    text = head + f": Überschätzung **{C.fmt_pp0(r['over'])}**, Faktor **{C.fmt_num(f)}**."
    if r["over"] < -0.1:
        return "info", head + f": der Pilot **unterschätzt** sie um {C.fmt_pp0(-r['over'])} (Faktor {C.fmt_num(f)}): die Korrektur schießt hier über das Ziel hinaus."
    if f >= FACTOR_WARN:
        return "warning", text + " Wer die Gewählten nach dem Piloten bewertet, sieht Gewinner, die es so nicht gibt."
    if f >= FACTOR_INFO:
        return "info", text
    return "success", text + " Im Mittel nahezu unverzerrt."


def verdict_kipp(a: Analysis):
    s, r = a.settings, a.reps
    if not a.defined:
        return "info", "Die Schrumpfung ist bei **m = 1** nicht definiert; es gibt keine Schwellenregel zu vergleichen."
    d, se = r["net_diff"], r["net_diff_se"]
    fixed = best_fixed_name(s.delta)
    base = f"Die Schwellenregel (behandeln, wo die Schätzung über der Schwelle liegt) gegen die beste feste Regel „{fixed}“: {C.fmt_pp(d, 2)} je Standort (± {C.fmt_num(se, 3)} Standardfehler, {r['n']} Piloten)"
    small = " Der Betrag ist klein (unter 0,05 Prozentpunkte)." if abs(d) < NET_SMALL else ""
    if d > max(NET_Z * se, NET_MIN):
        return "success", base + ". **Die Auswahl lohnt sich schon.**" + small
    if d < -max(NET_Z * se, NET_MIN):
        return "warning", base + ". **Die Auswahl schadet:** besser wäre die feste Regel." + small
    return "info", base + ". Kein Unterschied zu erkennen: **noch kein Gewinn gegenüber der festen Regel.**"


def verdict_kipp_table(welt, est, delta):
    """Kipppunkt aus der vorgerechneten Messreihe (Populationen der Welt)."""
    import wbh_sweeps as W
    k = W.kipp(welt, est, delta)
    name = C.EST_SHORT[est]
    if k["min"] >= 128:
        return "warning", (f"Kipppunkt der {name} in der Welt „{C.WELT_KURZ[welt]}“ bei Schwelle {delta:+d}: **in keiner der {C.N_POP} Populationen bis m = 64**. "
                           "Hier lohnt die gezielte Auswahl in den gemessenen Pilotumfängen nicht.")
    return "info", (f"Kipppunkt der {name} in der Welt „{C.WELT_KURZ[welt]}“ bei Schwelle {delta:+d}: **{W.kipp_text(k)}** "
                    f"(Median und Spanne über {C.N_POP} Populationen; bis m = 64 erreicht in {k['n_bis_64']} von {C.N_POP}). Ab dem Median-Pilotumfang schlägt die Auswahl „alle“ bzw. „keinen“ in der Hälfte der Populationen und bis m = 64 dauerhaft im Sinn von zwei Standardfehlern.")


def verdict_zoo(welt, m, budget):
    """Rangfolge der Schätzer nach Fang-Anteil (Median über die Populationen) bei diesem Pilotumfang und Budget."""
    import wbh_sweeps as W
    if welt == "homogen":
        return "info", ("In der Null-Welt haben alle Standorte denselben Effekt: das Orakel kennt keinen besseren Standort, der Fang-Anteil ist nicht definiert. "
                        "Jede Auswahl hat denselben wahren Wert.")
    rows = W.rank_at(welt, m, budget)
    top, second = rows[0], rows[1]
    text = (f"In der Welt „{C.WELT_KURZ[welt]}“ liegt bei m = {m} und {budget} % Budget **{C.EST_SHORT[top[0]]}** vorn: Fang-Anteil {C.fmt_pct(top[1])} "
            f"(Median über {C.N_POP} Populationen), danach {C.EST_SHORT[second[0]]} mit {C.fmt_pct(second[1])}.")
    if welt == "wechselwirkung":
        med, vorn, n = W.pair(welt, "twald", "tlin", budget, m)
        text += f" T-Wald liegt in {vorn} von {n} Populationen vor T-linear (Median {C.fmt_pp(med, 0)})."
    elif welt == "glatt":
        med, vorn, n = W.pair(welt, "twald", "tlin", budget, m)
        text += f" T-linear liegt in {n - vorn} von {n} Populationen vor T-Wald (T-Wald minus T-linear: Median {C.fmt_pp(med, 0)})."
    return "info", text


def verdict_falle(pop: P.Population):
    """Aussage über die Kopplung von Effekt und Streuung der aktuellen Standorte."""
    rho = pop.rho
    if np.isnan(rho):
        return "info", "In dieser Welt haben alle Standorte denselben Effekt: es gibt keine Kopplung zu messen."
    kind = "warning" if abs(rho) >= C.RHO_STRONG else "info"
    text = f"Korrelation zwischen wahrem Effekt und Streuung der Einzeldifferenz (Pearson) in diesen {pop.n} Standorten: **{C.fmt_num(rho, 2)}**."
    if abs(rho) >= C.RHO_STRONG:
        text += " Effekt und Streuung hängen eng zusammen: wer die Streuung sieht, sieht zugleich den Effekt."
    elif abs(rho) >= C.RHO_LIMIT:
        text += f" Mäßig: nicht unter der Grenze von {C.fmt_num(C.RHO_LIMIT, 1)}, aber weit entfernt von der engen Kopplung der Falle."
    else:
        text += f" Unter {C.fmt_num(C.RHO_LIMIT, 1)}: Effekt und Streuung sind hier nur schwach gekoppelt."
    return kind, text


def verdict_fluch_table(welt, est, budget, m):
    """Überschätzung der Gewählten aus der vorgerechneten Messreihe (Median über die Populationen der Welt)."""
    import wbh_sweeps as W
    mi = C.M_GRID.index(m)
    fl, fa = W.curve(welt, est, "fluch", budget)[mi], W.curve(welt, est, "faktor", budget)[mi]
    name = C.EST_SHORT[est]
    if fl is None:
        return "info", f"{name} ist bei m = {m} nicht definiert."
    text = (f"{name} in der Welt „{C.WELT_KURZ[welt]}“ bei m = {m} und {budget} % Budget: Die Gewählten werden um **{C.fmt_pp0(fl[0])}** überschätzt "
            f"(Median über {C.N_POP} Populationen, Spanne {C.fmt_num(fl[1])} bis {C.fmt_num(fl[2])}), Faktor geschätzt/wahr **{C.fmt_num(fa[0])}**.")
    if fl[0] < -0.1:
        return "info", text.replace("überschätzt", "unterschätzt").replace(f"**{C.fmt_pp0(fl[0])}**", f"**{C.fmt_pp0(-fl[0])}**")
    if fa[0] >= FACTOR_WARN:
        return "warning", text
    if fa[0] >= FACTOR_INFO:
        return "info", text
    return "success", text


def verdict_kollaps(welt, m):
    """Anteil der Piloten, in denen die Schrumpfung wegen Prior-Varianz null alle Schätzungen auf das Gesamtmittel setzt."""
    import wbh_sweeps as W
    r = W.eb_collapse(welt, m)
    if r is None:
        return "info", "Bei m = 1 ist die Schrumpfung nicht definiert (keine Streuung je Standort schätzbar)."
    text = (f"In der Welt „{C.WELT_KURZ[welt]}“ ist bei m = {m} in **{C.fmt_pct(100 * r)}** der Piloten die geschätzte Streuung der wahren Effekte null: "
            "dann liegen alle geschätzten Effekte beim Gesamtmittel, und die Auswahl ist Zufall.")
    return ("warning" if r >= 0.1 else "info"), text
