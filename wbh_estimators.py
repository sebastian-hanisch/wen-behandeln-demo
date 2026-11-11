"""Schätzer des Effekts je Standort aus einem Piloten (m Fenster je Standort und Arm).

Alle Schätzer arbeiten auf den Standort-Mittelwerten (jeder Standort hat gleich viele Fenster, also gleiches Gewicht):
  diff    Rohdifferenz D_s = Mittel(B) - Mittel(A) je Standort
  eb      Schrumpfung zum Gesamtmittel (Empirical Bayes, Normal-Normal, Momentenschätzer der Streuung der wahren Effekte); braucht m >= 2
  tlin    T-linear: je Arm eine lineare Regression des Standortmittels auf die Merkmale, Differenz der beiden Vorhersagen
  tlin2   wie tlin, zusätzlich mit allen paarweisen Produkten der (standardisierten) Merkmale
  twald   T-Wald: je Arm ein Wald (Bagging von Regressionsbäumen, selbst gebaut, numpy), Differenz der beiden Vorhersagen
Alle liefern ein Array (n,) mit dem geschätzten Effekt je Standort in Prozentpunkten; `estimate` gibt zusätzlich ein Info-Wörterbuch zurück."""

import itertools

import numpy as np

import wbh_constants as C


def ols(Z, y):
    beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    return beta


def moderated_se2(yA, yB, d0=C.EB_PRIOR_DF):
    """Geschätzte Varianz der Standortdifferenz je Standort: aus den Fensterstreuungen, mit d0 Pseudo-Freiheitsgraden zum Mittel gezogen. Braucht m >= 2."""
    m = yA.shape[1]
    s2 = yA.var(axis=1, ddof=1) + yB.var(axis=1, ddof=1)         # Varianz der Differenz je Fenster-Paar
    pooled = s2.mean()
    s2m = ((2 * (m - 1)) * s2 + d0 * pooled) / (2 * (m - 1) + d0)
    return s2m / m


def eb_shrink(D, se2):
    """Empirical Bayes: D_s -> mu + B_s (D_s - mu) mit mu = Mittel der D, B_s = tau2 / (tau2 + se2_s), tau2 = max(0, Var(D) - Mittel(se2))."""
    mu = D.mean()
    tau2 = max(0.0, float(D.var(ddof=1) - se2.mean()))
    B = tau2 / (tau2 + se2) if tau2 > 0 else np.zeros_like(D)
    return mu + B * (D - mu), tau2


# ---------------------------------------------------------------------------------------------------------------------------------------
# Regressionsbaum und Wald (numpy)
# ---------------------------------------------------------------------------------------------------------------------------------------
def tree_fit_predict(Xtr, ytr, Xpred, min_leaf):
    """Regressionsbaum (Kriterium: Quadratsummen-Reduktion, gierig, alle Merkmale, Blattgröße mindestens min_leaf Zeilen), Vorhersage für Xpred.

    Geteilt wird an der Mitte zwischen zwei verschiedenen benachbarten Werten (x <= Schwelle geht nach links). Bei Gleichständen in der Reduktion gewinnt das
    erste Merkmal und innerhalb eines Merkmals die kleinste Schwelle."""
    pred = np.zeros(len(Xpred))

    def grow(tr, pr):
        ys = ytr[tr]
        n = len(tr)
        best_gain, best = 0.0, None
        if n >= 2 * min_leaf:
            nl = np.arange(min_leaf, n - min_leaf + 1)
            nr = n - nl
            for j in range(Xtr.shape[1]):
                order = np.argsort(Xtr[tr, j], kind="stable")
                xs = Xtr[tr, j][order]
                cs = np.cumsum(ys[order])
                tot = cs[-1]
                gain = nl * nr / n * (cs[nl - 1] / nl - (tot - cs[nl - 1]) / nr) ** 2
                gain = np.where(xs[nl - 1] < xs[nl], gain, -1.0)
                i = int(np.argmax(gain))
                if gain[i] > best_gain + 1e-15:
                    best_gain = float(gain[i])
                    best = (j, 0.5 * (xs[nl[i] - 1] + xs[nl[i]]))
        if best is None:
            pred[pr] = ys.mean()
            return
        j, thr = best
        left_tr = Xtr[tr, j] <= thr
        left_pr = Xpred[pr, j] <= thr
        grow(tr[left_tr], pr[left_pr])
        grow(tr[~left_tr], pr[~left_pr])

    grow(np.arange(len(ytr)), np.arange(len(Xpred)))
    return pred


def forest_predict(X, y, stream, min_leaf, n_trees=C.N_TREES):
    """Wald = Mittel aus n_trees Bäumen auf Bootstrap-Stichproben der Standorte (Zurücklegen); Vorhersage für alle Standorte. Die Blattgröße zählt
    Bootstrap-Zeilen einschließlich Duplikaten."""
    n = len(y)
    out = np.zeros(n)
    for _ in range(n_trees):
        rows = stream.integers(n, n)
        out += tree_fit_predict(X[rows], y[rows], X, min_leaf)
    return out / n_trees


def leaf_for(m):
    """Blattgröße des Waldes je Pilotumfang (abgestimmt auf getrennten Seeds, tools/messreihe.py tune; Tabelle in wbh_constants.LEAF_BY_M)."""
    if m in C.LEAF_BY_M:
        return C.LEAF_BY_M[m]
    return C.LEAF_BY_M[max(k for k in C.LEAF_BY_M if k <= m)]      # zwischen den Stufen der Regler: die nächstkleinere Stufe


# ---------------------------------------------------------------------------------------------------------------------------------------
def estimate(name, X, yA, yB, stream=None):
    """Schätzung des Effekts je Standort. Rückgabe (theta, info); info enthält bei "eb" `tau2` und `collapsed` (Prior-Varianz null),
    bei m = 1 ist "eb" nicht definiert (theta = nan, info["defined"] = False)."""
    mA, mB = yA.mean(axis=1), yB.mean(axis=1)
    D = mB - mA
    n = len(D)
    m = yA.shape[1]
    info = {"defined": True}
    if name == "diff":
        return D, info
    if name == "eb":
        if m < 2:
            return np.full(n, np.nan), {"defined": False, "collapsed": False, "tau2": float("nan")}
        th, tau2 = eb_shrink(D, moderated_se2(yA, yB))
        return th, {"defined": True, "collapsed": tau2 == 0.0, "tau2": tau2}
    if name == "tlin":
        Z = np.column_stack([np.ones(n), X])
        return Z @ ols(Z, mB) - Z @ ols(Z, mA), info
    if name == "tlin2":
        Xc = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
        prods = [Xc[:, i] * Xc[:, j] for i, j in itertools.combinations(range(X.shape[1]), 2)]
        Z = np.column_stack([np.ones(n), Xc] + prods)
        return Z @ ols(Z, mB) - Z @ ols(Z, mA), info
    if name == "twald":
        ml = leaf_for(m)
        return forest_predict(X, mB, stream, ml) - forest_predict(X, mA, stream, ml), info
    raise ValueError(name)
