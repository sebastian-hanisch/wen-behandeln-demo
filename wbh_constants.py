"""Konstanten: Vehikel, Welten, Regler-Grenzen, Farben, Presets und Zahlenformate.

Das Vehikel (ein Standort-Fenster mit täglicher Nachbestellung) ist das der ganzen Linie "Experimente und Wirkungsmessung"; die ersten Konstanten
(BASE_MEAN bis Z_A) sind wörtlich die aus ab-test-demo und werden von der kopierten `wbh_vehicle.py` gelesen."""

# ---------------------------------------------------------------------------------------------------------------------------------------
# Vehikel (wie ab-test-demo)
# ---------------------------------------------------------------------------------------------------------------------------------------
BASE_MEAN = 20.0          # mittlere Tagesnachfrage des Standardstandorts (Stück)
DISP = 5                  # Dispersion der negativ-binomialen Tagesnachfrage (ganzzahlig: Gamma(5) = Summe von 5 Exponentialzahlen)
LEAD = 2                  # Wiederbeschaffungszeit in Tagen
BURN_DAYS = 28            # Einschwingtage vor dem Auswertefenster
WINDOW_DAYS = 28          # Standard-Auswertefenster (vier Wochen)
Z_A = 1.0                 # Sicherheitsfaktor der bestehenden Regel A
Z_B = 1.4                 # Sicherheitsfaktor der neuen Regel B in der Falle- und der Null-Welt

# ---------------------------------------------------------------------------------------------------------------------------------------
# Standorte, Pools und Welten
# ---------------------------------------------------------------------------------------------------------------------------------------
N_SITES = 100                                   # Standorte je Population (Zähleinheit der Auswahl)
POOL_SIZE = 2000                                # vorgerechnete Fenster je Pool (Standorttyp, Regel, Fensterlänge)
Z_GRID = tuple(round(1.0 + 0.1 * i, 1) for i in range(9))      # Sicherheitsfaktoren der Regel B in den Welten glatt und Wechselwirkung
LENGTHS = (7, 14, 28, 56, 112, 224)             # Länge des Auswertefensters je Standort in Tagen (nach 28 Einschwingtagen)
FALLE_LEADS = (1, 2, 3, 4)                      # Welt "falle": Standorttypen (Wiederbeschaffungszeit, Dispersion, Tagesmittel)
FALLE_DISPS = (2, 3, 5, 8, 15, 40)
FALLE_MEANS = (8.0, 14.0, 20.0, 26.0, 32.0, 40.0)
WELTEN = ("glatt", "wechselwirkung", "homogen", "falle")
WELT_LABEL = {
    "glatt": "Glatt: der Effekt wächst mit einem beobachteten Merkmal",
    "wechselwirkung": "Wechselwirkung: Effekt nur, wenn zwei Merkmale zugleich hoch sind",
    "homogen": "Null-Welt: alle Standorte haben denselben Effekt",
    "falle": "Falle: Effekt und Streuung hängen zusammen",
}
WELT_KURZ = {"glatt": "glatt", "wechselwirkung": "Wechselwirkung", "homogen": "Null-Welt", "falle": "Falle"}
WELT_FEATURES = {
    "glatt": ("Sicherheitsfaktor z_B", "Rotheringe 1", "Rotheringe 2", "Rotheringe 3"),
    "wechselwirkung": ("Merkmal u", "Merkmal v", "Rotheringe 1", "Rotheringe 2"),
    "homogen": ("Rotheringe 1", "Rotheringe 2", "Rotheringe 3", "Rotheringe 4"),
    "falle": ("Wiederbeschaffungszeit", "ln(Dispersion)", "Tagesmittel", "Rotheringe"),
}
RHO_STRONG = 0.8                                # ab diesem Betrag der Korrelation gilt die Kopplung als eng
RHO_LIMIT = 0.3                                 # Betrag der Pearson-Korrelation, unter dem Effekt und Streuung als nur schwach gekoppelt gelten

# ---------------------------------------------------------------------------------------------------------------------------------------
# Schätzer, Pilotumfänge, Budgets, Schwellen
# ---------------------------------------------------------------------------------------------------------------------------------------
ESTIMATORS = ("diff", "eb", "tlin", "tlin2", "twald")
EST_LABEL = {
    "diff": "Rohdifferenz (Mittel B minus Mittel A je Standort)",
    "eb": "Schrumpfung (Empirical Bayes)",
    "tlin": "Regression über die Merkmale (T-linear)",
    "tlin2": "Regression mit Produkttermen",
    "twald": "Wald über die Merkmale (T-Wald)",
}
EST_SHORT = {"diff": "Rohdifferenz", "eb": "Schrumpfung (EB)", "tlin": "T-linear", "tlin2": "T-linear + Produkte", "twald": "T-Wald"}
M_GRID = (1, 2, 4, 8, 16, 32, 64)                # Fenster je Standort und Arm im Piloten
BUDGETS = (10, 30, 50)                           # angebotene Budgetanteile in Prozent der Standorte
SWEEP_BUDGETS = (10, 30, 50, 80)                 # gemessene Budgetanteile (80 % wird nicht angeboten, siehe README)
DELTAS = (-1, 0, 1)                              # Behandlungsschwelle relativ zum mittleren Effekt in Prozentpunkten
EB_PRIOR_DF = 4.0                                # Pseudo-Freiheitsgrade, mit denen die Varianz je Standort zum Mittel gezogen wird
N_TREES = 40                                     # Bäume je Wald (T-Wald)
LEAF_GRID = (2, 5, 10)                           # Blattgrößen, unter denen je Pilotumfang abgestimmt wird

# Messreihe (vorgerechnet, tools/messreihe.py): Populationen und Wiederholungen
N_POP = 20                                       # Populationen je Welt
N_REP = 50                                       # Wiederholungen des Piloten je Population und Pilotumfang
LIVE_REPS = 40                                   # Wiederholungen der Live-Rechnung (gleiche Standorte, neue Piloten)
LIVE_REPS_WALD = 10                              # beim Wald (teuer) weniger

# ---------------------------------------------------------------------------------------------------------------------------------------
# Regler
# ---------------------------------------------------------------------------------------------------------------------------------------
SEED_MAX, DEFAULT_SEED = 999999, 35
DEFAULT_WELT, DEFAULT_M, DEFAULT_BUDGET, DEFAULT_DELTA, DEFAULT_EST = 0, 1, 10, 1, 0

COLOR_CHOSEN = "#2F6B65"
COLOR_OTHER = "rgba(150,150,150,0.55)"
COLOR_LINE = "#4c78a8"
COLOR_TRUTH = "#d95f02"
COLOR_WARN = "#c0392b"

PRESETS = {
    "Gewinner-Fluch (Voreinstellung)": {"welt": 0, "m": 1, "budget": 10, "delta": 1, "est": 0, "seed": DEFAULT_SEED},
    "Größerer Pilot": {"welt": 0, "m": 32, "budget": 10, "delta": 1, "est": 0, "seed": DEFAULT_SEED},
    "Nur Rauschen (Null-Welt)": {"welt": 2, "m": 1, "budget": 10, "delta": 1, "est": 0, "seed": DEFAULT_SEED},
    "Schrumpfung als Gegenmittel": {"welt": 0, "m": 16, "budget": 10, "delta": 1, "est": 1, "seed": DEFAULT_SEED},
    "Wechselwirkung: Wälder": {"welt": 1, "m": 8, "budget": 30, "delta": -1, "est": 4, "seed": DEFAULT_SEED},
    "Die Falle: Streuung verrät den Effekt": {"welt": 3, "m": 8, "budget": 10, "delta": 1, "est": 1, "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Gewinner-Fluch (Voreinstellung)": "Glatte Welt, ein Fenster je Standort und Arm, die zehn Standorte mit der größten gemessenen Differenz werden gewählt: der Pilot überschätzt sie stark.",
    "Größerer Pilot": "Dieselbe Welt mit 32 Fenstern je Standort und Arm: die Überschätzung der Gewählten ist deutlich kleiner.",
    "Nur Rauschen (Null-Welt)": "Alle Standorte haben denselben Effekt. Die zehn Gewählten wirken im Piloten trotzdem weit über dem Durchschnitt: reines Rauschen.",
    "Schrumpfung als Gegenmittel": "Glatte Welt, 16 Fenster je Standort und Arm: die Schrumpfung zum Gesamtmittel (Empirical Bayes) senkt die Überschätzung der Gewählten.",
    "Wechselwirkung: Wälder": "Der Effekt entsteht nur, wenn zwei Merkmale zugleich hoch sind (gesetzte Annahme): bei 30 % Budget und m = 8 ordnet ein Wald über die Merkmale besser als eine Gerade.",
    "Die Falle: Streuung verrät den Effekt": "Effekt und Streuung der Fenster hängen im Vehikel eng zusammen: die Schrumpfung, die je Standort nach der Streuung zieht, schneidet hier bei 10 % Budget schlechter ab als die Rohdifferenz.",
}


# ---------------------------------------------------------------------------------------------------------------------------------------
# Zahlenformate (Dezimalkomma)
# ---------------------------------------------------------------------------------------------------------------------------------------
def fmt_num(x, digits=1):
    return f"{x:.{digits}f}".replace(".", ",")


def fmt_pp(x, digits=1):
    """Prozentpunkte (die Werte liegen schon in Prozentpunkten), mit Vorzeichen: '+1,5 Prozentpunkte'."""
    return f"{x:+.{digits}f}".replace(".", ",") + " Prozentpunkte"


def fmt_pp0(x, digits=1):
    """Prozentpunkte ohne Vorzeichen: '1,5 Prozentpunkte'."""
    return f"{x:.{digits}f}".replace(".", ",") + " Prozentpunkte"


def fmt_pct(x, digits=0):
    """Zahl, die schon in Prozent steht, mit Prozentzeichen: 63 -> '63 %'."""
    return f"{x:.{digits}f}".replace(".", ",") + " %"

# Blattgröße des Waldes je Pilotumfang (abgestimmt auf getrennten Populationen und Zufalls-Seeds, tools/messreihe.py tune)
LEAF_BY_M = {1: 10, 2: 10, 4: 10, 8: 10, 16: 10, 32: 10, 64: 5}
