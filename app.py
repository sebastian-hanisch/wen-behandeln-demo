"""Wen behandeln? – Auswahl nach dem Piloten und der Gewinner-Fluch - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 13 der Reihe „Experimente und Wirkungsmessung“. Aufbau nach DEMO-PLAYBOOK §2. Lauffähig mit: streamlit run app.py
"""

import numpy as np
import streamlit as st

import wbh_constants as C
import wbh_sweeps as W
from wbh_evaluation import (
    Analysis,
    Settings,
    analyse,
    capture_share,
    verdict_falle,
    verdict_fluch,
    verdict_fluch_table,
    verdict_kipp,
    verdict_kipp_table,
    verdict_kollaps,
    verdict_zoo,
)
from wbh_layout import dunkles_design, kopf_css
from wbh_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from wbh_visualization import build_bars, build_curve, build_rho, build_scatter

st.set_page_config(page_title="Wen behandeln? – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings: Settings) -> Analysis:
    return analyse(settings)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in wbh_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🎯 Wen behandeln? – Auswahl nach dem Piloten und der Gewinner-Fluch")
st.markdown(
    """
Eine neue Bestandsregel B soll nicht überall eingeführt werden, sondern nur an den **Standorten, an denen sie am meisten bringt**. Ein **Pilot** mit wenigen Fenstern je Standort
und Arm misst den Effekt je Standort, die Standorte mit dem größten gemessenen Effekt bekommen die Regel. Das Stück zeigt: Wer so auswählt, **überschätzt** den Effekt der
Gewählten systematisch (**Gewinner-Fluch**), und die gezielte Auswahl nach der Rohdifferenz schlägt „alle“ oder „keinen“ bei den Schwellen ±1 erst ab einem **Kipppunkt** des Pilotumfangs (Modelle über die Merkmale kippen früher; haben alle Standorte denselben Effekt, bis m = 64 nie). Es ist das **dreizehnte Stück** der Reihe
„Experimente und Wirkungsmessung“ und verbindet sie mit der **Optimierung unter Budget** (k von 100 Standorten) und der **Schrumpfung** als Gegenmittel.

Weiter unten: die Experimente **🔬 Gewinner-Fluch**, **🔬 Kipppunkt**, **🔬 Schätzer** und **🔬 Schrumpfung und ihre Grenzen**, die Brücke **⚙️ Auswahl unter Budget als Optimierung**,
der Abschnitt **🚧 Wo die Annahmen enden** und die Expander **💬 In einfachen Worten** (ohne Zahlen und Formeln) und **📐 Mathematische Formulierung**.
"""
)

# erklaerseite:einfach:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/wen-behandeln-erklaert.json, nicht von Hand ändern)
with st.expander("💬 In einfachen Worten"):
    st.markdown(
        """
Eine neue Bestandsregel soll nicht überall eingeführt werden, sondern nur dort, wo sie am meisten bringt. Ein **Pilot** mit wenigen Fenstern je Standort und Arm misst den Effekt je Standort, die Standorte mit dem größten gemessenen Effekt bekommen die Regel. Das hat einen Haken: Wer so auswählt, wählt auch die Standorte, bei denen der Zufall nach oben zeigte, und überschätzt deshalb ihren Effekt (**Gewinner-Fluch**). Die Demo zeigt, wie stark, ab welchem Pilotumfang sich die gezielte Auswahl gegenüber „alle“ oder „keinen“ lohnt und was die **Schrumpfung** der Schätzungen und Modelle über die Merkmale der Standorte dagegen leisten. Die Standorte sind erzeugt, der wahre Effekt ist bekannt; die Aussagen gelten für diese erzeugten Welten.

[Ausführlich und verständlich erklärt, mit Fachteil](https://sebastianhanisch.net/wen-behandeln-erklaert.html)
        """
    )
# erklaerseite:einfach:end

with st.expander("So funktioniert die Auswahl nach dem Piloten", expanded=True):
    st.markdown(
        """
1. **Pilot:** an jedem der 100 Standorte laufen **m Fenster** (in den Welten glatt und Wechselwirkung je Standort 7 bis 224 Tage, sonst 28 Tage) mit Regel A und m Fenster mit Regel B; je Standort und Arm zählt der Mittelwert des Servicegrads.
2. **Schätzen:** die Differenz B minus A je Standort ist die **Rohdifferenz**. Die Alternativen **schrumpfen** sie zum Gesamtmittel (Empirical Bayes) oder **erklären** sie aus den Merkmalen des Standorts
   (Regression, Wald).
3. **Auswählen:** die **k Standorte mit dem größten geschätzten Effekt** bekommen Regel B (Budget). Oder: jeder Standort mit einer Schätzung über der **Behandlungsschwelle** bekommt sie.
4. **Bewerten:** die Standorte sind simuliert, der **wahre Effekt** je Standort ist bekannt (Mittel eines großen Pools von Fenstern). So lässt sich messen, was der Pilot den Gewählten zuschreibt und was sie wirklich bringen.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Wahrheit**")
    welt = st.selectbox("Welt der Standorte", list(range(len(C.WELTEN))), key="welt_select", format_func=lambda i: C.WELT_LABEL[C.WELTEN[i]],
                        help="Wie der wahre Effekt über die 100 Standorte verteilt ist. Bei „Wechselwirkung“ ist die Zuordnung der Regelstärke eine gesetzte Annahme.")
    st.markdown("**Pilot und Schätzer**")
    m = st.select_slider("Pilotumfang m (Fenster je Standort und Arm)", options=list(C.M_GRID), key="m_select",
                         help="Wie viele Fenster der Pilot je Standort und Arm liefert. Mit m = 1 ist die Schrumpfung nicht definiert.")
    est = st.selectbox("Schätzer", list(range(len(C.ESTIMATORS))), key="est_select", format_func=lambda i: C.EST_LABEL[C.ESTIMATORS[i]],
                       help="Wie der Effekt je Standort aus dem Piloten geschätzt wird.")
    st.markdown("**Entscheidung**")
    budget = st.select_slider("Budget (Anteil der Standorte mit Regel B)", options=list(C.BUDGETS), key="budget_select", format_func=lambda b: f"{b} %",
                              help="Die k Standorte mit der größten Schätzung bekommen Regel B (k = Anteil mal 100 Standorte).")
    delta = st.select_slider("Behandlungsschwelle relativ zum mittleren Effekt (Prozentpunkte)", options=list(C.DELTAS), key="delta_select", format_func=lambda d: f"{d:+d}",
                             help="Regel B lohnt sich an einem Standort, wenn ihr Effekt über dem mittleren Effekt aller Standorte plus diesem Wert liegt (Kosten der Umstellung). "
                                  "Bei +1 ist „keinen“ die beste feste Regel, bei −1 „alle“.")
    seed = st.number_input("Zufalls-Seed der Standorte und Piloten", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Standorte und neuer Pilot", width="stretch", on_click=randomize_seed)

sync_query_params({"welt_select": int(welt), "m_select": int(m), "budget_select": int(budget), "delta_select": int(delta), "est_select": int(est), "seed_input": int(seed)})

settings = Settings(int(welt), int(m), int(budget), int(delta), int(est), int(seed))
with st.spinner("Pilot ziehen und Piloten wiederholen …"):
    a = _analysis(settings)
welt_name, est_name = settings.welt_name, settings.est_name
st.markdown(f"**Gewählt:** {C.WELT_LABEL[welt_name]} · {C.EST_LABEL[est_name]} · m = {m} · Budget {budget} % (k = {a.k} von {a.pop.n}) · Zufalls-Seed {seed}")

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
share = float("nan")
if not a.defined:
    st.info("Die **Schrumpfung** ist bei m = 1 nicht definiert: mit einem Fenster je Standort und Arm lässt sich die Streuung je Standort nicht schätzen. Wählen Sie m ≥ 2 oder einen anderen Schätzer.")
else:
    if a.collapsed:
        st.warning("In diesem Piloten ist die geschätzte Streuung der wahren Effekte **null**: die Schrumpfung setzt alle Schätzungen auf das Gesamtmittel, die Auswahl der Gewählten ist **Zufall**.")
    m1, m2, m3, m4 = st.columns(4)
    # Delta immer als „mein Wert minus Referenz“; die Farbe regelt delta_color="inverse" (Vorzeichen-Falle, DEMO-PLAYBOOK §2).
    m1.metric("Pilot sagt (Mittel der Gewählten)", C.fmt_pp0(a.est_mean, 2), delta=f"{C.fmt_pp(a.est_mean - a.true_mean, 2)} gegen wahr", delta_color="inverse")
    m2.metric("Wahr (Mittel der Gewählten)", C.fmt_pp0(a.true_mean, 2), delta=f"{C.fmt_pp(a.true_mean - a.tbar, 2)} gegen Zufallsauswahl", delta_color="normal")
    m3.metric("Zufallsauswahl (alle Standorte)", C.fmt_pp0(a.tbar, 2), delta="Referenz", delta_color="off")
    m4.metric("Orakel (wahre besten k)", C.fmt_pp0(a.oracle_mean, 2), delta="Obergrenze", delta_color="off")
    share = capture_share(a.true_mean, a.tbar, a.oracle_mean)
    c1, c2, c3 = st.columns(3)
    c1.metric("Fang-Anteil (Orakel = 100 %, Zufall = 0 %)", "nicht definiert" if np.isnan(share) else C.fmt_pct(100 * share), delta="Null-Welt: alle Effekte gleich" if np.isnan(share) else None, delta_color="off")
    c2.metric(f"Schwellenregel: behandelte Standorte (Schwelle {delta:+d})", f"{a.n_treated} von {a.pop.n}", delta="Schätzung über der Schwelle", delta_color="off")
    best_fixed = max(a.net_all, 0.0)
    c3.metric("Nettonutzen der Schwellenregel (ein Pilot)", C.fmt_pp0(a.net, 2), delta=f"{C.fmt_pp(a.net - best_fixed, 2)} gegen beste feste Regel", delta_color="normal")
    st.plotly_chart(build_scatter(a), width="stretch", key="main_chart")
    st.caption("Jeder Punkt ist ein Standort: wahrer Effekt (waagerecht), im Piloten geschätzter Effekt (senkrecht). Farbig: die gewählten Standorte. Liegt ihr geschätztes Mittel (rote Linie) "
               "über dem wahren (orange Linie), ist das der **Gewinner-Fluch**: ausgewählt wird, wer im Piloten Glück hatte.")

# --- Kernfragen ----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfragen")
st.markdown(f"**1. Wie stark überschätzt der Pilot die Gewählten?** (Mittel über {C.LIVE_REPS_WALD if est_name == 'twald' else C.LIVE_REPS} neue Piloten auf denselben Standorten)")
kind, text = verdict_fluch(a)
getattr(st, kind)(text)
st.markdown("**2. Lohnt die gezielte Auswahl schon gegenüber „alle“ oder „keinen“?**")
kind, text = verdict_kipp(a)
getattr(st, kind)(text)

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Gewinner-Fluch: je kleiner der Pilot, desto größer die Überschätzung")
st.caption(f"Vorgerechnet: je Welt {C.N_POP} Populationen zu 100 Standorten, je Population {C.N_REP} Piloten je Pilotumfang (tools/messreihe.py). Kurve: Median über die Populationen, Band: Minimum bis Maximum. "
           "Der orange Punkt ist die Live-Rechnung dieser Einstellung (Mittel über die neuen Piloten auf den Standorten der Anzeige).")
live_over = a.reps.get("over", float("nan")) if a.defined else float("nan")
st.plotly_chart(build_curve(C.M_GRID, W.curve(welt_name, est_name, "fluch", budget), "Überschätzung der Gewählten (Prozentpunkte)", live_x=m, live_y=live_over, hline=0.0),
                width="stretch", key="curve_fluch")
kind, text = verdict_fluch_table(welt_name, est_name, budget, m)
getattr(st, kind)(text)

st.markdown("---")

st.subheader("🔬 Kipppunkt: ab welchem Pilotumfang lohnt die Auswahl?")
st.caption("Nettonutzen der Schwellenregel minus beste feste Regel („alle“ oder „keinen“), Prozentpunkte Servicegrad je Standort; über null lohnt die Auswahl. "
           "Kipppunkt einer Population: kleinster Pilotumfang, ab dem die Auswahl für alle größeren dauerhaft besser ist (Mittel über die Piloten größer als zwei Standardfehler). Orange: Live-Rechnung dieser Einstellung.")
live_net = a.reps.get("net_diff", float("nan")) if a.defined else float("nan")
st.plotly_chart(build_curve(C.M_GRID, W.curve(welt_name, est_name, "netdiff", delta), "Nettonutzen gegen beste feste Regel (Prozentpunkte)", live_x=m, live_y=live_net, hline=0.0),
                width="stretch", key="curve_kipp")
kind, text = verdict_kipp_table(welt_name, est_name, delta)
getattr(st, kind)(text)
rows = []
for e in C.ESTIMATORS:
    k = W.kipp(welt_name, e, delta)
    rows.append({"Schätzer": C.EST_SHORT[e], "Kipppunkt (Median, Spanne über die Populationen)": W.kipp_text(k), "Populationen mit Kipppunkt bis m = 64": f"{k['n_bis_64']} von {C.N_POP}"})
st.dataframe(rows, hide_index=True, width="stretch")
st.caption("„über m = 64“: in den gemessenen Pilotumfängen (bis 64 Fenster je Standort und Arm) schlägt die Auswahl die feste Regel nicht dauerhaft. In den Welten glatt, Wechselwirkung und Falle kippt T-linear bei den Schwellen ±1 früher als die Rohdifferenz.")

st.markdown("---")

st.subheader("🔬 Schätzer: welche Modellklasse passt zur Struktur?")
st.caption("Fang-Anteil bei dieser Welt, diesem Pilotumfang und diesem Budget: (wahrer Wert der Auswahl minus Zufall) geteilt durch (Orakel minus Zufall). Median über die Populationen, Balken: Minimum bis Maximum. "
           "Das ist ein **bedingter** Befund: welche Klasse gewinnt, hängt von der Struktur des Effekts ab, wechseln Sie die Welt zwischen „glatt“ und „Wechselwirkung“.")
if welt_name == "homogen":
    st.info("In der Null-Welt ist der Fang-Anteil nicht definiert (das Orakel liegt nicht über dem Zufall).")
else:
    rows_z = W.rank_at(welt_name, m, budget)
    live = (est_name, 100 * share) if a.defined and not np.isnan(share) else None
    st.plotly_chart(build_bars(rows_z, live), width="stretch", key="bars_zoo")
kind, text = verdict_zoo(welt_name, m, budget)
getattr(st, kind)(text)
zrows = []
for w_ in ("glatt", "wechselwirkung"):
    row = {"Welt": C.WELT_KURZ[w_]}
    for bu in C.BUDGETS:
        med, vorn, n = W.pair(w_, "twald", "tlin", bu, m)
        row[f"Budget {bu} %"] = f"{C.fmt_pp(med, 1)} ({vorn} von {n} Populationen mit T-Wald vorn)"
    zrows.append(row)
st.dataframe(zrows, hide_index=True, width="stretch")
st.caption(f"T-Wald minus T-linear im Fang-Anteil bei m = {m} (Median über die Populationen). Die Rangfolge hängt vom Budget ab: in der Welt „Wechselwirkung“ "
           "liegt T-linear bei 10 % Budget bis m = 16 im Median vor T-Wald, bei 30 % Budget liegt T-Wald ab m = 4 in 18 bis 20 von 20 Populationen vorn, bei 50 % Budget ab m = 16 im Median.")

st.markdown("---")

st.subheader("🔬 Schrumpfung und ihre Grenzen")
st.caption("Die Schrumpfung (Empirical Bayes) zieht jede Schätzung zum Gesamtmittel, umso stärker, je ungenauer sie ist. Sie senkt den Gewinner-Fluch, hat aber zwei Grenzen: bei kleinem m ist die geschätzte Streuung oft null, "
           "und wenn Effekt und Streuung zusammenhängen, wählt sie bei kleinem Budget schlechter aus als die Rohdifferenz (vermutlich, weil sie stark streuende Standorte zurückzieht).")
kind, text = verdict_kollaps(welt_name, m)
getattr(st, kind)(text)
st.markdown("**Die Falle: Effekt und Streuung hängen zusammen** (die Standorte der Anzeige)")
if a.pop.welt == "falle":
    st.plotly_chart(build_rho(a.pop), width="stretch", key="rho_chart")
kind, text = verdict_falle(a.pop)
getattr(st, kind)(text)
erows = []
for w_ in C.WELTEN:
    if w_ == "homogen":
        continue
    med, vorn, n = W.pair(w_, "eb", "diff", budget, 16)
    med8, vorn8, _ = W.pair(w_, "eb", "diff", budget, 8)
    rho_band, rho_over = W.rho(w_)
    erows.append({"Welt": C.WELT_KURZ[w_], "Korrelation Effekt/Streuung (Median)": C.fmt_num(rho_band[0], 2), "Schrumpfung minus Rohdifferenz bei m = 8": C.fmt_pp(med8, 0),
                  "Populationen mit Schrumpfung vorn (m = 8)": f"{vorn8} von {n}", "Schrumpfung minus Rohdifferenz bei m = 16": C.fmt_pp(med, 0), "Populationen mit Schrumpfung vorn (m = 16)": f"{vorn} von {n}"})
st.dataframe(erows, hide_index=True, width="stretch")
st.caption(f"Fang-Anteil bei {budget} % Budget, Prozentpunkte (Schrumpfung minus Rohdifferenz, Median über die Populationen). Das Ergebnis hängt am Budget: in der Falle liegt die Schrumpfung bei 10 und 30 % Budget "
           "für jeden gemessenen Pilotumfang in keiner der Populationen vorn, bei 50 % Budget ab m = 16 in 16 bis 20 von 20. "
           "Warum sie bei kleinem Budget schlechter abschneidet, ist eine **Deutung**, keine Messung: Standorte mit großem Effekt streuen stark und werden deshalb am stärksten zurückgezogen.")

st.markdown("---")

# --- Brücke zur Optimierung ----------------------------------------------------------------------------------------------------------------

st.subheader("⚙️ Auswahl unter Budget als Optimierung")
st.markdown(
    r"""
„Wähle $k$ von $N$ Standorten so, dass der Nutzen maximal wird“ ist ein **0-1-Rucksackproblem mit gleichen Kosten** (Kardinalitätsschranke): $\max \sum_s \hat\tau_s x_s$ unter $\sum_s x_s = k$. Die Lösung sind die
$k$ größten Schätzungen. Die Zielfunktion kennt aber nur die **geschätzten** Koeffizienten $\hat\tau_s$; das Optimum eines Problems mit verrauschten Koeffizienten ist **zu optimistisch**: das ist derselbe
Gewinner-Fluch, hier als Lücke zwischen Zielwert und wahrem Wert. Die Schrumpfung ersetzt die Koeffizienten durch ihre bessere Schätzung.
"""
)
brows = []
for key, label in (("diff", "Rohdifferenz"), ("eb", "Schrumpfung (EB)")):
    if key in a.bridge:
        est_v, true_v = a.bridge[key]
        brows.append({"Koeffizienten des Problems": label, "Zielwert des Problems (Summe der Schätzungen)": C.fmt_num(est_v, 1),
                      "wahrer Wert derselben Auswahl (Summe der wahren Effekte)": C.fmt_num(true_v, 1), "Lücke (Optimismus)": C.fmt_num(est_v - true_v, 1)})
    elif key == "eb":
        brows.append({"Koeffizienten des Problems": "Schrumpfung (EB)", "Zielwert des Problems (Summe der Schätzungen)": "nicht definiert (m = 1)",
                      "wahrer Wert derselben Auswahl (Summe der wahren Effekte)": "–", "Lücke (Optimismus)": "–"})
brows.append({"Koeffizienten des Problems": "Orakel (wahre Koeffizienten)", "Zielwert des Problems (Summe der Schätzungen)": C.fmt_num(a.oracle_mean * a.k, 1),
              "wahrer Wert derselben Auswahl (Summe der wahren Effekte)": C.fmt_num(a.oracle_mean * a.k, 1), "Lücke (Optimismus)": C.fmt_num(0.0, 1)})
st.dataframe(brows, hide_index=True, width="stretch")
st.caption(f"Summe über die k = {a.k} gewählten Standorte, Prozentpunkte Servicegrad, ein Pilot (die Anzeige oben). Nicht enthalten: ein Rucksack mit ungleichen Kosten je Standort (hier kostet jede Umstellung gleich viel).")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
rho_g, over_g = W.rho("glatt")
rho_w, over_w = W.rho("wechselwirkung")
_kol = [W.eb_collapse(w_, 2) for w_ in C.WELTEN if w_ != "homogen"]
st.markdown(
    f"""
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Der wahre Effekt je Standort ist das Mittel seines Pools** ({C.POOL_SIZE} vorgerechnete Fenster; Standorte gleichen Typs teilen sich einen Pool) | Der Pilot zieht mit Zurücklegen aus dem Pool: Standorte gleichen Typs haben exakt denselben wahren Effekt. In einem echten Betrieb gibt es keinen Pool, die Wahrheit bleibt unbekannt. | – |
| **Regel B ist an manchen Standorten gleich A** | In der Welt „glatt“ ist z_B an 1 von 9 Werten gleich 1,0 (11,9 % der Standorte der 20 Messpopulationen), in der Welt „Wechselwirkung“ an 46,8 %: dort ist der wahre Effekt exakt null. | – |
| **Die Welt „Wechselwirkung“ ist gesetzt** | Dass der Effekt nur bei zwei zugleich hohen Merkmalen auftritt, ist **eine Annahme dieser Demo**, kein Ergebnis des Vehikels; der Vorteil der Wälder gilt für diese Annahme und nur bei bestimmten Budgets und Pilotumfängen (30 % Budget ab m = 4, 50 % ab m = 16, 10 % erst bei m = 64). | – |
| **Effekt und Streuung sind nur schwach gekoppelt** (Welten „glatt“, „Wechselwirkung“) | Kurze Fenster haben einen etwas kleineren Effekt und eine größere Streuung: die Korrelation ist im Median {C.fmt_num(rho_g[0], 2)} und {C.fmt_num(rho_w[0], 2)}, in allen 40 Populationen negativ, und in {over_g} von {C.N_POP} bzw. {over_w} von {C.N_POP} Populationen mindestens {C.fmt_num(C.RHO_LIMIT, 1)} im Betrag. Die Falle-Welt zeigt, was bei enger Kopplung passiert. | Schrumpfung (Falle) |
| **Der Pilotumfang m ist für alle Standorte und Arme gleich und kostet nichts** | Ein größerer Pilot kostet in der Wirklichkeit Fenster, in denen Regel A oder B an einem Standort gefahren wird; die Demo wägt das nicht ab. | Teststärke, Bandit |
| **Jede Umstellung kostet gleich viel** | Bei ungleichen Kosten je Standort (Rucksack) ist nicht mehr der Effekt allein die Rangfolge; das wurde hier nicht gemessen. | – |
| **Die Einstellungen der Schätzer gelten für die gemessene Regel** | Die Blattgröße des Waldes ist je Pilotumfang auf getrennten Populationen abgestimmt (am Rand des Gitters), die Schrumpfung ist der Momentenschätzer. Aussagen gelten nicht für die ganze Verfahrensklasse. | – |
| **Die Schrumpfung ist bei kleinem m oft nicht verwertbar** | Bei m = 2 ist in {C.fmt_pct(100 * min(_kol))} bis {C.fmt_pct(100 * max(_kol))} der Piloten (je nach Welt, ohne Null-Welt) die geschätzte Streuung null; die Auswahl ist dann Zufall. | – |
| **Ein Budget von 80 % wird nicht angeboten** | Bei der Rohdifferenz lag der Fang-Anteil bei 80 % und kleinem m in der Falle-Welt unter null (schlechter als Zufall), in den anderen Welten nicht; die Ursache ist nicht untersucht. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    # erklaerseite:lesehilfe:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/wen-behandeln-erklaert.json, nicht von Hand ändern)
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind $N$ Standorte mit unbekanntem wahrem Effekt $\tau_s$ der Regel B gegen A (Prozentpunkte Servicegrad) und ein Pilot mit $m$ Fenstern je Standort und Arm. Gesucht ist eine Menge $S$ von $k$ Standorten mit möglichst großem $\sum_{s \in S} \tau_s$. Der Block beschreibt die Auswahl, den Gewinner-Fluch im Normalmodell, die Schrumpfung und die Schwellenregel; die Modellschätzer (Regression, Wald) sind nur benannt, ihre Formeln stehen nicht hier.

| Symbol | Bedeutung |
|---|---|
| $N$, $k$ | Zahl der Standorte (100), Zahl der Gewählten (Budget mal $N$) |
| $\tau_s$ | wahrer Effekt von Standort $s$ (Mittel seines Pools B minus Mittel seines Pools A) |
| $\bar y^A_s$, $\bar y^B_s$ | Pilotmittel der Arme A und B an Standort $s$ über $m$ Fenster |
| $\hat\tau_s$ | geschätzter Effekt (Rohdifferenz oder ein anderer Schätzer) |
| $x_s$ | 1, wenn Standort $s$ Regel B bekommt, sonst 0 |
| $\mu$, $\sigma^2$, $v^2$ | Mittel und Varianz der wahren Effekte, Varianz des Pilotfehlers (Normalmodell) |
| $B$ | Schrumpfungsfaktor im Normalmodell (nicht zu verwechseln mit Regel B) |
| $\hat\mu$, $\hat\sigma^2$, $\hat v_s^2$ | geschätztes Gesamtmittel, geschätzte Streuung der wahren Effekte, geschätzte Varianz der Rohdifferenz an Standort $s$ |
| $\tilde\tau_s$, $\hat B_s$ | geschrumpfte Schätzung und ihr Schrumpfungsfaktor an Standort $s$ |
| $c$ | Behandlungsschwelle: mittlerer wahrer Effekt plus $\delta$ |

[Schritt für Schritt erklärt, mit Herleitung und Literatur](https://sebastianhanisch.net/wen-behandeln-erklaert.html#fachleute)
        """
    )
    # erklaerseite:lesehilfe:end
    st.markdown(
        r"""
**Rohdifferenz.**

$$\hat\tau_s = \bar y^B_s - \bar y^A_s$$

**Auswahl unter Budget** (0-1-Problem mit Kardinalitätsschranke; die Lösung sind die $k$ größten $\hat\tau_s$):

$$\max_{x \in \{0,1\}^N} \; \sum_{s=1}^{N} \hat\tau_s \, x_s \qquad \text{unter} \qquad \sum_{s=1}^{N} x_s = k$$

**Gewinner-Fluch im Normalmodell.** Sind $\tau_s \sim \mathcal N(\mu, \sigma^2)$ und $\hat\tau_s \mid \tau_s \sim \mathcal N(\tau_s, v^2)$, so ist der Erwartungswert des wahren Effekts bei gegebener Schätzung

$$\mathbb E[\tau_s \mid \hat\tau_s] = \mu + B\,(\hat\tau_s - \mu), \qquad B = \frac{\sigma^2}{\sigma^2 + v^2} < 1,$$

und die erwartete Überschätzung der Gewählten ist

$$\mathbb E\Big[\frac1k \sum_{s \in S} (\hat\tau_s - \tau_s)\Big] = (1 - B)\; \mathbb E\Big[\frac1k \sum_{s \in S} (\hat\tau_s - \mu)\Big] > 0 \quad (k < N) .$$

**Schrumpfung (Empirical Bayes).** Mit $\hat\mu$ dem Mittel der $\hat\tau_s$ und $\hat v_s^2$ der geschätzten Varianz der Rohdifferenz an Standort $s$ (aus den Fensterstreuungen, mit 4 Pseudo-Freiheitsgraden zum Mittel gezogen):

$$\tilde\tau_s = \hat\mu + \hat B_s\,(\hat\tau_s - \hat\mu), \qquad \hat B_s = \frac{\hat\sigma^2}{\hat\sigma^2 + \hat v_s^2}, \qquad \hat\sigma^2 = \max\Big\{0,\; \widehat{\mathrm{Var}}(\hat\tau) - \overline{\hat v^2}\Big\}$$

Ist $\hat\sigma^2 = 0$, so sind alle $\tilde\tau_s = \hat\mu$ und die Auswahl ist Zufall. Bei $m = 1$ ist $\hat v_s^2$ nicht schätzbar.

**Schwellenregel und feste Regeln.** Wert je Standort (Nettonutzen):

$$V_{\text{Schwelle}} = \frac1N \sum_s (\tau_s - c)\,\mathbf 1[\hat\tau_s > c], \qquad V_{\text{alle}} = \frac1N \sum_s (\tau_s - c), \qquad V_{\text{keinen}} = 0$$

Der Kipppunkt ist der kleinste Pilotumfang $m$, ab dem $V_{\text{Schwelle}}$ für alle größeren $m$ über $\max\{V_{\text{alle}}, 0\}$ liegt.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Experimente und Wirkungsmessung: A/B bis Logdaten](https://sebastianhanisch.net/konzepte-experimente-wirkungsmessung.html)."
)
