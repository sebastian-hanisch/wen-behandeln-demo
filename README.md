# Wen behandeln? – Auswahl nach dem Piloten und der Gewinner-Fluch – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-wen-behandeln-demo.streamlit.app/)** -->

Interaktive Demo zur **Auswahl der Standorte, an denen eine neue Regel eingeführt wird**. **Dreizehntes Stück der Konzepte-Linie „Experimente und Wirkungsmessung“** im Portfolio von
[Sebastian Hanisch](https://sebastianhanisch.net) (Operations Research und Machine Learning): ein Verfahren, ein wachsendes Beispiel, jedes Stück hebt eine Annahme auf. Die Vorgänger fragen, ob
Regel B **im Mittel** wirkt ([ab-test-demo](https://github.com/sebastian-hanisch/ab-test-demo), [test-power-demo](https://github.com/sebastian-hanisch/test-power-demo)). Hier wirkt sie
**an jedem Standort anders**, und ein **Pilot** mit wenigen Fenstern je Standort und Arm entscheidet, welche Standorte sie bekommen: die Kante vom Mittelwert zur Auswahl (die
Schwäche des Vorgängers: ein Mittelwert sagt nicht, **wo** die Regel wirkt).

Vehikel ist wie in der ganzen Linie ein **Standort mit täglicher Nachbestellung** (Base-Stock-Regel, Zentrallager unbegrenzt); Kennzahl ist der **Servicegrad**, eine Einheit des Experiments ein
**Standort-Fenster**. Regel A fährt den Sicherheitsfaktor 1,0, Regel B einen größeren oder (an manchen Standorten) denselben: in der Welt glatt ist z_B an 1 von 9 Werten gleich 1,0 (11,9 % der Standorte der 20 Messpopulationen), in der Welt Wechselwirkung an 46,8 %; dort sind Regel A und B gleich und der wahre Effekt exakt null. Die **Standorte sind simuliert**, ihr **wahrer Effekt** ist das Mittel eines großen Pools vorgerechneter
Fenster und damit bekannt: so lässt sich messen, was der Pilot den gewählten Standorten zuschreibt und was sie wirklich bringen.

## Warum dieses Problem

Wer aus dem Piloten die Standorte mit dem **größten gemessenen Effekt** auswählt, wählt zugleich die, bei denen das Rauschen nach oben zeigte: der geschätzte Effekt der Gewählten ist **systematisch zu groß**
(Gewinner-Fluch, Fluch des Optimierers). Und die gezielte Auswahl nach der Rohdifferenz schlägt bei den Schwellen ±1 „alle behandeln“ oder „keinen behandeln“ **erst ab einem Pilotumfang** (Kipppunkt); davor ist sie nicht besser, bei kleinem m im Median schlechter. Ein Modell über die Merkmale kippt in den Welten glatt und Wechselwirkung schon bei m = 1. Die Kernfrage der Demo:
**Wie stark überschätzt der Pilot die Gewählten, ab wann lohnt die Auswahl, und was hilft?** Gegenmittel sind die **Schrumpfung** der Schätzungen (Empirical Bayes) und Modelle über die Standortmerkmale; welches Modell
passt, hängt von der Struktur des Effekts ab (bedingter Befund, kein eigener Haken).

## Modell

- **Vehikel:** Gamma-Poisson-Nachfrage, Bestellung täglich bis zum Niveau `S = round(L·μ + z·σ_L)`, ein Fenster liefert den Servicegrad Typ 2 (Quote der Summen im Fenster nach 28 Einschwingtagen). Die Regeln
  stammen wörtlich aus [ab-test-demo](https://github.com/sebastian-hanisch/ab-test-demo) (`wbh_vehicle.py`, `wbh_rng.py`, `wbh_formulas.py` sind kopiert, `tests/test_copies.py` prüft die Gleichheit); `wbh_sim.py`
  rechnet dieselben Gleichungen vektorisiert; Tagesmittel, Dispersion, Wiederbeschaffungszeit und Fensterlänge sind dort je Pool einstellbar. Die Welten nutzen davon nur: glatt und Wechselwirkung Sicherheitsfaktor z_B und Fensterlänge (Standardstandort: Tagesmittel 20, Dispersion 5, Wiederbeschaffungszeit 2), Falle Wiederbeschaffungszeit, Dispersion und Tagesmittel, Null-Welt nichts.
- **Pools und Wahrheit:** je (Standorttyp, Regel, Fensterlänge) ein Pool von 2 000 unabhängigen Fenstern (`precomputed_pools.npz`: 342 Pools, SplitMix64, je Pool ein eigener Strom, `generate_precomputed.py`).
  **Wahrheit = Poolmittel**: der wahre Effekt eines Standorts ist Mittel(B) minus Mittel(A) seiner Pools in Prozentpunkten Servicegrad je Fenster. Standorte gleichen Typs teilen sich einen Pool und haben
  denselben wahren Effekt. Der **Pilot** zieht m Fenster je Standort und Arm mit Zurücklegen aus den Pools.
- **Vier Welten** (100 Standorte, ganz aus dem Zufalls-Seed erzeugt, `wbh_pools.py`):

  | Welt | Was sich je Standort ändert |
  |---|---|
  | **glatt** | Standardstandorte, Regel B fährt je Standort einen beobachteten Sicherheitsfaktor z_B ∈ {1,0; 1,1; …; 1,8}; die Fensterlänge je Standort ist zufällig aus {7, 14, 28, 56, 112, 224} Tagen und unabhängig von z_B gezogen (kurze Fenster haben allerdings einen etwas kleineren wahren Effekt und eine größere Streuung, daher ist die Korrelation von Effekt und Streuung im Median −0,23, nicht null) |
  | **Wechselwirkung** | z_B = 1,0 + 0,8·s(u)·s(v) mit zwei beobachteten Merkmalen u, v (s = steile Sigmoide um 0,5): großer Effekt nur, wenn u **und** v hoch sind. **Gesetzte Zuordnung der Regelstärke, kein Ergebnis des Vehikels** |
  | **Null-Welt (homogen)** | alle Standorte gleich (Standardstandort, z_B = 1,4, Fenster 28 Tage), alle Merkmale Rotheringe |
  | **Falle** | Vehikel pur: Wiederbeschaffungszeit 1 bis 4, Dispersion 2 bis 40, Tagesmittel 8 bis 40 je Standort aus Gittern, Fenster 28 Tage; Effekt und Streuung der Fenster hängen eng zusammen |

  Beobachtet wird je Standort ein Merkmalsvektor mit 4 Spalten (die Merkmale der Welt, der Rest Rotheringe aus der Standardnormalverteilung).
- **Schätzer** (`wbh_estimators.py`, alle auf den Standort-Mitteln): **Rohdifferenz** (Mittel B minus Mittel A); **Schrumpfung** (Empirical Bayes, Normal-Normal, Momentenschätzer der Streuung der wahren Effekte, Varianz je
  Standort mit 4 Pseudo-Freiheitsgraden zum Mittel gezogen; braucht m ≥ 2); **T-linear** (je Arm eine Regression des Standortmittels auf die Merkmale, Differenz der Vorhersagen); **T-linear mit Produkttermen**
  (alle paarweisen Produkte der standardisierten Merkmale); **T-Wald** (je Arm ein Wald aus 40 selbst gebauten Regressionsbäumen auf Bootstrap-Stichproben der Standorte, Blattgröße zählt Bootstrap-Zeilen). Die
  Blattgröße des Waldes ist je Pilotumfang auf **getrennten Populationen und Zufalls-Seeds** abgestimmt (Gitter {2, 5, 10}, Kriterium Wert bei 30 % Budget, Mittel über die Welten glatt, Wechselwirkung und Falle):
  10 für m ≤ 32, 5 für m = 64; der Wert liegt bei 6 von 7 Pilotumfängen am Rand des Gitters (`tools/tuning.json`).
- **Auswahl und Wertmaße:** Rangfolge absteigend nach der Schätzung, Gleichstände zufällig. **Budget** = die k Standorte mit der größten Schätzung (k = Anteil mal 100). **Wert** = Mittel der **wahren** Effekte der
  Gewählten; **Fang-Anteil** = (Wert − Zufallsniveau) / (Orakel − Zufallsniveau), Zufallsniveau = mittlerer wahrer Effekt aller Standorte, Orakel = die k wahrhaft besten. **Gewinner-Fluch** = geschätzter minus wahrer
  Mittelwert der Gewählten (Prozentpunkte) und ihr Quotient (Faktor, Verhältnis der Mittel über die Piloten). **Schwellenregel**: behandle jeden Standort mit einer Schätzung über c = mittlerer wahrer Effekt + δ
  (δ = −1, 0, +1 Prozentpunkte); Nettonutzen (τ − c) je Standort, verglichen mit der besten festen Regel („alle“: mittlerer wahrer Effekt minus c, „keinen“: 0). **Kipppunkt einer Population** = kleinster
  Pilotumfang m, ab dem die Schwellenregel für **alle größeren m** besser ist als die beste feste Regel (Mittel über die Piloten größer als zwei Standardfehler und größer als Rundungsrauschen); „über m = 64“ heißt: bis m = 64 nie.
- **Zähleinheiten:** Standort (Auswahl, 100 je Population), Fenster (Pilot: m je Standort und Arm), Wert in Prozentpunkten Servicegrad je behandeltem Standort; Faktoren sind Verhältnisse zweier Mittel, keine
  Mittel von Verhältnissen.
- **Vorgerechnete Messreihe** (`tools/messreihe.py`, `precomputed_sweeps.json`): je Welt **20 Populationen** zu 100 Standorten (Zufalls-Seed 3 000 000 + 1 000 · Weltindex + k), je Population und Pilotumfang
  m ∈ {1, 2, 4, 8, 16, 32, 64} **50 Piloten** (Zufalls-Seed `(70 000 000, Weltindex, m, k, r)`), **28 000 Piloten** (4 · 20 · 7 · 50; eine Zelle ist ein Pilot), je Pilot alle fünf Schätzer: 140 000 Auswertungen (136 000 definierte, die Schrumpfung ist bei m = 1 nicht definiert). Angegeben ist der **Median [Minimum bis Maximum] über die 20 Populationen** (nicht bei den Kollaps-Anteilen und den Zählungen „x von 20“)
  (je Population das Mittel über 50 Piloten).
- **Live in der App:** ein Pilot (die Anzeige) und **40 weitere Piloten auf denselben Standorten** (beim Wald 10, weil er teuer ist); die Kurven der Experimente sind die vorgerechnete Messreihe, der orange Punkt ist die Live-Rechnung.

## Methodik

Die App zeigt zuerst den einen Piloten: wahrer gegen geschätzter Effekt je Standort, die Gewählten farbig, dazu die Kennzahlen (Pilot sagt / wahr / Zufall / Orakel, Fang-Anteil, Schwellenregel). Dann zwei Kernfragen mit
Verdikt (Überschätzung, lohnt die Auswahl schon?) und die Experimente: **Gewinner-Fluch über den Pilotumfang**, **Kipppunkt**, **Schätzer** (bedingter Befund über Welt und Budget), **Schrumpfung und ihre Grenzen** (Kollaps der
Prior-Varianz, Falle) und die Brücke **Auswahl unter Budget als Optimierung**. Die Seitenleiste enthält Presets (Schnellstart) und die sechs Regler Welt, Pilotumfang, Schätzer, Budget, Behandlungsschwelle und Zufalls-Seed; jeder
Regler verändert das Hauptergebnis (`tests/test_app.py`).

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: Median [Minimum bis Maximum] über 20 Populationen zu 100 Standorten, je Population Mittel über 50 Piloten, Budget 10 % (zehn Standorte), wenn nichts anderes steht; die Spalte „Test“ nennt die Prüfung in
`tests/test_claims.py` (neu gemessen = Welten glatt und Falle, Rohdifferenz und T-linear, mit denselben Seeds; sonst: gegen `precomputed_sweeps.json`, **Messung dieser README, nicht neu gemessen**; T-Wald nur so).

| Frage | Befund | Test |
|---|---|---|
| Wie stark überschätzt der Pilot die Gewählten (Rohdifferenz, Faktor geschätzt/wahr)? | Welt glatt: bei **m = 1** Faktor **4,9 [4,3 bis 5,7]**, bei m = 8 **2,0 [1,8 bis 2,2]**, bei m = 64 **1,2 [1,1 bis 1,2]**. Welt Falle: **4,4 [4,3 bis 4,6]**, **2,1 [2,0 bis 2,2]**, **1,3 [1,3 bis 1,4]**. In der Welt Wechselwirkung (kleinerer mittlerer Effekt) bei m = 1 **7,5 [5,4 bis 9,1]**. | `test_winners_curse_factor_of_the_ten_chosen_with_the_raw_difference`, `test_remeasure_rohdifferenz_and_t_linear_…[glatt, falle]` |
| Wie viele Prozentpunkte sind das? | Welt glatt: **12,7 [11,3 bis 14,1]** Prozentpunkte bei m = 1, **3,8 [3,4 bis 4,3]** bei m = 8, **0,9 [0,7 bis 1,0]** bei m = 64; die Überschätzung sinkt in jeder Welt mit jedem Pilotumfang. | `test_winners_curse_in_percentage_points_and_in_the_null_world`, `test_the_winners_curse_shrinks_monotonically_…` |
| Und wenn alle Standorte gleich sind? | Null-Welt, m = 1: die zehn „Gewinner“ liegen **11,7 [11,3 bis 12,2]** Prozentpunkte über dem wahren Effekt, der für alle Standorte 3,4 beträgt: die „Entdeckung“ ist reines Rauschen. | dasselbe |
| Ab welchem Pilotumfang lohnt die Auswahl (Rohdifferenz, Schwelle +1)? | **Kipppunkt** Welt glatt **m = 16 (Spanne m = 8 bis m = 32)**, Wechselwirkung **m = 4 oder m = 8** (je 10 von 20 Populationen), Falle **m = 64 (Spanne m = 32 bis über m = 64)**, bis m = 64 erreicht in **17 von 20** Populationen. Schwelle −1: glatt m = 8 (Spanne m = 4 bis m = 16); Falle m = 64 (Spanne m = 64 bis über m = 64), erreicht in **14 von 20**. Bei Schwelle 0 (alle und keinen gleich gut) liegt die Rohdifferenz in allen drei Welten schon bei m = 1 vorn (Nettoabstand +0,21 / +0,24 / +0,10 Prozentpunkte je Standort in glatt / Wechselwirkung / Falle); bei Schwelle +1 und m = 1 liegt sie in allen drei Welten darunter (−0,22 / −0,17 / −0,25). | `test_kipppunkt_of_the_raw_difference_at_threshold_plus_one`, `test_remeasure_…` |
| Und mit einem Modell über die Merkmale (T-linear)? | Kipppunkt Welt glatt m = 1 (Spanne m = 1 bis m = 2), Wechselwirkung m = 1, Falle m = 8 (Spanne m = 4 bis m = 16); bei den Schwellen ±1 in allen 60 Populationen der drei Welten höchstens m = 16, und der Median liegt unter dem der Rohdifferenz. | `test_kipppunkt_of_t_linear_…`, `test_t_linear_tips_earlier_…` |
| Gibt es einen Kipppunkt in der Null-Welt? | **Nein:** in keiner der 20 Populationen schlägt irgendeine Schwellenregel die beste feste Regel (alle fünf Schätzer, alle drei Schwellen). | `test_in_the_null_world_no_threshold_rule_…` |
| Wie weit senkt die Schrumpfung den Gewinner-Fluch? | Ab m = 2 liegt die Überschätzung im Median in allen vier Welten zwischen **−0,6 und +0,6** Prozentpunkten (Rohdifferenz bei m = 2: 8,1 bis 9,5). Bei m = 2 stammt das zum Teil vom Kollaps (31 bis 53 % der Piloten, dort Überschätzung im Median der Populationen zwischen −0,1 und +0,1); in den nicht kollabierten Piloten liegt sie bei m = 2 im Median bei 0,7 bis 1,2 Prozentpunkten. In der Welt Wechselwirkung schießt sie bei m = 8 über das Ziel (Median −0,6). | `test_the_shrinkage_keeps_the_median_overestimation_…` |
| Wann ist die Schrumpfung nicht verwertbar? | Die geschätzte Streuung der wahren Effekte ist **null**, dann liegen alle Schätzungen beim Gesamtmittel und die Auswahl ist Zufall: bei m = 2 in **31 %** (Wechselwirkung), **33 %** (glatt), **43 %** (Falle) und **53 %** (Null-Welt) der Piloten, bei m = 4 in **13 %**, **15 %** und **35 %** der drei heterogenen Welten; bei m = 16 in **0 %**, **0 %**, **6 %** der drei heterogenen Welten. In der Null-Welt bleibt es für jedes m ≥ 2 bei etwa der Hälfte der Piloten. Bei m = 1 ist die Schrumpfung nicht definiert. | `test_shrinkage_collapse_rate_of_the_prior_variance` |
| Welche Schätzerklasse gewinnt (bedingt)? | Welt glatt, Budget 30 %, m = 8, Fang-Anteil in %: **T-linear 95,6 [93,5 bis 97,4]**, T-Wald **87,1 [83,3 bis 91,4]**, Rohdifferenz **59,7 [49,6 bis 65,2]**, Schrumpfung **67,5 [56,6 bis 75,5]**; T-linear liegt in 20 von 20 Populationen vor T-Wald. Welt Wechselwirkung, Budget 30 %, m = 64: T-linear **93,1 [89,7 bis 96,0]** (die Gerade erreicht den Wald nicht), T-Wald **98,5 [97,7 bis 99,8]**; ab m = 4 liegt T-Wald in 18 bis 20 von 20 Populationen vor T-linear. | `test_capture_share_by_estimator_…`, `test_in_the_interaction_world_the_forest_wins_…` |
| Gilt das für jedes Budget? | **Nein:** in der Welt Wechselwirkung liegt T-linear bei **10 % Budget** für m ≤ 16 im Median vor T-Wald (bei m = 64 T-Wald in 16 von 20 Populationen vorn), bei 50 % Budget liegt T-Wald erst ab m = 16 im Median vorn (15 von 20 Populationen, bei m = 64 20 von 20). | `test_the_ranking_of_forest_and_line_in_the_interaction_world_depends_on_the_budget` |
| Was kostet die Schrumpfung in der Falle? | Welt Falle, Budget 30 %, m = 8: Fang-Anteil der Schrumpfung **17,3** Prozentpunkte unter der Rohdifferenz, in **0 von 20** Populationen vorn (für jedes m ≥ 2); bei 10 % Budget 29,5 Prozentpunkte. In der Welt glatt liegt sie bei m = 8 **8,4** Prozentpunkte vor der Rohdifferenz (Median der Unterschiede je Population, 20 von 20; die Differenz der beiden Mediane 67,5 − 59,7 ist 7,8). **Das gilt für kleines Budget:** bei 50 % Budget liegt sie in der Falle ab m = 16 im Median vor der Rohdifferenz (16 von 20 Populationen, bei m = 32 und 64 je 20 von 20). | `test_shrinkage_beats_the_raw_difference_from_m_8_…` |
| Wie eng ist die Kopplung von Effekt und Streuung? | Korrelation (Pearson) zwischen wahrem Effekt und Streuung der Einzeldifferenz: Welt Falle **0,93 [0,91 bis 0,95]** (20 von 20 Populationen über 0,3); Welt glatt **−0,23 [−0,44 bis −0,05]**, davon **6 von 20** mit Betrag ab 0,3; Welt Wechselwirkung **−0,17 [−0,34 bis −0,05]**, **2 von 20**. | `test_coupling_of_effect_and_dispersion`, `test_remeasure_the_correlation_…` |

Zahl → Test: Die Zeichenketten dieser Tabelle stehen wörtlich in `test_claims.py::test_the_readme_names_the_numbers_of_this_file`; Fang-Anteile mit einer Nachkommastelle (Wald-Werte stehen nur in der Datei).

## Befunde und Korrekturen

Die **Vormessung** (`uplift-planung/messreihe_uplift`, zwei Runden, andere Zufallsquelle: numpy-Generator, stetige Standorteigenschaften, Pools von 6 000 Fenstern) war die Erwartung; im Bau wurde mit dem Zufallsgenerator der Demo
(SplitMix64, Standorttypen auf Gittern, Pools von 2 000 Fenstern) **neu gemessen**. Wo beide abweichen, gilt diese README.

- **Bestätigt:** Faktor geschätzt/wahr 4,5 (m = 1) bis 1,3 (m = 64) in der gekoppelten Welt (hier 4,4 bis 1,3, glatt 4,9 bis 1,2); Null-Welt 11,5 gegen **11,7** Prozentpunkte; Kipppunkt der Rohdifferenz 8 bis 32 in den Welten ohne Kopplung (hier 16 und 6 im Median), der Regression 1 bis 8 (hier 1, 1, 8); Regression 93 % gegen Wald 99 % bei Wechselwirkung (hier 93,1 gegen 98,5); Schrumpfung in der Falle schlechter als die Rohdifferenz (hier bei 10 und 30 % Budget in 0 von 20 Populationen vorn; bei 50 % Budget nicht, siehe Befunde).
- **Verschoben:** (1) **Der Vorteil der Wälder bei Wechselwirkung hängt von Budget und Pilotumfang ab** (die Vormessung maß nur 30 %): bei 30 % Budget ab m = 4, bei 50 % ab m = 16, bei 10 % erst bei m = 64 (Median; bei 10 % bei m = 32 gleichauf); die Aussage „bei Wechselwirkung gewinnt der Wald“ steht deshalb nur mit Budget und Pilotumfang. (2) **Kollaps der Schrumpfung:** Vormessung 24 bis 42 % bei m = 2 bis 4; hier 31 bis 43 % bei m = 2 und 13 bis 35 % bei m = 4 (heterogene Welten); bei m = 16 höchstens 6 % (Vormessung 9 %). Neu: in der **Null-Welt** ist die Prior-Varianz in etwa der Hälfte aller Piloten null, für jedes m. (3) Die Korrelation von Effekt und Streuung in der Falle liegt bei 0,93 (Vormessung 0,95 [0,93 bis 0,98]; die Abweichung ist klein, ihre Ursache nicht untersucht). (4) Populationen mit |ρ| ≥ 0,3 (von 20): glatt 6 (Vormessung 4), Wechselwirkung 2 (Vormessung 1).
- **Weggelassen gegenüber der Vormessung:** der Haken „Gütemaß gegen Entscheidungswert“ (die Vormessung zeigte ihn als Artefakt der Kopplung; nicht gebaut), der Uplift-Baum und der S-Wald (kein eigener Befund), die Welt „entkoppelt: Vehikel-Merkmale“ (die Welt glatt entspricht der Welt „entkoppelt: Dosis“).
- **Auflagen der Vormessung:** (a) **Rucksack** (Nutzen τ·μ gegen τ allein, ungleiche Kosten) **nicht neu gemessen und weggelassen**; die App bildet nur die Auswahl mit gleichen Kosten ab und sagt das. (b) |ρ| < 0,3 gilt nur **im Median** der Populationen (siehe oben), nicht in jeder, und die Korrelation ist in allen 40 Populationen negativ: die Welten glatt und Wechselwirkung sind nur schwach gekoppelt, nicht entkoppelt; das steht in der App. (c) Die **Wechselwirkungswelt ist eine gesetzte Annahme** und steht als solche in der Annahmen-Tabelle der App und in der Tabelle oben. (d) Die **Anomalie bei 80 % Budget** (Rohdifferenz bei kleinem m schlechter als Zufall) ist nicht geklärt: Sie zeigt sich hier nur in der Welt Falle (Fang-Anteil **-16,4 [-25,1 bis -12,6]** bei m = 1), nicht in den Welten glatt und Wechselwirkung, wo er positiv ist. Ursache **nicht untersucht**; 80 % wird deshalb nicht angeboten und nicht als Befund geführt.
- **Zahlen der Zähleinheit:** Faktoren sind Verhältnisse der Mittel über die Piloten; „Population“ ist eine Ziehung von 100 Standorten, „Zelle“ ein Pilot einer Population bei einem Pilotumfang.

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Wahrheit = Poolmittel** | Standorte gleichen Typs haben exakt denselben wahren Effekt; der wahre Effekt hat selbst einen Fehler (Standardfehler des wahren Effekts, einer Differenz zweier Poolmittel, bei z = 1,4 und Fensterlänge 7 Tage 0,24 Prozentpunkte, bei 224 Tagen 0,05). Alle Vergleiche benutzen dieselbe Wahrheit auf beiden Seiten. | – |
| **Wechselwirkungswelt** | Der Vorteil der Wälder gilt für diese **gesetzte** Zuordnung der Regelstärke und nur bei bestimmten Budgets und Pilotumfängen (30 % Budget ab m = 4, 50 % ab m = 16, 10 % erst bei m = 64), nicht allgemein; gegenüber T-linear mit Produkttermen ist er klein (30 % Budget, m = 8: +2,1 Prozentpunkte Fang-Anteil, in 17 von 20 Populationen vorn). | – |
| **Effekt und Streuung nur schwach gekoppelt** (glatt, Wechselwirkung) | Die Korrelation ist im Median −0,23 und −0,17, in allen 40 Populationen negativ (kurze Fenster: kleinerer Effekt, größere Streuung), in 6 und 2 von 20 mindestens 0,3 im Betrag. In der Falle hängen beide eng zusammen, und die Schrumpfung schneidet dort bei 10 und 30 % Budget schlechter ab als die Rohdifferenz (bei 50 % Budget ab m = 16 besser); **warum**, ist eine Deutung (Standorte mit großem Effekt streuen stark und werden stärker zurückgezogen), keine Messung. | Schrumpfung (Falle) |
| **Pilotumfang kostet nichts** | Ein größerer Pilot kostet in der Wirklichkeit Fenster; die Demo wägt Pilot und Behandlung nicht ab. | Teststärke, Bandit |
| **Gleiche Kosten je Standort** | Bei ungleichen Kosten (Rucksack) ist nicht mehr der Effekt allein die Rangfolge; nicht gemessen. | – |
| **Einstellungen der Schätzer** | Blattgröße (Rand des Gitters), 4 Pseudo-Freiheitsgrade, 40 Bäume: Aussagen gelten für diese Regeln, nicht für die Verfahrensklasse; T-Wald-Zahlen sind nicht neu gemessen. | – |
| **Schrumpfung bei kleinem m** | Kollaps der Prior-Varianz (siehe Befunde); bei m = 1 nicht definiert. | – |
| **Eine Welt je Aussage** | Aussagen über „glatt“ oder „Wechselwirkung“ gelten für diese Welten, nicht für jede Form des Effekts; „Falle“ ist ein Vehikel-Effekt, kein Modell der Praxis. | – |

## Bezug zu OR

„Wähle k von N Standorten“ ist ein **0-1-Rucksackproblem mit gleichen Kosten** (Kardinalitätsschranke): max Σ τ̂ₛ xₛ unter Σ xₛ = k; die Lösung sind die k größten Schätzungen. Die Zielfunktion kennt nur die **geschätzten**
Koeffizienten, und das Optimum eines Problems mit verrauschten Koeffizienten ist **zu optimistisch**: der Gewinner-Fluch ist die Lücke zwischen dem Zielwert des Problems und dem wahren Wert derselben Auswahl (die App zeigt beide
Zahlen für den Piloten der Anzeige, für Rohdifferenz und Schrumpfung). Die **Schrumpfung** ersetzt die Koeffizienten durch ihre bessere Schätzung und senkt die Lücke (Überschätzung der Gewählten bei 10 % Budget) ab m = 2 im Median auf höchstens 0,6 Prozentpunkte; ein Modell über die
Merkmale senkt sie ebenfalls (T-linear bei m = 8 in der Welt glatt: **0,6 [0,5 bis 0,8]**). Das ist dieselbe Frage wie bei jedem Optimierer, der auf Schätzungen läuft (Prognose, dann Optimierung), hier an einem Fall, in dem die Wahrheit bekannt ist.

## Verwandte Demos im Portfolio

- [ab-test-demo](https://github.com/sebastian-hanisch/ab-test-demo): Vehikel und Zufall dieses Stücks; ein Vergleich, hier viele Standorte mit eigenem Effekt.
- [test-power-demo](https://github.com/sebastian-hanisch/test-power-demo): wie viele Fenster für **einen** Effekt; hier der Pilotumfang je Standort und die Folge für die Auswahl.
- [multiple-testing-demo](https://github.com/sebastian-hanisch/multiple-testing-demo): Gewinnerfluch für die **beste von k Varianten** bei gleicher Wahrheit; dort heißt es, die Schrumpfung sei nicht gebaut. Hier: k von 100 Standorten mit **eigenem** Effekt, und die Schrumpfung ist das Gegenmittel.
- [cuped-demo](https://github.com/sebastian-hanisch/cuped-demo): weniger Streuung durch Vorab-Wissen; hier nutzen Modelle die Merkmale der Standorte für den Effekt.
- [bandit-vs-ab-demo](https://github.com/sebastian-hanisch/bandit-vs-ab-demo) und [off-policy-demo](https://github.com/sebastian-hanisch/off-policy-demo): adaptive Zuteilung und Auswertung aus Logdaten; hier wird nicht verteilt, sondern nach einem Piloten ausgewählt.
- [regret-statt-fehler-demo](https://github.com/sebastian-hanisch/regret-statt-fehler-demo) und [spo-kuerzester-weg-demo](https://github.com/sebastian-hanisch/spo-kuerzester-weg-demo) (Linie „Entscheidungsorientiertes Lernen“): dort wird ein Modell nach dem **Regret der Entscheidung** statt nach dem Fehler der Vorhersage bewertet; hier ist die Entscheidung eine Auswahl, und gemessen wird der wahre Wert der Gewählten.

## Verifikation

- **Vehikel und Zufall:** die vektorisierte Fassung gleich der skalaren Kopie aus ab-test-demo Bit für Bit (3 Sicherheitsfaktoren, 3 Seeds); gegen eine unabhängig geschriebene skalare Referenz mit allen Parametern (5 Standorttypen, Fensterlängen 7 bis 56);
  Poisson-Inversion gleich `scipy.stats.poisson.ppf` auf 20 000 Ziehungen; Pool-Mittel gegen die exakte Formel des Gleichgewichts (Band, andere Einheit); jeder Pool der Datei aus seinem Schlüssel reproduzierbar (`tests/test_oracle_vehicle.py`, `test_stream.py`, `test_rng.py`, `test_copies.py`).
- **Orakel** (`tests/test_oracle_wen_behandeln.py`): Gewinner-Fluch im Normalmodell gegen die geschlossene Formel (Erwartungswerte der Ordnungsstatistiken per Quadratur; gilt für Mittel 3, Streuung der wahren Effekte 1, Pilotfehler 2, 100 Standorte, 10 Gewählte, in beliebigen Einheiten: 3,094 erwartet, Simulation mit 2 500 Wiederholungen im Band ±0,06); Schrumpfung mit bekannter Varianz je Standort (Mittel und Streuung der wahren Effekte werden geschätzt, mit ihnen wäre sie der exakte Posteriormittelwert) senkt die Überschätzung der Gewählten im Mittel über 1 500 Wiederholungen unter 0,1 (ohne Schrumpfung 3,1); Schrumpfung und moderierte Varianz von Hand; Auswahl gegen Aufzählung aller Teilmengen (40 Instanzen) und gegen `scipy.optimize.milp` (30 Instanzen); Regressionsbaum gleich
  `sklearn.tree.DecisionTreeRegressor` (60 Instanzen ohne Gleichstände, dazu neue Zeilen); Regression und Produktterme gegen exakt lineare bzw. Wechselwirkungs-Wahrheit; Kipppunkt an handgesetzten Reihen.
- **Wertmaße, Verdikte, Populationen:** Wertmaße und Referenzgrößen an handgesetzten Zahlen; jedes Verdikt mit Schwelle und jeder genannten Zahl (`tests/test_evaluation.py`, `test_verdicts.py`); Aufbau jeder Welt, Zweig-Test (jede Variante weicht von der Nachbarvariante ab, keine Nullspalte), Randfälle (`test_pools.py`, `test_edge_cases.py`).
- **Messreihe:** Aufbau und Widerspruchsfreiheit der Datei (`test_sweeps.py`); Welten glatt und Falle für Rohdifferenz und T-linear **neu gemessen** und gegen die Datei geprüft; jede Zahl dieser README (`test_claims.py`).
- **Oberfläche:** AppTest (jeder Button, jeder Regler an seinen Grenzen verändert das Ergebnis, Permalink rastet ein, Schrumpfung bei m = 1), Layout (Schnellstart, fester Kopf, aktives Preset), Footer, Laufzeit-Importe. AppTest löst keine Plotly-Klicks aus; die Diagramme sind im Browser durchgesehen (Preview), nicht mit Mausklicks getestet.
- **Insgesamt 172 Tests** (`pytest --collect-only -q tests`).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets, Arbeitsordner `_scratch/` |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: ein Pilot, Kernfragen, vier Experimente, Brücke zur Optimierung, Annahmen, Formeln |
| `wbh_constants.py` | Vehikel, Welten, Regler-Grenzen, Presets, Zahlenformate |
| `wbh_rng.py`, `wbh_vehicle.py`, `wbh_formulas.py` | wörtliche Kopien aus ab-test-demo (SplitMix64, skalares Vehikel, exakte Formel) |
| `wbh_stream.py` | fortsetzbarer SplitMix64-Strom, Seeds mischen |
| `wbh_sim.py` | vektorisiertes Vehikel mit je Standort eigenen Parametern |
| `wbh_pools.py` | Bank der Pools, Welten, Populationen, Pilot |
| `wbh_estimators.py` | Rohdifferenz, Schrumpfung, T-linear, Produktterme, Regressionsbaum und Wald |
| `wbh_evaluation.py` | Auswahl, Wertmaße, Messzelle, Kipppunkt, Live-Auswertung, Verdikte |
| `wbh_sweeps.py` | Zugriff auf die vorgerechnete Messreihe |
| `wbh_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `wbh_presets.py`, `wbh_layout.py` | Permalink, Presets, Zufalls-Seed-Button; fester Kopf |
| `generate_precomputed.py`, `precomputed_pools.npz` | die Pools (342 à 2 000 Fenster, etwa 1,1 MB) |
| `tools/messreihe.py`, `precomputed_sweeps.json`, `tools/tuning.json` | Messreihe (tune, run, aggregate), ihre Ergebnisse, Abstimmung des Waldes |
| `requirements.txt`, `requirements-dev.txt` | Laufzeit; plus pytest, scikit-learn (Gegenproben) |
| `tests/` | Vehikel, Orakel, Schätzer, Wertmaße, Verdikte, Populationen, Messreihe, Befunde, Presets, Layout, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Rucksack mit ungleichen Kosten und Nutzen τ·μ:** in der Vormessung nur in der gekoppelten Welt gemessen, hier nicht neu gemessen; die App zeigt nur die Auswahl mit gleichen Kosten.
- **Uplift-Baum, S-Learner, Uplift-Wald als eigene Verfahren:** Vormessung ohne eigenen Befund.
- **Welche Standorte der Pilot überhaupt bekommt** (Zuteilung des Piloten, Bandit): [bandit-vs-ab-demo](https://github.com/sebastian-hanisch/bandit-vs-ab-demo).
- **Pilotumfang planen** (wie viele Fenster für einen gegebenen Effekt): [test-power-demo](https://github.com/sebastian-hanisch/test-power-demo).
- **Gewinner-Fluch für die beste von k Varianten bei gleicher Wahrheit und Korrekturen der Fehlalarmrate:** [multiple-testing-demo](https://github.com/sebastian-hanisch/multiple-testing-demo).
- **Modell nach Regret statt Fehler trainieren** (SPO+, gewichtete Stichprobe): Linie „Entscheidungsorientiertes Lernen“, hier werden die Schätzer nach ihrem Fehler gefittet und nur die Auswahl nach dem Wert bewertet.
- **Budget von 80 %:** gemessen, aber nicht angeboten (siehe Befunde und Korrekturen, Punkt d).
- **Praxisdaten:** alle Fenster sind erzeugt.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Die Messreihe neu erzeugen (Minuten, alle Kerne): `python generate_precomputed.py`, `python tools/messreihe.py tune`, `run`, `aggregate`.

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und SciPy (nur die kopierte exakte Formel des Vehikels, die die App nicht aufruft).

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Experimente und Wirkungsmessung: A/B bis Logdaten](https://sebastianhanisch.net/konzepte-experimente-wirkungsmessung.html).
