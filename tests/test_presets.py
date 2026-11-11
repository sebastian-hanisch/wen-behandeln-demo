"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, Permalink rastet ein."""

import pytest

import wbh_constants as C
import wbh_evaluation as ev
import wbh_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_valid_choices_of_their_controls():
    for name, p in C.PRESETS.items():
        assert 0 <= p["welt"] < len(C.WELTEN) and 0 <= p["est"] < len(C.ESTIMATORS), name
        assert p["m"] in C.M_GRID and p["budget"] in C.BUDGETS and p["delta"] in C.DELTAS and 0 <= p["seed"] <= C.SEED_MAX, name


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Gewinner-Fluch (Voreinstellung)"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key_and_every_step_control():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    for key, steps in P.STEPS.items():
        spec = P.SETTING_SPECS[key]
        assert (spec.lo, spec.hi) == (min(steps), max(steps))


def test_preset_names_describe_their_values():
    p = C.PRESETS
    assert C.WELTEN[p["Nur Rauschen (Null-Welt)"]["welt"]] == "homogen"
    assert C.WELTEN[p["Wechselwirkung: Wälder"]["welt"]] == "wechselwirkung" and C.ESTIMATORS[p["Wechselwirkung: Wälder"]["est"]] == "twald"
    assert C.WELTEN[p["Die Falle: Streuung verrät den Effekt"]["welt"]] == "falle"
    assert C.ESTIMATORS[p["Schrumpfung als Gegenmittel"]["est"]] == "eb" and p["Schrumpfung als Gegenmittel"]["m"] >= 2
    assert p["Größerer Pilot"]["m"] > p["Gewinner-Fluch (Voreinstellung)"]["m"]


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_each_preset_runs_through_the_analysis(name):
    a = ev.analyse(ev.Settings(**C.PRESETS[name]))
    assert a.defined and a.k == ev.k_of(C.PRESETS[name]["budget"])


def test_each_preset_shows_the_expected_verdict_lines():
    """Je Preset die erwartete Verdikt-Zeile (DEMO-PLAYBOOK §11a Punkt 7): Überschätzung und Lohnt-die-Auswahl, live und aus der Messreihe."""
    expected = {                       # (Art Überschätzung live, Art Kipppunkt live, Art Überschätzung Messreihe)
        "Gewinner-Fluch (Voreinstellung)": ("warning", "warning", "warning"),
        "Größerer Pilot": ("info", "success", "info"),
        "Nur Rauschen (Null-Welt)": ("warning", "warning", "warning"),
        "Schrumpfung als Gegenmittel": ("success", "success", "success"),
        "Wechselwirkung: Wälder": ("success", "success", "success"),
        "Die Falle: Streuung verrät den Effekt": ("success", "warning", "success"),
    }
    assert set(expected) == set(C.PRESETS)
    for name, (k1, k2, k3) in expected.items():
        p = C.PRESETS[name]
        a = ev.analyse(ev.Settings(**p))
        assert (ev.verdict_fluch(a)[0], ev.verdict_kipp(a)[0], ev.verdict_fluch_table(C.WELTEN[p["welt"]], C.ESTIMATORS[p["est"]], p["budget"], p["m"])[0]) == (k1, k2, k3), name
    # Null-Welt: kein Effektunterschied, also darf keine Auswahl gewinnen; Falle: der Betrag ist winzig und wird so genannt
    assert "Der Betrag ist klein" in ev.verdict_kipp(ev.analyse(ev.Settings(**C.PRESETS["Die Falle: Streuung verrät den Effekt"])))[1]
