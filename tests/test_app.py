"""End-to-end Rauchtest über Streamlits AppTest: lädt app.py mit den Standardeinstellungen, klickt jeden Preset-Button, fährt jeden Regler an seine Grenzen
und prüft, dass kein Python-Fehler auftritt (insbesondere `StreamlitDuplicateElementId` bei mehreren `st.plotly_chart` mit gleichem Inhalt) und dass jede Einstellung
das Hauptergebnis verändert (keine toten Regler). AppTest löst keine Plotly-Klicks aus; die Diagramme sind nur auf Fehler geprüft."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import wbh_constants as C
import wbh_presets as P

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _control(at, key):
    for art in ("select_slider", "selectbox", "number_input"):
        try:
            return getattr(at.sidebar, art)(key=key)
        except KeyError:
            continue
    raise AssertionError(key)


def _signature(at):
    """Alles, was das Hauptergebnis zeigt: Kennzahlen und Verdikttexte."""
    return (tuple(m.value for m in at.metric), tuple(x.value for grp in (at.success, at.info, at.warning) for x in grp))


def test_app_loads_without_exception():
    at = _fresh_app()
    assert at.title and len(at.metric) == 7 and len(at.dataframe) >= 4


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index in range(len(labels)):
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (labels[index], [str(e) for e in at.exception])


@pytest.mark.parametrize("key", [k for k in P.SETTING_SPECS])
def test_every_control_at_its_min_and_max_does_not_raise_and_changes_the_result(key):
    spec = P.SETTING_SPECS[key]
    base = _signature(_fresh_app())
    for value in (spec.lo, spec.hi):
        at = _fresh_app()
        _control(at, key).set_value(value).run(timeout=TIMEOUT)
        assert not at.exception, (key, value, [str(e) for e in at.exception])
        if value != spec.default:
            assert _signature(at) != base, (key, value)


def test_the_undefined_shrinkage_at_m_one_shows_an_info_instead_of_a_result():
    at = _fresh_app()
    _control(at, "est_select").set_value(1).run(timeout=TIMEOUT)          # Standard: m = 1
    assert not at.exception
    assert any("nicht definiert" in x.value for x in at.info)
    assert len(at.metric) == 0


def test_the_wald_estimator_runs_with_few_repetitions():
    at = _fresh_app()
    _control(at, "est_select").set_value(C.ESTIMATORS.index("twald")).run(timeout=TIMEOUT)
    assert not at.exception and any("10 neue Piloten" in m.value for m in at.markdown)


def test_permalink_snaps_to_the_steps_and_clamps_the_bounds():
    at = AppTest.from_file(APP_PATH)
    at.query_params["m"], at.query_params["budget"], at.query_params["delta"] = "5", "999", "-7"
    at.query_params["welt"], at.query_params["seed"] = "2", "-3"
    at.run(timeout=TIMEOUT)
    assert not at.exception
    assert at.sidebar.select_slider(key="m_select").value == 4 and at.sidebar.select_slider(key="budget_select").value == 50
    assert at.sidebar.select_slider(key="delta_select").value == -1 and at.sidebar.selectbox(key="welt_select").value == 2
    assert at.sidebar.number_input(key="seed_input").value == 0


def test_permalink_ignores_garbage():
    at = AppTest.from_file(APP_PATH)
    at.query_params["m"] = "abc"
    at.run(timeout=TIMEOUT)
    assert not at.exception and at.sidebar.select_slider(key="m_select").value == C.DEFAULT_M
