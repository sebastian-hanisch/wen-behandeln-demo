"""Kopien aus ab-test-demo: Rechenkern, Zufall und Vehikel sind wörtlich übernommen (nur der Präfix der Modulnamen ist `wbh_`).

Eine Korrektur an einer Stelle muss in der anderen nachgezogen werden; ohne den Ordner `ab-test-demo` neben diesem Repo (CI) überspringen sich die Tests."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT.parent / "ab-test-demo"


@pytest.mark.parametrize("name", ["rng", "vehicle", "formulas"])
def test_kopie_stimmt_mit_der_quelle_ueberein_falls_vorhanden(name):
    src = SOURCE / f"abt_{name}.py"
    if not src.exists():
        pytest.skip("ab-test-demo liegt nicht neben diesem Repo")
    assert (ROOT / f"wbh_{name}.py").read_text(encoding="utf-8") == src.read_text(encoding="utf-8").replace("abt_", "wbh_")
