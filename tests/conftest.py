"""Gemeinsame Fixtures: Wurzelordner im Importpfad, eine Mini-Population von Hand (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def mini():
    """5 Standorte mit wahren Effekten 4, 1, 3, 5, 2 Prozentpunkte. Handrechnung: die zwei mit der größten Schätzung (Schätzung = 4.5, 0.5, 3.5, 5.5, 2.5)
    sind Index 3 und 0 (Schätzungen 5.5 und 4.5, wahre Effekte 5 und 4)."""
    tau = np.array([4.0, 1.0, 3.0, 5.0, 2.0])
    th = tau + 0.5
    return tau, th
