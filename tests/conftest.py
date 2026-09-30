from __future__ import annotations

import json
from pathlib import Path

import pytest

from triage_graph.data import load_alert

ROOT = Path(__file__).resolve().parents[1]
ALERTS = ROOT / "fixtures" / "alerts"
EXPECTED = json.loads((ROOT / "fixtures" / "expected.json").read_text())
SCENARIOS = sorted(EXPECTED)


def alert_path(scenario: str) -> Path:
    return ALERTS / f"{scenario}.json"


def alert_dict(scenario: str) -> dict:
    return load_alert(alert_path(scenario)).model_dump(mode="json")


@pytest.fixture(params=SCENARIOS)
def scenario(request: pytest.FixtureRequest) -> str:
    return request.param
