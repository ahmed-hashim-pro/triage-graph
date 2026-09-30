import filecmp
import importlib.util
import sys

from conftest import ALERTS, EXPECTED, ROOT, SCENARIOS

from triage_graph.actions import ALLOWED_ACTIONS
from triage_graph.data import load_alert, load_dataset, load_runbooks


def test_every_alert_has_an_expected_answer_and_vice_versa():
    assert {p.stem for p in ALERTS.glob("*.json")} == set(SCENARIOS)
    assert len(SCENARIOS) >= 3


def test_expected_actions_are_allowed_and_include_page_human():
    actions = {e["expected_action"] for e in EXPECTED.values()}
    assert actions <= ALLOWED_ACTIONS
    assert "page_human" in actions


def test_alerts_do_not_leak_the_expected_answer():
    for scenario in SCENARIOS:
        text = (ALERTS / f"{scenario}.json").read_text()
        assert EXPECTED[scenario]["expected_action"] not in text


def test_every_alert_loads_with_its_dataset(scenario):
    alert = load_alert(ALERTS / f"{scenario}.json")
    dataset = load_dataset(alert.dataset)
    assert any(r.service == alert.service for r in dataset.logs)
    assert any(p.service == alert.service for p in dataset.metrics)


def test_every_alerting_service_has_a_runbook(scenario):
    alert = load_alert(ALERTS / f"{scenario}.json")
    assert any(s.title == alert.service for s in load_runbooks())


def test_generated_fixtures_are_reproducible(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("gen", ROOT / "scripts" / "generate_fixtures.py")
    gen = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "gen", gen)
    spec.loader.exec_module(gen)
    gen.main(tmp_path)
    for kind in ("logs", "metrics"):
        committed = ROOT / "fixtures" / kind
        regenerated = tmp_path / "fixtures" / kind
        names = sorted(p.name for p in regenerated.iterdir())
        _, mismatch, errors = filecmp.cmpfiles(committed, regenerated, names, shallow=False)
        assert not mismatch and not errors, (mismatch, errors)
