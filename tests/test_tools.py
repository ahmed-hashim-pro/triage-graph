from datetime import timedelta

import pytest
from conftest import EXPECTED, alert_path

from triage_graph.data import load_alert, load_dataset, load_runbooks
from triage_graph.tools import (
    ToolInputError,
    detect_metric_anomalies,
    get_metric_series,
    search_logs,
    search_runbooks,
)

EXPECTED_ANOMALIES = {
    "high-latency": {"rps", "cpu_pct", "latency_p99_ms"},
    "error-spike-after-deploy": {"error_rate_pct"},
    "memory-leak": {"memory_pct", "cpu_pct", "latency_p99_ms", "error_rate_pct"},
    "intermittent-5xx": {"error_rate_pct", "latency_p99_ms"},
}


def window(scenario, before, after=timedelta(minutes=5)):
    alert = load_alert(alert_path(scenario))
    iso = lambda t: t.isoformat().replace("+00:00", "Z")  # noqa: E731
    return (
        alert,
        load_dataset(alert.dataset),
        iso(alert.fired_at - before),
        iso(alert.fired_at + after),
    )


def test_metric_anomalies_over_three_hours(scenario):
    alert, ds, start, end = window(scenario, timedelta(hours=3))
    result = detect_metric_anomalies(ds, alert.service, start, end)
    assert {a["metric"] for a in result["anomalies"]} == EXPECTED_ANOMALIES[scenario]


def test_short_baseline_misses_a_slow_memory_leak():
    alert, ds, start, end = window("memory-leak", timedelta(hours=2))
    result = detect_metric_anomalies(ds, alert.service, start, end)
    assert "memory_pct" not in {a["metric"] for a in result["anomalies"]}


def test_log_search_groups_by_template_and_filters_by_level():
    alert, ds, start, end = window("error-spike-after-deploy", timedelta(hours=1))
    result = search_logs(ds, alert.service, start, end, min_level="ERROR")
    assert {g["level"] for g in result["groups"]} == {"ERROR"}
    top = result["groups"][0]
    assert "KeyError: 'currency'" in top["example"]
    assert top["count"] > 100
    assert result["matched_lines"] == sum(g["count"] for g in result["groups"])


def test_log_search_pattern_finds_the_deploy():
    alert, ds, start, end = window("error-spike-after-deploy", timedelta(hours=3))
    result = search_logs(ds, alert.service, start, end, min_level="INFO", pattern="rolled out")
    assert [g["example"] for g in result["groups"]] == [
        "deploy finished: payments-api v3.8.0 rolled out to 8/8 pods"
    ]


def test_invalid_regex_falls_back_to_a_literal_match():
    alert, ds, start, end = window("high-latency", timedelta(hours=1))
    result = search_logs(ds, alert.service, start, end, pattern="queue depth (")
    assert result["matched_lines"] == 0


@pytest.mark.parametrize(
    ("start", "end", "level"),
    [
        ("yesterday", "2026-09-14T11:00:00Z", "WARN"),
        ("2026-09-14T11:00:00Z", "2026-09-14T10:00:00Z", "WARN"),
        ("2026-09-14T10:00:00Z", "2026-09-14T11:00:00Z", "LOUD"),
    ],
)
def test_bad_log_arguments_raise_tool_input_error(start, end, level):
    _, ds, _, _ = window("high-latency", timedelta(hours=1))
    with pytest.raises(ToolInputError):
        search_logs(ds, "checkout-api", start, end, min_level=level)


def test_metric_series_is_capped_at_sixty_points():
    alert, ds, start, end = window("high-latency", timedelta(hours=4))
    result = get_metric_series(ds, alert.service, "cpu_pct", start, end, step_minutes=1)
    assert len(result["points"]) <= 60
    with pytest.raises(ToolInputError):
        get_metric_series(ds, alert.service, "disk_pct", start, end)


def test_runbook_search_scoped_to_service_suggests_the_expected_action(scenario):
    alert = load_alert(alert_path(scenario))
    result = search_runbooks(load_runbooks(), f"{alert.title} {alert.description}", alert.service)
    assert result["results"][0]["suggested_action"] == EXPECTED[scenario]["expected_action"]


def test_runbook_search_rejects_an_empty_query():
    with pytest.raises(ToolInputError):
        search_runbooks(load_runbooks(), "the of and")
