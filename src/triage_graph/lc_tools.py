"""LangChain tool wrappers around `tools.py`, scoped to the alert in graph state."""

from __future__ import annotations

import json
from typing import Annotated, Any

from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import InjectedState

from triage_graph import tools
from triage_graph.data import load_dataset, load_runbooks
from triage_graph.schemas import Alert

State = Annotated[dict[str, Any], InjectedState]


def _alert(state: dict[str, Any]) -> Alert:
    return Alert.model_validate(state["alert"])


@tool
def search_logs(
    state: State, start: str, end: str, min_level: str = "WARN", pattern: str | None = None
) -> str:
    """Search the alerting service's logs between two ISO-8601 UTC timestamps.

    Matching lines are grouped by message template, with counts, first/last seen,
    and one example. min_level is one of DEBUG, INFO, WARN, ERROR. pattern is an
    optional case-insensitive regular expression matched against the message.
    """
    alert = _alert(state)
    result = tools.search_logs(
        load_dataset(alert.dataset), alert.service, start, end, min_level, pattern
    )
    return json.dumps(result)


@tool
def detect_metric_anomalies(state: State, start: str, end: str, baseline_minutes: int = 60) -> str:
    """Find anomalous metrics for the alerting service in [start, end].

    Each metric (cpu_pct, memory_pct, latency_p99_ms, error_rate_pct, rps) is
    compared with its median over the `baseline_minutes` before `start`.
    Timestamps are ISO-8601 UTC.
    """
    alert = _alert(state)
    result = tools.detect_metric_anomalies(
        load_dataset(alert.dataset), alert.service, start, end, baseline_minutes
    )
    return json.dumps(result)


@tool
def get_metric_series(
    state: State, metric: str, start: str, end: str, step_minutes: int = 5
) -> str:
    """One metric for the alerting service, averaged into step_minutes buckets.

    metric is one of cpu_pct, memory_pct, latency_p99_ms, error_rate_pct, rps.
    Timestamps are ISO-8601 UTC.
    """
    alert = _alert(state)
    result = tools.get_metric_series(
        load_dataset(alert.dataset), alert.service, metric, start, end, step_minutes
    )
    return json.dumps(result)


def make_search_runbooks(include_suggestions: bool) -> BaseTool:
    @tool("search_runbooks")
    def search_runbooks(query: str, service: str | None = None) -> str:
        """Search the runbooks for sections relevant to `query`.

        Pass `service` to search only that service's runbook plus the general policy.
        Each result includes the matching section's text.
        """
        result = tools.search_runbooks(
            load_runbooks(), query, service, include_suggestions=include_suggestions
        )
        return json.dumps(result)

    return search_runbooks


def specialist_tools(runbook_suggestions: bool = True) -> dict[str, list[BaseTool]]:
    return {
        "log_investigator": [search_logs],
        "metrics_investigator": [detect_metric_anomalies, get_metric_series],
        "runbook_agent": [make_search_runbooks(runbook_suggestions)],
    }


SPECIALIST_TOOLS = specialist_tools()
