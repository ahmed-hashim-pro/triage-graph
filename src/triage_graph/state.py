"""Graph state. Values are JSON-shaped; see `schemas.py` for their validated forms."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, NotRequired, TypedDict

Outcome = Literal["executed", "rejected", "refused", "escalated"]

DEFAULT_MAX_STEPS = 6


def add_usage(left: dict[str, int] | None, right: dict[str, int] | None) -> dict[str, int]:
    merged = dict(left or {})
    for key, value in (right or {}).items():
        merged[key] = merged.get(key, 0) + value
    return merged


class TriageInput(TypedDict):
    """The only keys a caller may set. Anything else passed in is dropped by LangGraph."""

    alert: dict[str, Any]
    max_steps: NotRequired[int]


class TriageState(TypedDict, total=False):
    alert: dict[str, Any]
    max_steps: int
    # Which model produced the findings, e.g. "fake" or "anthropic:claude-opus-5-5".
    model: str
    # Supervisor turns that did not end the investigation, including invalid replies.
    steps: int
    next: str
    focus: str
    findings: Annotated[list[dict[str, Any]], operator.add]
    supervisor_log: Annotated[list[dict[str, Any]], operator.add]
    proposal: dict[str, Any]
    decision: dict[str, Any]
    execution: dict[str, Any]
    outcome: Outcome
    outcome_reason: str
    report: str
    usage: Annotated[dict[str, int], add_usage]
