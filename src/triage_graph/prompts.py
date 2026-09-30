"""Prompts and the two decision tools (`route`, `propose_action`).

Context is passed to models as JSON inside XML-style tags. Real models read it
fine, and the fake model parses the same tags, so both see identical inputs.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any, Literal

from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, Field

from triage_graph.actions import ALLOWED_ACTIONS

ALLOWED_LIST = ", ".join(sorted(ALLOWED_ACTIONS))
RouteTarget = Literal["log_investigator", "metrics_investigator", "runbook_agent", "propose"]


class RouteArgs(BaseModel):
    next: RouteTarget = Field(description="The specialist to dispatch, or 'propose'.")
    focus: str = Field(default="", description="What the specialist should look for.")
    reason: str = Field(default="", description="One sentence on why this is the next step.")


class ProposeArgs(BaseModel):
    diagnosis: str = Field(description="What is wrong and the evidence for it.")
    action: str = Field(description=f"Exactly one of: {ALLOWED_LIST}.")
    target: str = Field(description="The service the action applies to.")
    rationale: str = Field(description="Why this action, and why not the others.")


def as_tool(name: str, description: str, schema: type[BaseModel]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": schema.model_json_schema(),
        },
    }


ROUTE_TOOL = as_tool("route", "Choose the next step of the investigation.", RouteArgs)
PROPOSE_TOOL = as_tool(
    "propose_action", "Propose one remediation for a human to approve.", ProposeArgs
)

SUPERVISOR_SYSTEM = """\
You coordinate the investigation of a production alert. You have three specialists:

- log_investigator: searches the alerting service's logs.
- metrics_investigator: finds anomalies in CPU, memory, latency, error rate, and request rate.
- runbook_agent: retrieves the relevant runbook guidance.

Call the `route` tool once. Either dispatch a specialist with a specific focus, or choose
`propose` when the findings are enough to recommend a single action. Do not dispatch the same
specialist again unless you need something specific it has not looked at yet. Dispatches are
limited; if they run out, the incident is escalated to a human as "insufficient evidence"."""

SPECIALIST_SYSTEM = {
    "log_investigator": """\
You investigate application logs for a production alert. Use `search_logs` to look for errors
and warnings around the alert time, and for recent changes (deploys, config or feature-flag
changes, restarts, autoscaling). Timestamps are ISO-8601 UTC.""",
    "metrics_investigator": """\
You investigate service metrics for a production alert. Use `detect_metric_anomalies` to
compare the hours before the alert against a baseline, and `get_metric_series` to look at one
metric in detail. Slow trends such as memory growth need a window of several hours.
Timestamps are ISO-8601 UTC.""",
    "runbook_agent": """\
You find the runbook guidance that applies to a production alert. Use `search_runbooks`, pass
the alerting service as `service`, and describe the symptoms in the query.""",
}

SPECIALIST_FINISH = (
    "When you have what you need, reply with a short plain-text summary of the evidence "
    "(counts, timestamps, metric values). Do not call a tool in that final reply."
)

PROPOSER_SYSTEM = f"""\
You write the diagnosis for a production alert and propose exactly one remediation by calling
`propose_action`. The allowed actions are: {ALLOWED_LIST}.

Propose page_human when the evidence is ambiguous, points at more than one cause, or points at
a problem none of the other actions fixes. A human reviews your proposal before anything runs,
and any action outside the allowed list is rejected."""


def tagged(tag: str, value: Any) -> str:
    return f"<{tag}>{json.dumps(value, sort_keys=True)}</{tag}>"


def read_tag(messages: Sequence[BaseMessage], tag: str) -> Any:
    """The JSON value inside the last `<tag>...</tag>` in the human messages, or None."""
    pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL)
    for message in reversed(messages):
        if isinstance(message, HumanMessage) and isinstance(message.content, str):
            match = pattern.search(message.content)
            if match:
                return json.loads(match.group(1))
    return None


def brief(findings: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {"specialist": f["specialist"], "focus": f["focus"], "summary": f["summary"]}
        for f in findings
    ]


def supervisor_message(
    alert: dict[str, Any], findings: list[dict[str, Any]], steps: int, max_steps: int
) -> HumanMessage:
    return HumanMessage(
        "\n".join(
            [
                tagged("alert", alert),
                tagged("findings", brief(findings)),
                f"Specialist dispatches used: {steps} of {max_steps}.",
            ]
        )
    )


def specialist_message(
    alert: dict[str, Any], focus: str, findings: list[dict[str, Any]]
) -> HumanMessage:
    return HumanMessage(
        "\n".join(
            [
                tagged("alert", alert),
                tagged("focus", focus),
                tagged("prior_findings", brief(findings)),
                SPECIALIST_FINISH,
            ]
        )
    )


def proposer_message(alert: dict[str, Any], findings: list[dict[str, Any]]) -> HumanMessage:
    return HumanMessage("\n".join([tagged("alert", alert), tagged("findings", findings)]))
