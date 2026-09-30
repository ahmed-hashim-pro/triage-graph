"""Supervisor and proposer nodes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from pydantic import ValidationError

from triage_graph.progress import emit
from triage_graph.prompts import (
    PROPOSE_TOOL,
    PROPOSER_SYSTEM,
    ROUTE_TOOL,
    SUPERVISOR_SYSTEM,
    RouteArgs,
    proposer_message,
    supervisor_message,
)
from triage_graph.proposals import validate_proposal
from triage_graph.schemas import Proposal
from triage_graph.specialists import usage_of
from triage_graph.state import DEFAULT_MAX_STEPS, TriageState

INSUFFICIENT_EVIDENCE = "insufficient evidence, escalating"


def _tool_args(reply: AIMessage, name: str) -> dict[str, Any] | None:
    return next((c["args"] for c in reply.tool_calls if c["name"] == name), None)


def make_supervisor(model: BaseChatModel) -> Callable[[TriageState], dict[str, Any]]:
    bound = model.bind_tools([ROUTE_TOOL])

    def supervisor(state: TriageState) -> dict[str, Any]:
        steps = state.get("steps", 0)
        max_steps = state.get("max_steps", DEFAULT_MAX_STEPS)
        findings = state.get("findings", [])

        if steps >= max_steps:
            reason = (
                f"{INSUFFICIENT_EVIDENCE}: no proposal after {steps} supervisor steps "
                f"(limit {max_steps})"
            )
            emit("escalate", reason=reason, max_steps=max_steps)
            return {"next": "escalate", "outcome": "escalated", "outcome_reason": reason}

        reply = bound.invoke(
            [
                SystemMessage(SUPERVISOR_SYSTEM),
                supervisor_message(state["alert"], findings, steps, max_steps),
            ]
        )
        update: dict[str, Any] = {"usage": usage_of([reply])}
        try:
            route = RouteArgs.model_validate(_tool_args(reply, "route"))
        except ValidationError:
            route = None

        if route is None:
            # An unusable reply still spends a step, so a model that never answers
            # properly runs into the limit instead of looping forever.
            entry = {"step": steps + 1, "next": "retry", "reason": "no valid route call"}
            emit("route", **entry, max_steps=max_steps)
            return {**update, "steps": steps + 1, "next": "retry", "supervisor_log": [entry]}

        if route.next == "propose":
            entry = {"step": steps, "next": "propose", "reason": route.reason}
            emit("route", **entry, max_steps=max_steps)
            return {**update, "next": "propose", "supervisor_log": [entry]}

        entry = {
            "step": steps + 1,
            "next": route.next,
            "focus": route.focus,
            "reason": route.reason,
        }
        emit("route", **entry, max_steps=max_steps)
        return {
            **update,
            "steps": steps + 1,
            "next": route.next,
            "focus": route.focus,
            "supervisor_log": [entry],
        }

    return supervisor


def proposal_from(reply: AIMessage, service: str) -> Proposal:
    return validate_proposal(
        _tool_args(reply, "propose_action"),
        service,
        missing_note="the model did not call propose_action",
        fallback_diagnosis=reply.text,
    )


def make_proposer(model: BaseChatModel) -> Callable[[TriageState], dict[str, Any]]:
    bound = model.bind_tools([PROPOSE_TOOL])

    def proposer(state: TriageState) -> dict[str, Any]:
        alert = state["alert"]
        reply = bound.invoke(
            [SystemMessage(PROPOSER_SYSTEM), proposer_message(alert, state.get("findings", []))]
        )
        proposal = proposal_from(reply, alert["service"])
        emit("proposal", **proposal.model_dump(mode="json"))
        return {"proposal": proposal.model_dump(mode="json"), "usage": usage_of([reply])}

    return proposer
