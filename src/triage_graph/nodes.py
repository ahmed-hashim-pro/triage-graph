"""Supervisor and proposer nodes."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from pydantic import ValidationError

from triage_graph.actions import Action, ActionNotAllowedError, require_allowed
from triage_graph.progress import emit
from triage_graph.prompts import (
    PROPOSE_TOOL,
    PROPOSER_SYSTEM,
    ROUTE_TOOL,
    SUPERVISOR_SYSTEM,
    ProposeArgs,
    RouteArgs,
    proposer_message,
    supervisor_message,
)
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
    """Turn the proposer's reply into a Proposal, refusing anything unsafe.

    Refusals are not errors: the proposal becomes page_human and records what
    the model asked for, so the human reviewing it sees the attempt.
    """
    proposal_id = uuid.uuid4().hex[:12]

    def escalate(diagnosis: str, note: str, refused: str | None = None) -> Proposal:
        return Proposal(
            id=proposal_id,
            diagnosis=diagnosis,
            action=Action.PAGE_HUMAN,
            target=service,
            rationale="The model's proposal could not be used, so the incident goes to a human.",
            refused_action=refused,
            note=note,
        )

    raw = _tool_args(reply, "propose_action")
    if raw is None:
        return escalate(
            reply.text or "No diagnosis returned.", "the model did not call propose_action"
        )
    try:
        args = ProposeArgs.model_validate(raw)
    except ValidationError as exc:
        return escalate("No usable diagnosis returned.", f"invalid propose_action arguments: {exc}")

    try:
        action = require_allowed(args.action)
    except ActionNotAllowedError:
        return escalate(
            args.diagnosis,
            f"the model proposed {args.action!r}, which is not on the allow-list; refused",
            refused=args.action[:100],
        )
    if args.target != service:
        return escalate(
            args.diagnosis,
            f"the model proposed {action} on {args.target!r}, but only the alerting service "
            f"({service}) may be acted on; refused",
            refused=f"{action} on {args.target[:60]}",
        )
    return Proposal(
        id=proposal_id,
        diagnosis=args.diagnosis,
        action=action,
        target=service,
        rationale=args.rationale,
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
