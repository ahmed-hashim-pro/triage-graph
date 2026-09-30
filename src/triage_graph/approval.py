"""The human approval gate and the executor behind it."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langgraph.types import interrupt
from pydantic import ValidationError

from triage_graph.actions import ALLOWED_ACTIONS, Action, ActionNotAllowedError, require_allowed
from triage_graph.infra import DryRunInfra
from triage_graph.progress import emit
from triage_graph.schemas import HumanDecision
from triage_graph.state import TriageState


class ApprovalRequiredError(RuntimeError):
    """The executor was reached without a valid approval. This should be unreachable."""


def approval(state: TriageState) -> dict[str, Any]:
    proposal = state["proposal"]
    # LangGraph re-runs this node from the top when it resumes, so everything above
    # interrupt() must be deterministic and free of side effects.
    raw = interrupt(
        {
            "type": "approval_request",
            "alert_id": state["alert"]["id"],
            "proposal": proposal,
            "allowed_actions": sorted(ALLOWED_ACTIONS),
        }
    )

    def refuse(reason: str) -> dict[str, Any]:
        emit("decision", decision="refused", reason=reason)
        return {"next": "report", "outcome": "refused", "outcome_reason": reason}

    try:
        decision = HumanDecision.model_validate(raw)
    except ValidationError as exc:
        return refuse(f"malformed decision, treated as a rejection: {exc.errors()[0]['msg']}")
    if decision.proposal_id != proposal["id"]:
        return refuse(
            f"decision is for proposal {decision.proposal_id!r}, "
            f"but the pending proposal is {proposal['id']!r}"
        )
    if decision.decision == "edit":
        try:
            require_allowed(decision.action)
        except ActionNotAllowedError:
            return refuse(f"edited action {decision.action!r} is not on the allow-list")

    recorded = {"decision": decision.model_dump(mode="json")}
    emit("decision", **recorded["decision"])
    if decision.decision == "reject":
        reason = decision.reason or "no reason given"
        return {**recorded, "next": "report", "outcome": "rejected", "outcome_reason": reason}
    return {**recorded, "next": "executor"}


def authorize(state: TriageState) -> tuple[Action, str, str]:
    """Re-check the approval from scratch. Returns (action, target, proposal_id)."""
    proposal = state.get("proposal")
    raw_decision = state.get("decision")
    if not proposal or not raw_decision:
        raise ApprovalRequiredError("no approved proposal in state")
    try:
        decision = HumanDecision.model_validate(raw_decision)
    except ValidationError as exc:
        raise ApprovalRequiredError(f"recorded decision is invalid: {exc}") from exc
    if decision.decision not in ("approve", "edit"):
        raise ApprovalRequiredError(f"decision is {decision.decision!r}, not an approval")
    if decision.proposal_id != proposal["id"]:
        raise ApprovalRequiredError("the approval is for a different proposal")
    action = require_allowed(decision.action if decision.decision == "edit" else proposal["action"])
    if proposal["target"] != state["alert"]["service"]:
        raise ApprovalRequiredError("the proposal targets a service other than the alerting one")
    return action, proposal["target"], proposal["id"]


def make_executor(infra: DryRunInfra) -> Callable[[TriageState], dict[str, Any]]:
    def executor(state: TriageState) -> dict[str, Any]:
        action, target, proposal_id = authorize(state)
        record = infra.execute(action, target, proposal_id)
        emit("executed", **record.model_dump(mode="json"))
        return {
            "execution": record.model_dump(mode="json"),
            "outcome": "executed",
            "outcome_reason": record.detail,
        }

    return executor


def route_after_approval(state: TriageState) -> str:
    return state["next"]
