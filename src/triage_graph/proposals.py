"""Turning a model's proposal into a `Proposal`. Shared by the LangGraph and CrewAI versions."""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import ValidationError

from triage_graph.actions import Action, ActionNotAllowedError, require_allowed
from triage_graph.prompts import ProposeArgs
from triage_graph.schemas import Proposal


def validate_proposal(
    raw: dict[str, Any] | None,
    service: str,
    *,
    missing_note: str,
    fallback_diagnosis: str = "",
) -> Proposal:
    """Build a Proposal from the model's arguments, refusing anything unsafe.

    Refusals are not errors: the proposal becomes page_human and records what the
    model asked for, so the human reviewing it sees the attempt.
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

    if raw is None:
        return escalate(fallback_diagnosis or "No diagnosis returned.", missing_note)
    try:
        args = ProposeArgs.model_validate(raw)
    except ValidationError as exc:
        return escalate("No usable diagnosis returned.", f"invalid proposal arguments: {exc}")

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
