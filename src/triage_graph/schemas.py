"""Pydantic models used to validate data at node boundaries.

Graph state itself holds the `model_dump(mode="json")` form of these, so
checkpoints stay plain JSON and do not depend on class serialization.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from triage_graph.actions import Action

SERVICE_NAME = r"^[a-z0-9][a-z0-9-]{0,62}$"

Specialist = Literal["log_investigator", "metrics_investigator", "runbook_agent"]
SPECIALISTS: tuple[Specialist, ...] = ("log_investigator", "metrics_investigator", "runbook_agent")


class Alert(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    service: str = Field(pattern=SERVICE_NAME)
    title: str
    severity: Literal["critical", "high", "medium", "low"]
    fired_at: AwareDatetime
    description: str
    # Which fixture dataset backs the simulated log and metrics backends.
    dataset: str = Field(pattern=r"^[a-z0-9-]+$")
    labels: dict[str, str] = Field(default_factory=dict)


class ToolCallRecord(BaseModel):
    name: str
    args: dict[str, Any]
    output: str


class Finding(BaseModel):
    specialist: Specialist
    focus: str
    summary: str
    tool_calls: list[ToolCallRecord]


class Proposal(BaseModel):
    id: str
    diagnosis: str
    action: Action
    target: str = Field(pattern=SERVICE_NAME)
    rationale: str
    # Set when the model asked for something the allow-list refused.
    refused_action: str | None = None
    note: str | None = None


class HumanDecision(BaseModel):
    """What a human sends back through `Command(resume=...)`.

    An edit replaces the action; the target stays the alerting service.
    """

    model_config = ConfigDict(extra="forbid")

    decision: Literal["approve", "reject", "edit"]
    proposal_id: str
    action: str | None = None
    reason: str | None = None
    approver: str | None = None

    @model_validator(mode="after")
    def _edit_needs_action(self) -> HumanDecision:
        if self.decision == "edit" and self.action is None:
            raise ValueError("an edit must name the action to run instead")
        if self.decision != "edit" and self.action is not None:
            raise ValueError("only an edit may name an action")
        return self


class ExecutionRecord(BaseModel):
    proposal_id: str
    action: Action
    target: str
    dry_run: bool
    detail: str
    recorded_at: str
