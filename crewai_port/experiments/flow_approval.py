"""Pause for approval with a CrewAI Flow, then resume from a different process.

    python -m crewai_port.experiments.flow_approval start DB LEDGER    # pauses, prints flow id
    python -m crewai_port.experiments.flow_approval resume DB LEDGER FLOW_ID DECISION

The flow runs the crew (investigate + propose), then pauses in @human_feedback via
a provider that raises HumanFeedbackPending. Findings are in
docs/LANGGRAPH_VS_CREWAI.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, ClassVar

from crewai.flow.async_feedback.types import HumanFeedbackPending, PendingFeedbackContext
from crewai.flow.flow import Flow, listen, start
from crewai.flow.human_feedback import HumanFeedbackResult, human_feedback
from crewai.flow.persistence.sqlite import SQLiteFlowPersistence
from pydantic import BaseModel

import crewai_port  # noqa: F401  (telemetry opt-outs)
from crewai_port.crew import run_triage
from crewai_port.fake_llm import FakeCrewLLM
from triage_graph.actions import require_allowed
from triage_graph.data import data_root, load_alert
from triage_graph.infra import DryRunInfra


class CountingLLM(FakeCrewLLM):
    calls: ClassVar[int] = 0

    def call(self, *args: Any, **kwargs: Any) -> Any:
        CountingLLM.calls += 1
        return super().call(*args, **kwargs)


class PauseForApproval:
    """A HumanFeedbackProvider that never answers in-process: it always pauses."""

    def request_feedback(self, context: PendingFeedbackContext, flow: Flow[Any]) -> str:
        raise HumanFeedbackPending(context=context)


class ApprovalState(BaseModel):
    id: str = ""
    proposal: dict[str, Any] = {}
    outcome: str = ""


class ApprovalFlow(Flow[ApprovalState]):
    ledger: ClassVar[Path | None] = None

    @start()
    @human_feedback(
        message="Approve this proposal? Reply 'approve' or 'reject'.", provider=PauseForApproval()
    )
    def propose(self) -> dict[str, Any]:
        alert = load_alert(data_root() / "fixtures" / "alerts" / "high-latency.json")
        result = run_triage(alert, CountingLLM)
        assert result.proposal is not None
        self.state.proposal = result.proposal.model_dump(mode="json")
        return self.state.proposal

    @listen(propose)
    def act(self, feedback: HumanFeedbackResult) -> str:
        # The approval check is ours; CrewAI hands over the human's text as-is.
        if feedback.feedback.strip() == "approve":
            proposal = self.state.proposal
            record = DryRunInfra(self.ledger).execute(
                require_allowed(proposal["action"]), proposal["target"], proposal["id"]
            )
            self.state.outcome = f"executed {record.action}"
        else:
            self.state.outcome = "rejected"
        return self.state.outcome


def main() -> None:
    mode, db, ledger = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    ApprovalFlow.ledger = ledger
    persistence = SQLiteFlowPersistence(db)
    if mode == "start":
        result = ApprovalFlow(persistence=persistence).kickoff()
        paused = isinstance(result, HumanFeedbackPending)
        flow_id = result.context.flow_id if paused else None
        print(json.dumps({"paused": paused, "flow_id": flow_id, "llm_calls": CountingLLM.calls}))
        return
    flow_id, decision = sys.argv[4], sys.argv[5]
    flow = ApprovalFlow.from_pending(flow_id, persistence)
    outcome = flow.resume(decision)
    print(json.dumps({"outcome": outcome, "llm_calls_in_resume": CountingLLM.calls}))


if __name__ == "__main__":
    main()
