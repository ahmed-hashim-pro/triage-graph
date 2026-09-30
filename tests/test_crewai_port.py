import importlib.util
import json
from typing import ClassVar

import pytest

if importlib.util.find_spec("crewai") is None:
    pytest.skip("the crewai extra is not installed", allow_module_level=True)

# crewai_port sets the telemetry opt-outs, so it is imported before anything from crewai.
import crewai_port  # noqa: F401, I001

from conftest import EXPECTED, alert_dict, alert_path
from crewai.hooks import get_before_tool_call_hooks

from crewai_port.crew import run_triage
from crewai_port.fake_llm import DELEGATE, FakeCrewLLM, tool_call
from crewai_port.roles import LOGS, METRICS, PROPOSER, RUNBOOKS
from triage_graph.data import load_alert
from triage_graph.fake_model import FakeTriageModel
from triage_graph.graph import build_graph
from triage_graph.nodes import INSUFFICIENT_EVIDENCE
from triage_graph.runner import start


def triage(scenario="high-latency", llm=FakeCrewLLM, **kwargs):
    return run_triage(load_alert(alert_path(scenario)), llm, **kwargs)


class Counted(FakeCrewLLM):
    proposer_calls: ClassVar[int] = 0

    def propose(self, messages):
        type(self).proposer_calls += 1
        return super().propose(messages)


def test_crew_proposes_what_the_langgraph_version_proposes(scenario):
    crew_result = triage(scenario)
    graph_result = start(build_graph(FakeTriageModel()), alert_dict(scenario), thread_id="t")
    assert crew_result.status == "proposed"
    assert crew_result.proposal.action == EXPECTED[scenario]["expected_action"]
    assert crew_result.proposal.action == graph_result.pending["proposal"]["action"]
    assert crew_result.delegations == [LOGS, METRICS, RUNBOOKS]


def test_token_usage_counts_each_call_once():
    usage = triage().token_usage
    assert usage["successful_requests"] == 12
    assert usage["total_tokens"] > 0


class StubbornSupervisor(Counted):
    def supervise(self, messages):
        return self.delegate(messages, LOGS)


def test_step_limit_escalates_and_skips_the_proposer():
    StubbornSupervisor.proposer_calls = 0
    result = triage(llm=StubbornSupervisor, max_steps=3)
    assert result.status == "escalated"
    assert result.reason.startswith(INSUFFICIENT_EVIDENCE)
    assert result.delegations == [LOGS] * 3
    assert result.proposal is None
    assert StubbornSupervisor.proposer_calls == 0


def test_too_few_steps_for_the_evidence_also_escalates():
    result = triage(max_steps=2)
    assert result.status == "escalated"
    assert result.delegations == [LOGS, METRICS]


class DelegatesTwiceAtOnce(FakeCrewLLM):
    """Asks for two specialists in a single turn; CrewAI runs such batches in parallel."""

    def supervise(self, messages):
        if any(m["role"] == "tool" for m in messages):
            return self.final_answer(messages)
        return self.delegate(messages, LOGS) + self.delegate(messages, METRICS)


def test_step_limit_holds_when_delegations_run_in_parallel():
    result = triage(llm=DelegatesTwiceAtOnce, max_steps=1)
    assert result.status == "escalated"
    assert len(result.delegations) == 1


class RogueProposer(FakeCrewLLM):
    def propose(self, messages):
        args = json.loads(super().propose(messages))
        return json.dumps({**args, "action": "drop_database"})


def test_off_list_action_from_the_proposer_is_refused():
    proposal = triage(llm=RogueProposer).proposal
    assert proposal.action == "page_human"
    assert proposal.refused_action == "drop_database"
    assert "not on the allow-list" in proposal.note


class ChattyProposer(FakeCrewLLM):
    def propose(self, messages):
        return "I would probably scale it up."


def test_non_json_proposal_becomes_page_human():
    proposal = triage(llm=ChattyProposer).proposal
    assert proposal.action == "page_human"
    assert proposal.note == "the proposer did not return a JSON object"


class AsksProposerFirst(Counted):
    def supervise(self, messages):
        if not any(m.get("name") == DELEGATE for m in messages if m["role"] == "tool"):
            return tool_call(
                "call_early",
                DELEGATE,
                {"task": "propose something", "context": "", "coworker": PROPOSER},
            )
        return super().supervise(messages)


def test_supervisor_cannot_delegate_to_the_proposer():
    AsksProposerFirst.proposer_calls = 0
    result = triage(llm=AsksProposerFirst)
    assert result.status == "proposed"
    assert result.delegations == [LOGS, METRICS, RUNBOOKS]
    # The blocked attempt used a step, and the proposer only ran for its own task.
    assert AsksProposerFirst.proposer_calls == 1


class Crashing(FakeCrewLLM):
    def supervise(self, messages):
        raise RuntimeError("provider down")


def test_step_limit_hook_is_removed_after_every_run():
    before = len(get_before_tool_call_hooks())
    triage()
    with pytest.raises(RuntimeError, match="provider down"):
        triage(llm=Crashing)
    assert len(get_before_tool_call_hooks()) == before
