import json

from conftest import alert_dict
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from triage_graph.fake_model import FakeTriageModel
from triage_graph.lc_tools import SPECIALIST_TOOLS
from triage_graph.prompts import ROUTE_TOOL, supervisor_message


def route(findings):
    model = FakeTriageModel().bind_tools([ROUTE_TOOL])
    reply = model.invoke([supervisor_message(alert_dict("high-latency"), findings, 0, 6)])
    return reply.tool_calls[0]["args"]["next"]


def test_supervisor_dispatches_each_specialist_then_proposes():
    finding = lambda name: {"specialist": name, "focus": "", "summary": "", "tool_calls": []}  # noqa: E731
    assert route([]) == "log_investigator"
    assert route([finding("log_investigator")]) == "metrics_investigator"
    assert route([finding("log_investigator"), finding("metrics_investigator")]) == "runbook_agent"
    everyone = [finding(n) for n in ("log_investigator", "metrics_investigator", "runbook_agent")]
    assert route(everyone) == "propose"


def test_role_is_chosen_from_the_bound_tools():
    alert = alert_dict("error-spike-after-deploy")
    prompt = [HumanMessage(f"<alert>{json.dumps(alert)}</alert>")]
    for specialist, first_tool in [
        ("log_investigator", "search_logs"),
        ("metrics_investigator", "detect_metric_anomalies"),
        ("runbook_agent", "search_runbooks"),
    ]:
        reply = FakeTriageModel().bind_tools(SPECIALIST_TOOLS[specialist]).invoke(prompt)
        assert reply.tool_calls[0]["name"] == first_tool


def test_replies_are_deterministic_and_report_estimated_usage():
    prompt = [HumanMessage(f"<alert>{json.dumps(alert_dict('memory-leak'))}</alert>")]
    model = FakeTriageModel().bind_tools(SPECIALIST_TOOLS["log_investigator"])
    first, second = model.invoke(prompt), model.invoke(prompt)
    assert first.tool_calls == second.tool_calls
    assert first.usage_metadata["input_tokens"] > 0


def test_investigator_summarizes_after_its_tool_calls():
    alert = alert_dict("memory-leak")
    model = FakeTriageModel().bind_tools(SPECIALIST_TOOLS["metrics_investigator"])
    call = model.invoke([HumanMessage(f"<alert>{json.dumps(alert)}</alert>")])
    output = json.dumps({"anomalies": [], "normal": [{"metric": "rps"}]})
    reply = model.invoke(
        [
            HumanMessage(f"<alert>{json.dumps(alert)}</alert>"),
            AIMessage("", tool_calls=call.tool_calls),
            ToolMessage(
                output, tool_call_id=call.tool_calls[0]["id"], name="detect_metric_anomalies"
            ),
        ]
    )
    assert not reply.tool_calls
    assert "no anomalous metrics" in reply.content
