import json
import os

import pytest

from evals.harness import (
    PRICES,
    VARIANTS,
    BudgetGuard,
    Plan,
    Price,
    fit_runs,
    load_expected,
    render_report,
    run_eval,
    run_graph,
    write_results,
)
from triage_graph.fake_model import FakeTriageModel
from triage_graph.llm import describe_model, make_model

SCENARIOS = sorted(load_expected())


def plan_for(model, runs=2, scenarios=SCENARIOS, variants=tuple(VARIANTS), price=None, cap=None):
    return Plan(
        model=describe_model(model),
        scenarios=list(scenarios),
        variants=list(variants),
        runs=runs,
        requested_runs=runs,
        price=price,
        cap_usd=cap,
        est_input_tokens=0,
        est_output_tokens=0,
    )


def test_fake_model_eval_matches_every_scenario_in_both_variants():
    model = FakeTriageModel()
    result = run_eval(model, plan_for(model))
    assert len(result.records) == len(SCENARIOS) * len(VARIANTS) * 2
    assert all(r.matched and r.error is None for r in result.records)
    assert all(r.steps == 3 and r.model_calls == 12 for r in result.records)
    report = render_report(result)
    assert "**This is the fake model.**" in report
    assert "| **all** | | 8/8 (100%) | 8/8 (100%) |" in report
    assert "flaky" not in report.split("## Match rate")[1]


class FlipFlop(FakeTriageModel):
    """Alternates between the right answer and page_human on every proposal."""

    calls: int = 0

    def propose(self, messages):
        self.calls += 1
        reply = super().propose(messages)
        if self.calls % 2 == 0:
            args = {**reply.tool_calls[0]["args"], "action": "page_human"}
            return self.tool_call("propose_action", args)
        return reply


def test_inconsistent_runs_are_reported_as_flaky():
    model = FlipFlop()
    result = run_eval(
        model, plan_for(model, scenarios=["high-latency"], variants=["with-suggestions"])
    )
    report = render_report(result)
    assert "| high-latency | scale_up | 1/2 **flaky**: scale_up x1, page_human x1 |" in report


def test_budget_guard_refuses_the_call_that_could_cross_the_cap():
    model = FakeTriageModel()
    price = Price(input_per_mtok=100.0, output_per_mtok=100.0)
    guard = BudgetGuard(price, cap_usd=0.5, max_output_tokens=1000)
    result = run_eval(model, plan_for(model, runs=5, price=price, cap=0.5), guard=guard)
    assert result.stopped_by_budget
    assert result.records[-1].outcome == "budget"
    assert 0 < guard.spent_usd <= 0.5
    assert "**stopped early by the budget cap**" in render_report(result)


def test_budget_guard_sees_calls_inside_specialist_subgraphs():
    guard = BudgetGuard(Price(0.0, 0.0), cap_usd=1.0, max_output_tokens=1000)
    state = run_graph(FakeTriageModel(), "memory-leak", True, "t", [guard]).state
    assert guard.calls == state["usage"]["model_calls"] == 12


def test_hidden_variant_removes_suggestions_from_everything_the_agents_see():
    shown = run_graph(FakeTriageModel(), "high-latency", True, "a").state
    hidden = run_graph(FakeTriageModel(), "high-latency", False, "b").state
    assert "Suggested action" in json.dumps(shown["findings"])
    for text in (json.dumps(hidden["findings"]), json.dumps(hidden["supervisor_log"])):
        assert "Suggested action" not in text
        assert "suggested_action" not in text
        assert "scale_up" not in text
    assert hidden["proposal"]["action"] == "scale_up"


def test_plan_is_cut_to_fit_the_budget():
    assert fit_runs(5, cells=8, cost_per_run=0.10, cap_usd=2.0) == 2
    assert fit_runs(5, cells=8, cost_per_run=0.01, cap_usd=2.0) == 5
    assert fit_runs(5, cells=8, cost_per_run=0.30, cap_usd=2.0) == 0
    assert "claude-opus-5-5" in PRICES


def test_results_files_never_contain_credentials(tmp_path, monkeypatch):
    pytest.importorskip("langchain_anthropic", reason="needs the anthropic extra")
    sentinel = "sk-ant-api03-SENTINEL-must-not-be-written"
    monkeypatch.setenv("ANTHROPIC_API_KEY", sentinel)
    real = make_model("anthropic")
    fake = FakeTriageModel()
    result = run_eval(fake, plan_for(fake, runs=1, scenarios=["high-latency"]))
    result.plan = plan_for(
        real, runs=1, scenarios=["high-latency"], price=PRICES["claude-opus-5-5"], cap=2.0
    )
    for path in write_results(result, tmp_path, "run"):
        assert sentinel not in path.read_text()
    assert result.plan.model == "anthropic:claude-opus-5-5"


def test_provider_errors_are_recorded_not_raised():
    class Broken(FakeTriageModel):
        def supervise(self, messages):
            raise TimeoutError("provider timed out")

    model = Broken()
    result = run_eval(
        model, plan_for(model, runs=1, scenarios=["high-latency"], variants=["with-suggestions"])
    )
    [record] = result.records
    assert (record.outcome, record.matched) == ("error", False)
    assert "TimeoutError: provider timed out" in record.error


@pytest.mark.real_model
@pytest.mark.skipif(
    not (os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("TRIAGE_EVAL_REAL") == "1"),
    reason="needs ANTHROPIC_API_KEY and TRIAGE_EVAL_REAL=1 (it spends money)",
)
def test_real_model_eval(tmp_path):
    from evals.run import main

    assert (
        main(["--provider", "anthropic", "--yes", "--budget-usd", "2.0", "--out", str(tmp_path)])
        == 0
    )
