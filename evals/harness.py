"""Run each scenario N times and compare the proposed action with the expected one."""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

from triage_graph.data import data_root, load_alert
from triage_graph.fake_model import FakeTriageModel
from triage_graph.graph import build_graph
from triage_graph.runner import start

# variant name -> whether the runbooks' "Suggested action" lines are shown
VARIANTS = {"with-suggestions": True, "suggestions-hidden": False}


@dataclass(frozen=True)
class Price:
    input_per_mtok: float
    output_per_mtok: float

    def cost(self, input_tokens: float, output_tokens: float) -> float:
        return (input_tokens * self.input_per_mtok + output_tokens * self.output_per_mtok) / 1e6


# USD per million tokens: Anthropic list prices for these models as of 2026-09-25.
# Check current pricing before trusting an estimate.
PRICES = {
    "claude-fable-5-1": Price(10.0, 50.0),
    "claude-opus-5-5": Price(4.0, 20.0),
    "claude-sonnet-5-5": Price(2.0, 10.0),
    "claude-haiku-4-5": Price(1.0, 5.0),
}

# Assumed output tokens per model call, thinking included. These are guesses, used
# only for the estimate printed before a run; the budget guard uses real usage.
OUTPUT_TOKENS_PER_CALL = {"low": 400, "medium": 800, "high": 1500, "xhigh": 2500, "max": 4000}
# Effort a model uses when none is sent. Models not listed don't think by default.
DEFAULT_EFFORT = {
    "claude-opus-5-5": "medium",
    "claude-sonnet-5-5": "high",
    "claude-fable-5-1": "high",
}
NO_THINKING_OUTPUT_TOKENS = 300
# The API rejects an effort setting for these.
NO_EFFORT_MODELS = frozenset({"claude-haiku-4-5"})
# Real models write longer summaries than the fake one, and later prompts carry them.
INPUT_GROWTH = 1.5
CALL_GROWTH = 1.25
# Tool schemas and system prompts are not in the message text the guard can see.
TOOL_SCHEMA_ALLOWANCE = 2000


def load_expected() -> dict[str, str]:
    data = json.loads((data_root() / "fixtures" / "expected.json").read_text())
    return {scenario: entry["expected_action"] for scenario, entry in data.items()}


class BudgetExhausted(RuntimeError):
    pass


class BudgetGuard(BaseCallbackHandler):
    """Refuses a model call if its worst-case cost could take spend past the cap.

    Checked before every call (including calls inside specialist subgraphs), so the
    total can't exceed the cap by more than a mis-estimated prompt size. Prompt
    tokens are taken as characters / 2, twice the usual characters / 4 rule.
    """

    raise_error = True

    def __init__(self, price: Price, cap_usd: float, max_output_tokens: int) -> None:
        self.price = price
        self.cap_usd = cap_usd
        self.max_output_tokens = max_output_tokens
        self.spent_usd = 0.0
        self.calls = 0

    def on_chat_model_start(
        self, serialized: dict[str, Any], messages: list[list[BaseMessage]], **kwargs: Any
    ) -> None:
        prompt_chars = sum(len(str(m.content)) for batch in messages for m in batch)
        worst = self.price.cost(prompt_chars / 2 + TOOL_SCHEMA_ALLOWANCE, self.max_output_tokens)
        if self.spent_usd + worst > self.cap_usd:
            raise BudgetExhausted(
                f"next call could cost up to ${worst:.3f}; spent ${self.spent_usd:.3f} "
                f"of ${self.cap_usd:.2f}"
            )

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        for generations in response.generations:
            for generation in generations:
                message = getattr(generation, "message", None)
                usage = getattr(message, "usage_metadata", None) or {}
                self.calls += 1
                self.spent_usd += self.price.cost(
                    usage.get("input_tokens", 0), usage.get("output_tokens", 0)
                )


@dataclass
class Plan:
    model: str
    scenarios: list[str]
    variants: list[str]
    runs: int
    requested_runs: int
    price: Price | None
    cap_usd: float | None
    est_input_tokens: int
    est_output_tokens: int

    @property
    def total_runs(self) -> int:
        return self.runs * len(self.scenarios) * len(self.variants)

    @property
    def est_cost_per_run(self) -> float:
        return self.price.cost(self.est_input_tokens, self.est_output_tokens) if self.price else 0.0

    def describe(self) -> str:
        lines = [
            f"model:      {self.model}",
            f"scenarios:  {', '.join(self.scenarios)}",
            f"variants:   {', '.join(self.variants)}",
            f"runs:       {self.runs} per scenario and variant, {self.total_runs} in total"
            + (f" (asked for {self.requested_runs})" if self.runs != self.requested_runs else ""),
        ]
        if self.price is not None and self.cap_usd is not None:
            lines += [
                f"estimate:   ~{self.est_input_tokens:,} input + "
                f"~{self.est_output_tokens:,} output tokens per run -> "
                f"${self.est_cost_per_run:.3f} per run, "
                f"${self.est_cost_per_run * self.total_runs:.2f} in total",
                f"hard cap:   ${self.cap_usd:.2f}; a model call that could cross it is not made",
                "            (the per-run token figures are assumptions, see evals/harness.py)",
            ]
        return "\n".join(lines)


def estimate_tokens(scenarios: list[str], model_name: str, effort: str | None) -> tuple[int, int]:
    """Per-run (input, output) token estimate, scaled up from a fake-model dry run.

    The fake model's input count already includes tool schemas and system prompts.
    """
    effort = effort or DEFAULT_EFFORT.get(model_name)
    per_call = OUTPUT_TOKENS_PER_CALL[effort] if effort else NO_THINKING_OUTPUT_TOKENS
    inputs, calls = [], []
    for scenario in scenarios:
        state = run_graph(FakeTriageModel(), scenario, True, f"estimate-{scenario}").state
        inputs.append(state["usage"]["input_tokens"])
        calls.append(state["usage"]["model_calls"])
    return round(fmean(inputs) * INPUT_GROWTH), round(fmean(calls) * CALL_GROWTH * per_call)


def fit_runs(requested: int, cells: int, cost_per_run: float, cap_usd: float) -> int:
    if cost_per_run <= 0:
        return requested
    return min(requested, math.floor(cap_usd / (cost_per_run * cells)))


@dataclass
class RunRecord:
    scenario: str
    variant: str
    run: int
    expected: str
    proposed: str | None
    matched: bool
    outcome: str
    refused_action: str | None
    steps: int
    model_calls: int
    input_tokens: int
    output_tokens: int
    seconds: float
    error: str | None = None

    @property
    def label(self) -> str:
        return self.proposed or self.outcome


def run_graph(
    model: BaseChatModel,
    scenario: str,
    suggestions: bool,
    thread_id: str,
    callbacks: list[BaseCallbackHandler] | None = None,
) -> Any:
    alert = load_alert(data_root() / "fixtures" / "alerts" / f"{scenario}.json")
    graph = build_graph(model, runbook_suggestions=suggestions)
    return start(graph, alert.model_dump(mode="json"), thread_id=thread_id, callbacks=callbacks)


def run_once(
    model: BaseChatModel,
    scenario: str,
    variant: str,
    run: int,
    expected: str,
    guard: BudgetGuard | None,
) -> RunRecord:
    began = time.monotonic()
    base = {"scenario": scenario, "variant": variant, "run": run, "expected": expected}

    def failed(outcome: str, error: str) -> RunRecord:
        return RunRecord(
            **base,
            proposed=None,
            matched=False,
            outcome=outcome,
            refused_action=None,
            steps=0,
            model_calls=0,
            input_tokens=0,
            output_tokens=0,
            seconds=time.monotonic() - began,
            error=error[:300],
        )

    try:
        result = run_graph(
            model,
            scenario,
            VARIANTS[variant],
            f"eval-{scenario}-{variant}-{run}",
            [guard] if guard else None,
        )
    except BudgetExhausted as exc:
        return failed("budget", str(exc))
    except Exception as exc:
        # Provider errors (rate limits, refusals, timeouts) are results to report.
        return failed("error", f"{type(exc).__name__}: {exc}")

    state = result.state
    proposal = result.pending["proposal"] if result.pending else None
    usage = state.get("usage", {})
    return RunRecord(
        **base,
        proposed=proposal["action"] if proposal else None,
        matched=bool(proposal) and proposal["action"] == expected,
        outcome="proposed" if proposal else state.get("outcome", "unknown"),
        refused_action=proposal.get("refused_action") if proposal else None,
        steps=state.get("steps", 0),
        model_calls=usage.get("model_calls", 0),
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        seconds=time.monotonic() - began,
    )


@dataclass
class EvalResult:
    plan: Plan
    records: list[RunRecord] = field(default_factory=list)
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    spent_usd: float | None = None

    @property
    def stopped_by_budget(self) -> bool:
        return any(r.outcome == "budget" for r in self.records)

    def cell(self, scenario: str, variant: str) -> list[RunRecord]:
        return [
            r
            for r in self.records
            if r.scenario == scenario and r.variant == variant and r.outcome != "budget"
        ]


def run_eval(
    model: BaseChatModel,
    plan: Plan,
    *,
    guard: BudgetGuard | None = None,
    progress: Callable[[RunRecord], None] | None = None,
) -> EvalResult:
    """Run every scenario and variant `plan.runs` times.

    Runs are interleaved (run 1 of everything, then run 2 ...), so if the budget
    stops the eval early, every cell has about the same number of runs.
    """
    expected = load_expected()
    result = EvalResult(plan)
    for run in range(1, plan.runs + 1):
        for scenario in plan.scenarios:
            for variant in plan.variants:
                record = run_once(model, scenario, variant, run, expected[scenario], guard)
                result.records.append(record)
                if progress:
                    progress(record)
                if record.outcome == "budget":
                    result.spent_usd = guard.spent_usd if guard else None
                    return result
    result.spent_usd = guard.spent_usd if guard else None
    return result


def _cell_text(records: list[RunRecord]) -> str:
    if not records:
        return "not run"
    matches = sum(r.matched for r in records)
    text = f"{matches}/{len(records)}"
    labels = Counter(r.label for r in records)
    if len(labels) > 1:
        text += " **flaky**: " + ", ".join(f"{k} x{v}" for k, v in labels.most_common())
    elif matches == 0:
        text += f" (got {records[0].label})"
    return text


def render_report(result: EvalResult) -> str:
    plan = result.plan
    fake = plan.model == "fake"
    records = [r for r in result.records if r.outcome != "budget"]
    total_in = sum(r.input_tokens for r in records)
    total_out = sum(r.output_tokens for r in records)
    errors = [r for r in records if r.outcome == "error"]

    lines = [
        f"# Eval report: {plan.model}",
        "",
        f"- Started: {result.started_at}",
        f"- Runs per scenario and variant: {plan.runs}"
        + (
            f" (asked for {plan.requested_runs}; reduced to fit the budget)"
            if plan.runs != plan.requested_runs
            else ""
        ),
        f"- Runs completed: {len(records)} of {plan.total_runs}; errors: {len(errors)}"
        + ("; **stopped early by the budget cap**" if result.stopped_by_budget else ""),
        f"- Tokens in/out: {total_in:,}/{total_out:,}"
        + (" (chars / 4 estimates from the fake model, not real counts)" if fake else ""),
    ]
    if plan.price is not None:
        spent = result.spent_usd if result.spent_usd is not None else 0.0
        lines.append(
            f"- Cost at list price: ${spent:.2f} (estimated before running: "
            f"${plan.est_cost_per_run * plan.total_runs:.2f}; cap ${plan.cap_usd:.2f})"
        )

    lines += ["", "## Read this first", ""]
    if fake:
        lines.append(
            "- **This is the fake model.** Its rules were written against these fixtures, so "
            "matching every scenario is by construction. It says nothing about how an LLM "
            "would do. It exists so the eval harness is exercised by the offline test suite."
        )
    lines += [
        "- A run matches when the proposed action equals the scenario's expected action. "
        "A run that ends in escalation (step limit) is a miss, even where the expected "
        "action is page_human.",
        "- **Runbook bias.** Each service's runbook has a section whose `Suggested action` is "
        'the expected answer. The `with-suggestions` column partly measures "found and '
        'followed the runbook".',
        "- The `suggestions-hidden` column removes the `Suggested action` lines at the source "
        "(the runbook tool), so no agent sees them. The sections' prose still describes the "
        'fix (for example "Roll back to the previous version"), so this is a partial '
        "ablation, not a runbook-free baseline.",
    ]
    if plan.runs < 3:
        lines.append(
            f"- With {plan.runs} run(s) per cell, a flaky scenario can easily look stable. "
            '"flaky" below means the runs disagreed; its absence proves little.'
        )

    variants = plan.variants
    lines += [
        "",
        "## Match rate",
        "",
        "| Scenario | Expected | " + " | ".join(variants) + " |",
        "|---|---|" + "---|" * len(variants),
    ]
    expected = load_expected()
    for scenario in plan.scenarios:
        cells = [_cell_text(result.cell(scenario, v)) for v in variants]
        lines.append(f"| {scenario} | {expected[scenario]} | " + " | ".join(cells) + " |")
    totals = []
    for variant in variants:
        done = [r for r in records if r.variant == variant]
        hits = sum(r.matched for r in done)
        totals.append(f"{hits}/{len(done)} ({hits / len(done):.0%})" if done else "not run")
    lines.append("| **all** | | " + " | ".join(totals) + " |")

    lines += [
        "",
        "## Averages per run",
        "",
        "| Scenario | Variant | Supervisor steps | Model calls | Input tokens | Output tokens "
        "| Seconds |",
        "|---|---|---|---|---|---|---|",
    ]
    for scenario in plan.scenarios:
        for variant in variants:
            done = [r for r in result.cell(scenario, variant) if r.outcome != "error"]
            if not done:
                continue
            lines.append(
                f"| {scenario} | {variant} | {fmean(r.steps for r in done):.1f} | "
                f"{fmean(r.model_calls for r in done):.1f} | "
                f"{fmean(r.input_tokens for r in done):,.0f} | "
                f"{fmean(r.output_tokens for r in done):,.0f} | "
                f"{fmean(r.seconds for r in done):.1f} |"
            )

    lines += ["", "## Every run", ""]
    for r in result.records:
        extra = f" (model asked for {r.refused_action!r}, refused)" if r.refused_action else ""
        extra += f" error: {r.error}" if r.error else ""
        mark = "match" if r.matched else "miss"
        lines.append(f"- {r.scenario} / {r.variant} / run {r.run}: {r.label}, {mark}{extra}")
    return "\n".join(lines) + "\n"


def write_results(result: EvalResult, out_dir: Path, stem: str) -> tuple[Path, Path]:
    """Write the markdown report and the raw records. Only explicit fields are written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / f"{stem}.md"
    raw = out_dir / f"{stem}.json"
    report.write_text(render_report(result))
    plan = result.plan
    raw.write_text(
        json.dumps(
            {
                "model": plan.model,
                "started_at": result.started_at,
                "runs_per_cell": plan.runs,
                "requested_runs": plan.requested_runs,
                "scenarios": plan.scenarios,
                "variants": plan.variants,
                "cap_usd": plan.cap_usd,
                "spent_usd": result.spent_usd,
                "records": [asdict(r) for r in result.records],
            },
            indent=2,
        )
        + "\n"
    )
    return report, raw
