"""The markdown incident report. Built from state by code, not written by a model."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from triage_graph.state import DEFAULT_MAX_STEPS, TriageState

TITLES = {
    "log_investigator": "Logs",
    "metrics_investigator": "Metrics",
    "runbook_agent": "Runbooks",
}


def _call(name: str, args: dict[str, Any]) -> str:
    rendered = ", ".join(f"{k}={v!r}" for k, v in args.items())
    return f"`{name}({rendered})`"


def _render_output(name: str, output: str) -> list[str]:
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        return [f"  - tool error: {output[:300]}"]
    if name == "search_logs":
        lines = [f"  - {data['matched_lines']} matching lines"]
        lines += [
            f"  - {g['count']}x {g['level']} [{g['source']}] {g['example']} "
            f"({g['first_seen']} to {g['last_seen']})"
            for g in data["groups"][:8]
        ]
        return lines
    if name == "detect_metric_anomalies":
        lines = [
            f"  - **{a['metric']}**: {a['baseline']} -> {a['peak']} ({a['ratio']}x), "
            f"onset {a['onset']}"
            for a in data["anomalies"]
        ] or ["  - no anomalies"]
        normal = ", ".join(n["metric"] for n in data["normal"])
        return [*lines, f"  - within baseline: {normal or 'none'}"]
    if name == "search_runbooks":
        return [
            f"  - {r['file']} > {r['heading']} (score {r['score']}, "
            f"suggests {r['suggested_action'] or 'nothing'})"
            for r in data["results"]
        ] or ["  - no matching sections"]
    if name == "get_metric_series":
        points = data["points"]
        return [f"  - {len(points)} points, {data['step_minutes']}-minute buckets"]
    return [f"  - {output[:300]}"]


def _outcome_line(state: TriageState) -> str:
    outcome = state.get("outcome")
    execution = state.get("execution")
    if outcome == "executed" and execution:
        return (
            f"**executed** `{execution['action']}` on `{execution['target']}` "
            f"({'dry run' if execution['dry_run'] else 'live'})"
        )
    if outcome is None:
        return "**not executed**: a proposal was made but no approval was recorded"
    reason = state.get("outcome_reason")
    return f"**{outcome}**" + (f": {reason}" if reason else "")


def render_report(state: TriageState, *, estimated_tokens: bool = False) -> str:
    alert = state["alert"]
    usage = state.get("usage", {})
    token_note = " (estimated: chars / 4, fake model)" if estimated_tokens else ""
    lines = [
        f"# Incident report: {alert['title']}",
        "",
        f"- Alert: {alert['id']} ({alert['severity']}) on `{alert['service']}`, "
        f"fired {alert['fired_at']}",
        f"- Outcome: {_outcome_line(state)}",
        f"- Supervisor steps: {state.get('steps', 0)} of "
        f"{state.get('max_steps', DEFAULT_MAX_STEPS)}",
        f"- Model calls: {usage.get('model_calls', 0)}; tokens in/out: "
        f"{usage.get('input_tokens', 0)}/{usage.get('output_tokens', 0)}{token_note}",
    ]

    proposal = state.get("proposal")
    if proposal:
        lines += [
            "",
            "## Diagnosis",
            "",
            proposal["diagnosis"],
            "",
            "## Proposed action",
            "",
            f"`{proposal['action']}` on `{proposal['target']}` (proposal {proposal['id']})",
            "",
            proposal["rationale"],
        ]
        if proposal.get("note"):
            lines += ["", f"> Note: {proposal['note']}"]

    decision = state.get("decision")
    if decision:
        lines += ["", "## Human decision", "", f"- Decision: **{decision['decision']}**"]
        if decision.get("action"):
            lines.append(f"- Action chosen by the human: `{decision['action']}`")
        if decision.get("reason"):
            lines.append(f"- Reason: {decision['reason']}")

    execution = state.get("execution")
    if execution:
        lines += [
            "",
            "## Execution",
            "",
            f"- {execution['detail']}",
            f"- Recorded at {execution['recorded_at']}",
        ]

    lines += ["", "## Evidence"]
    findings = state.get("findings", [])
    if not findings:
        lines += ["", "No specialist reported."]
    for finding in findings:
        lines += [
            "",
            f"### {TITLES.get(finding['specialist'], finding['specialist'])}",
            "",
            f"Focus: {finding['focus'] or '(none given)'}",
            "",
            "Specialist's summary:",
            "",
            *(f"> {line}" for line in finding["summary"].splitlines()),
            "",
            "What the tools returned:",
            "",
        ]
        for call in finding["tool_calls"]:
            lines.append(f"- {_call(call['name'], call['args'])}")
            lines += _render_output(call["name"], call["output"])

    trail = state.get("supervisor_log", [])
    if trail:
        lines += ["", "## Supervisor decisions", ""]
        lines += [
            f"{i}. -> {entry['next']}: {entry.get('reason') or ''}".rstrip()
            for i, entry in enumerate(trail, start=1)
        ]
    return "\n".join(lines) + "\n"


def make_report_node(*, estimated_tokens: bool) -> Callable[[TriageState], dict[str, Any]]:
    def report(state: TriageState) -> dict[str, Any]:
        return {"report": render_report(state, estimated_tokens=estimated_tokens)}

    return report
