"""A deterministic, rule-based stand-in for an LLM, so everything runs offline.

It is a `BaseChatModel`, so the graph calls it exactly as it would call a real
model: it receives the same messages and bound tools, and returns tool calls
that go through the same parsing and validation. It decides which role it is
playing from the tools it has been given.

Its "reasoning" is a handful of keyword and threshold rules written for the
fixtures in this repo. It is useful for exercising the graph; its accuracy on
the fixtures says nothing about how an LLM would do.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from typing import Any

from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool

from triage_graph.actions import Action
from triage_graph.prompts import read_tag
from triage_graph.schemas import SPECIALISTS

CHANGE_PATTERN = "deploy|rolled out|config|flag|restart|scal"
DEPLOY = re.compile(r"\bdeploy\b|rolled out", re.IGNORECASE)
DEFAULT_FOCUS = {
    "log_investigator": "errors and warnings around the alert, and recent changes",
    "metrics_investigator": "which metrics moved, and when",
    "runbook_agent": "runbook guidance for these symptoms",
}


def _iso(ts: datetime) -> str:
    return ts.isoformat().replace("+00:00", "Z")


def _tool_results(messages: Sequence[BaseMessage]) -> list[tuple[str, Any]]:
    results: list[tuple[str, Any]] = []
    for message in messages:
        if isinstance(message, ToolMessage):
            try:
                content: Any = json.loads(str(message.content))
            except json.JSONDecodeError:
                content = str(message.content)
            results.append((message.name or "", content))
    return results


class FakeTriageModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "triage-fake"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        return self.bind(tools=[convert_to_openai_tool(t) for t in tools], **kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        tools = kwargs.get("tools") or []
        names = {t["function"]["name"] for t in tools}
        if "route" in names:
            reply = self.supervise(messages)
        elif "propose_action" in names:
            reply = self.propose(messages)
        elif "search_logs" in names:
            reply = self.investigate_logs(messages)
        elif "detect_metric_anomalies" in names:
            reply = self.investigate_metrics(messages)
        elif "search_runbooks" in names:
            reply = self.consult_runbooks(messages)
        else:
            reply = self.summarize(messages)
        reply.usage_metadata = _estimate_usage(messages, tools, reply)
        return ChatResult(generations=[ChatGeneration(message=reply)])

    # Roles. Tests subclass the model and override one of these to script misbehaviour.

    def supervise(self, messages: Sequence[BaseMessage]) -> AIMessage:
        reported = {f["specialist"] for f in read_tag(messages, "findings") or []}
        for specialist in SPECIALISTS:
            if specialist not in reported:
                return self.tool_call(
                    "route",
                    {
                        "next": specialist,
                        "focus": DEFAULT_FOCUS[specialist],
                        "reason": f"no findings from {specialist} yet",
                    },
                )
        return self.tool_call(
            "route", {"next": "propose", "focus": "", "reason": "all specialists reported"}
        )

    def investigate_logs(self, messages: Sequence[BaseMessage]) -> AIMessage:
        fired_at = self._fired_at(messages)
        done = _tool_results(messages)
        if not done:
            return self.tool_call(
                "search_logs",
                {
                    "start": _iso(fired_at - timedelta(hours=1)),
                    "end": _iso(fired_at + timedelta(minutes=5)),
                    "min_level": "WARN",
                },
            )
        if len(done) == 1:
            return self.tool_call(
                "search_logs",
                {
                    "start": _iso(fired_at - timedelta(hours=3)),
                    "end": _iso(fired_at + timedelta(minutes=5)),
                    "min_level": "INFO",
                    "pattern": CHANGE_PATTERN,
                },
            )
        errors, changes = done[0][1], done[1][1]
        lines = [
            f"{g['count']}x {g['level']} {g['example']!r} ({g['first_seen']} to {g['last_seen']})"
            for g in errors["groups"][:4]
        ] or ["no warnings or errors in the hour before the alert"]
        lines += [f"change: {g['example']!r} at {g['first_seen']}" for g in changes["groups"]] or [
            "no deploys, config changes or restarts in the three hours before the alert"
        ]
        return AIMessage("\n".join(lines))

    def investigate_metrics(self, messages: Sequence[BaseMessage]) -> AIMessage:
        fired_at = self._fired_at(messages)
        done = _tool_results(messages)
        if not done:
            return self.tool_call(
                "detect_metric_anomalies",
                {
                    "start": _iso(fired_at - timedelta(hours=3)),
                    "end": _iso(fired_at + timedelta(minutes=5)),
                },
            )
        result = done[0][1]
        lines = [
            f"{a['metric']} anomalous: baseline {a['baseline']}, peak {a['peak']} "
            f"({a['ratio']}x), onset {a['onset']}"
            for a in result["anomalies"]
        ] or ["no anomalous metrics"]
        lines.append("within baseline: " + ", ".join(n["metric"] for n in result["normal"]))
        return AIMessage("\n".join(lines))

    def consult_runbooks(self, messages: Sequence[BaseMessage]) -> AIMessage:
        alert = read_tag(messages, "alert")
        done = _tool_results(messages)
        if not done:
            return self.tool_call(
                "search_runbooks",
                {"query": f"{alert['title']} {alert['description']}", "service": alert["service"]},
            )
        results = done[0][1]["results"]
        if not results:
            return AIMessage("no runbook section matched")
        top = results[0]
        return AIMessage(
            f"best match: {top['file']} > {top['heading']} (score {top['score']}), "
            f"suggested action: {top.get('suggested_action', 'not shown')}"
        )

    def summarize(self, messages: Sequence[BaseMessage]) -> AIMessage:
        return AIMessage(f"{len(_tool_results(messages))} tool calls made; see tool output.")

    def propose(self, messages: Sequence[BaseMessage]) -> AIMessage:
        alert = read_tag(messages, "alert")
        findings = read_tag(messages, "findings") or []
        outputs: dict[str, list[Any]] = {}
        for finding in findings:
            for call in finding["tool_calls"]:
                try:
                    outputs.setdefault(call["name"], []).append(json.loads(call["output"]))
                except json.JSONDecodeError:
                    continue

        anomalous = {
            a["metric"]
            for out in outputs.get("detect_metric_anomalies", [])
            for a in out["anomalies"]
        }
        log_lines = [g["example"] for out in outputs.get("search_logs", []) for g in out["groups"]]
        deploys = [line for line in log_lines if DEPLOY.search(line)]
        ooms = [line for line in log_lines if "OutOfMemoryError" in line]
        runbook_hits = [r for out in outputs.get("search_runbooks", []) for r in out["results"]]
        suggested = runbook_hits[0].get("suggested_action") if runbook_hits else None

        candidates: dict[Action, str] = {}
        if deploys and "error_rate_pct" in anomalous:
            candidates[Action.ROLLBACK_DEPLOY] = (
                f"The error rate is anomalous and a deploy happened in the window: {deploys[-1]!r}."
            )
        if {"cpu_pct", "rps"} <= anomalous and "memory_pct" not in anomalous:
            candidates[Action.SCALE_UP] = (
                "Request rate and CPU rose together with no memory growth: the service is short "
                "of capacity."
            )
        if ooms or "memory_pct" in anomalous:
            evidence = f"{ooms[0]!r} in the logs" if ooms else "sustained memory growth"
            candidates[Action.RESTART_SERVICE] = f"Memory exhaustion: {evidence}."

        if len(candidates) == 1:
            ((action, diagnosis),) = candidates.items()
            rationale = "Exactly one rule matched the evidence."
        else:
            action = Action.PAGE_HUMAN
            diagnosis = (
                "The evidence does not isolate a single cause that an allowed action fixes. "
                f"Anomalous metrics: {sorted(anomalous) or 'none'}. "
                f"Rules matched: {sorted(candidates) or 'none'}."
            )
            rationale = "Ambiguous evidence goes to a human."
        if suggested:
            agreement = "agrees" if suggested == action else f"suggests {suggested} instead"
            rationale += f" The top runbook section {agreement}."

        return self.tool_call(
            "propose_action",
            {
                "diagnosis": diagnosis,
                "action": str(action),
                "target": alert["service"],
                "rationale": rationale,
            },
        )

    # Helpers

    def tool_call(self, name: str, args: dict[str, Any]) -> AIMessage:
        digest = hashlib.sha256(json.dumps(args, sort_keys=True).encode()).hexdigest()[:10]
        call_id = f"call_{name}_{digest}"
        return AIMessage(
            content="",
            tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
        )

    @staticmethod
    def _fired_at(messages: Sequence[BaseMessage]) -> datetime:
        return datetime.fromisoformat(read_tag(messages, "alert")["fired_at"])


def _estimate_usage(
    messages: Sequence[BaseMessage], tools: list[dict[str, Any]], reply: AIMessage
) -> UsageMetadata:
    """Characters / 4. These are not real token counts; reports label them as estimates."""
    prompt_chars = sum(len(str(m.content)) for m in messages) + len(json.dumps(tools))
    reply_chars = len(str(reply.content)) + len(json.dumps(reply.tool_calls))
    input_tokens, output_tokens = prompt_chars // 4, reply_chars // 4
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
