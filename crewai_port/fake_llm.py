"""A deterministic stand-in for an LLM that speaks CrewAI's custom-LLM protocol.

With native function calling, CrewAI calls `call(messages, tools=...)` and expects
either a list of OpenAI-style tool calls (which the executor runs) or a string
(the final answer). The fake picks its behaviour from `from_agent.role`.

The specialists reuse the LangGraph fake model's behaviour through a message
adapter. The supervisor and proposer can't: CrewAI owns the delegation protocol
and writes the prompts, so they get their own rules here.
"""

from __future__ import annotations

import json
import re
from typing import Any

from crewai import BaseLLM
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from crewai_port.roles import LOGS, METRICS, PROPOSER, RUNBOOKS, SPECIALISTS, SUPERVISOR
from triage_graph.fake_model import DEFAULT_FOCUS, FakeTriageModel, decide, signals_from_text

DELEGATE = "delegate_work_to_coworker"
CONTEXT_MARKER = "This is the context you're working with:"
_ALERT = re.compile(r"<alert>(.*?)</alert>", re.DOTALL)


def to_langchain(messages: list[dict[str, Any]]) -> list[BaseMessage]:
    converted: list[BaseMessage] = []
    for m in messages:
        content = m.get("content") or ""
        if m["role"] == "system":
            converted.append(SystemMessage(content))
        elif m["role"] == "user":
            converted.append(HumanMessage(content))
        elif m["role"] == "assistant":
            calls = [
                {
                    "name": c["function"]["name"],
                    "args": json.loads(c["function"]["arguments"] or "{}"),
                    "id": c["id"],
                    "type": "tool_call",
                }
                for c in m.get("tool_calls") or []
            ]
            converted.append(AIMessage(content, tool_calls=calls))
        elif m["role"] == "tool":
            converted.append(
                ToolMessage(content, tool_call_id=m["tool_call_id"], name=m.get("name"))
            )
    return converted


def to_crewai(reply: AIMessage) -> str | list[dict[str, Any]]:
    if not reply.tool_calls:
        return str(reply.content)
    return [
        {
            "id": c["id"],
            "type": "function",
            "function": {"name": c["name"], "arguments": json.dumps(c["args"])},
        }
        for c in reply.tool_calls
    ]


def tool_call(call_id: str, name: str, args: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }
    ]


def find_alert(messages: list[dict[str, Any]]) -> str:
    for m in messages:
        match = _ALERT.search(m.get("content") or "")
        if match:
            return match.group(1)
    raise ValueError("no <alert> tag in the conversation")


def delegation_results(messages: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """(coworker, answer) for each delegation the supervisor has made, in order."""
    coworker_by_call: dict[str, str] = {}
    for m in messages:
        for c in m.get("tool_calls") or []:
            if c["function"]["name"] == DELEGATE:
                coworker_by_call[c["id"]] = json.loads(c["function"]["arguments"])["coworker"]
    return [
        (coworker_by_call.get(m["tool_call_id"], "?"), m.get("content") or "")
        for m in messages
        if m["role"] == "tool" and m.get("name") == DELEGATE
    ]


class FakeCrewLLM(BaseLLM):
    def __init__(self, **data: Any) -> None:
        # BaseLLM validates the raw input for `model`, so a field default isn't enough.
        super().__init__(**{"model": "triage-fake", **data})

    def supports_function_calling(self) -> bool:
        return True

    def supports_stop_words(self) -> bool:
        return False

    def call(
        self,
        messages: str | list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any = None,
        from_agent: Any = None,
        response_model: Any = None,
    ) -> str | list[dict[str, Any]]:
        msgs = [{"role": "user", "content": messages}] if isinstance(messages, str) else messages
        tool_names = {t["function"]["name"] for t in tools or []}
        role = getattr(from_agent, "role", None)
        if role == SUPERVISOR and DELEGATE in tool_names:
            reply: str | list[dict[str, Any]] = self.supervise(msgs)
        elif role == PROPOSER:
            reply = self.propose(msgs)
        elif role in SPECIALISTS and tool_names:
            reply = self.investigate(role, msgs)
        else:
            # CrewAI's forced final answer after max_iter arrives with no agent and no tools.
            reply = self.final_answer(msgs)
        self._record_usage(msgs, reply)
        return reply

    def supervise(self, messages: list[dict[str, Any]]) -> str | list[dict[str, Any]]:
        done = [
            coworker for coworker, answer in delegation_results(messages) if "blocked" not in answer
        ]
        for specialist in (LOGS, METRICS, RUNBOOKS):
            if specialist not in done:
                return self.delegate(messages, specialist)
        return self.final_answer(messages)

    def delegate(self, messages: list[dict[str, Any]], coworker: str) -> list[dict[str, Any]]:
        focus = DEFAULT_FOCUS[SPECIALISTS[coworker]]
        return tool_call(
            f"call_delegate_{len(delegation_results(messages)) + 1}",
            DELEGATE,
            {
                "task": f"Investigate {focus}. <alert>{find_alert(messages)}</alert>",
                "context": "An alert fired; see the alert JSON in the task.",
                "coworker": coworker,
            },
        )

    def investigate(self, role: str, messages: list[dict[str, Any]]) -> str | list[dict[str, Any]]:
        model = FakeTriageModel()
        method = {
            LOGS: model.investigate_logs,
            METRICS: model.investigate_metrics,
            RUNBOOKS: model.consult_runbooks,
        }[role]
        return to_crewai(method(to_langchain(messages)))

    def propose(self, messages: list[dict[str, Any]]) -> str:
        task_text = "\n".join(m.get("content") or "" for m in messages if m["role"] == "user")
        alert = json.loads(find_alert(messages))
        evidence = task_text.split(CONTEXT_MARKER, 1)[-1]
        return json.dumps(decide(signals_from_text(evidence), alert["service"]))

    def final_answer(self, messages: list[dict[str, Any]]) -> str:
        results = delegation_results(messages)
        if not results:
            return "No evidence was gathered."
        return "\n\n".join(f"{coworker}:\n{answer}" for coworker, answer in results)

    def _record_usage(
        self, messages: list[dict[str, Any]], reply: str | list[dict[str, Any]]
    ) -> None:
        prompt = sum(len(str(m.get("content") or "")) for m in messages) // 4
        completion = len(json.dumps(reply) if isinstance(reply, list) else reply) // 4
        self._track_token_usage_internal(
            {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": prompt + completion,
            }
        )
