"""Specialists: each is a small compiled subgraph (model <-> ToolNode) with private messages."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from triage_graph.progress import emit
from triage_graph.prompts import SPECIALIST_SYSTEM, specialist_message
from triage_graph.schemas import Finding, Specialist, ToolCallRecord
from triage_graph.state import TriageState
from triage_graph.tools import ToolInputError

MAX_TOOL_OUTPUT_CHARS = 6000
OUT_OF_ROUNDS = "Tool budget used up. Reply now with your summary and do not call a tool."


class SpecialistState(MessagesState):
    alert: dict[str, Any]


def usage_of(messages: Sequence[BaseMessage]) -> dict[str, int]:
    usage = {"model_calls": 0, "input_tokens": 0, "output_tokens": 0}
    for message in messages:
        if isinstance(message, AIMessage):
            usage["model_calls"] += 1
            meta = message.usage_metadata or {}
            usage["input_tokens"] += meta.get("input_tokens", 0)
            usage["output_tokens"] += meta.get("output_tokens", 0)
    return usage


def build_specialist(
    name: Specialist, model: BaseChatModel, tools: list[BaseTool], max_tool_rounds: int
) -> CompiledStateGraph:
    with_tools = model.bind_tools(tools)

    def agent(state: SpecialistState) -> dict[str, Any]:
        messages = state["messages"]
        rounds = sum(1 for m in messages if isinstance(m, AIMessage) and m.tool_calls)
        if rounds < max_tool_rounds:
            reply = with_tools.invoke(messages)
            for call in reply.tool_calls:
                emit("tool_call", specialist=name, tool=call["name"], args=call["args"])
            return {"messages": [reply]}
        # Tools stay bound because Anthropic rejects histories containing tool_use blocks
        # without tool definitions; any tool call in this final reply is dropped.
        reply = with_tools.invoke([*messages, HumanMessage(OUT_OF_ROUNDS)])
        return {"messages": [AIMessage(reply.text, usage_metadata=reply.usage_metadata)]}

    graph = StateGraph(SpecialistState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tools, handle_tool_errors=(ToolInputError,)))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition, ["tools", END])
    graph.add_edge("tools", "agent")
    return graph.compile(name=name)


def finding_from(name: Specialist, focus: str, messages: Sequence[BaseMessage]) -> Finding:
    outputs = {m.tool_call_id: str(m.content) for m in messages if isinstance(m, ToolMessage)}
    calls = [
        ToolCallRecord(
            name=call["name"],
            args=call["args"],
            output=outputs.get(call["id"] or "", "")[:MAX_TOOL_OUTPUT_CHARS],
        )
        for m in messages
        if isinstance(m, AIMessage)
        for call in m.tool_calls
    ]
    final = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
    summary = final.text.strip() if final is not None else ""
    return Finding(
        specialist=name,
        focus=focus,
        summary=summary or "(no summary returned)",
        tool_calls=calls,
    )


def make_specialist_node(
    name: Specialist, model: BaseChatModel, tools: list[BaseTool], max_tool_rounds: int
) -> Callable[[TriageState], dict[str, Any]]:
    subgraph = build_specialist(name, model, tools, max_tool_rounds)

    def node(state: TriageState) -> dict[str, Any]:
        focus = state.get("focus", "")
        prompt = [
            SystemMessage(SPECIALIST_SYSTEM[name]),
            specialist_message(state["alert"], focus, state.get("findings", [])),
        ]
        result = subgraph.invoke({"messages": prompt, "alert": state["alert"]})
        finding = finding_from(name, focus, result["messages"])
        emit(
            "finding",
            specialist=name,
            tool_calls=[{"name": c.name, "args": c.args} for c in finding.tool_calls],
            summary=finding.summary,
        )
        return {
            "findings": [finding.model_dump(mode="json")],
            "usage": usage_of(result["messages"]),
        }

    return node
