"""Graph wiring: supervisor loop over specialists, proposal, human approval, execution, report."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Checkpointer

from triage_graph.approval import approval, make_executor, route_after_approval
from triage_graph.infra import DryRunInfra
from triage_graph.llm import describe_model
from triage_graph.nodes import make_proposer, make_supervisor
from triage_graph.report import report
from triage_graph.schemas import SPECIALISTS, Alert
from triage_graph.specialists import make_specialist_node
from triage_graph.state import DEFAULT_MAX_STEPS, TriageInput, TriageState

MAX_STEPS_CEILING = 20


def make_intake(model_label: str) -> Callable[[TriageState], dict[str, Any]]:
    def intake(state: TriageState) -> dict[str, Any]:
        alert = Alert.model_validate(state["alert"])
        max_steps = state.get("max_steps", DEFAULT_MAX_STEPS)
        if not isinstance(max_steps, int) or not 1 <= max_steps <= MAX_STEPS_CEILING:
            raise ValueError(f"max_steps must be an integer from 1 to {MAX_STEPS_CEILING}")
        return {
            "alert": alert.model_dump(mode="json"),
            "max_steps": max_steps,
            "steps": 0,
            "model": model_label,
        }

    return intake


def route_after_supervisor(state: TriageState) -> str:
    target = state["next"]
    return {"propose": "proposer", "escalate": "report", "retry": "supervisor"}.get(target, target)


def build_graph(
    model: BaseChatModel,
    *,
    checkpointer: Checkpointer = None,
    infra: DryRunInfra | None = None,
    max_tool_rounds: int = 4,
) -> CompiledStateGraph:
    """Compile the triage graph.

    The approval step pauses with interrupt(), which needs a checkpointer; without
    one, an in-memory saver is used and nothing survives the process.
    """
    graph = StateGraph(TriageState, input_schema=TriageInput)
    graph.add_node("intake", make_intake(describe_model(model)))
    graph.add_node("supervisor", make_supervisor(model))
    for name in SPECIALISTS:
        graph.add_node(name, make_specialist_node(name, model, max_tool_rounds))
    graph.add_node("proposer", make_proposer(model))
    graph.add_node("approval", approval)
    graph.add_node("executor", make_executor(infra or DryRunInfra()))
    graph.add_node("report", report)

    graph.add_edge(START, "intake")
    graph.add_edge("intake", "supervisor")
    graph.add_conditional_edges(
        "supervisor", route_after_supervisor, [*SPECIALISTS, "supervisor", "proposer", "report"]
    )
    for name in SPECIALISTS:
        graph.add_edge(name, "supervisor")
    graph.add_edge("proposer", "approval")
    graph.add_conditional_edges("approval", route_after_approval, ["executor", "report"])
    graph.add_edge("executor", "report")
    graph.add_edge("report", END)
    return graph.compile(checkpointer=checkpointer if checkpointer is not None else InMemorySaver())


def run_config(thread_id: str, max_steps: int = DEFAULT_MAX_STEPS) -> RunnableConfig:
    """Config for one run. The recursion limit sits above the supervisor's own step
    limit so a stuck investigation ends as "escalated", not as GraphRecursionError."""
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 2 * max_steps + 10,
    }
