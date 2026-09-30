"""Start, resume and inspect triage runs. The CLI and the tests both go through here."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from triage_graph.graph import run_config
from triage_graph.state import DEFAULT_MAX_STEPS

StreamPart = dict[str, Any]


class NotWaitingError(RuntimeError):
    """Resume was requested for a thread that is not waiting for approval."""


@dataclass(frozen=True)
class RunResult:
    thread_id: str
    status: Literal["awaiting_approval", "done"]
    state: dict[str, Any]
    # The approval request payload when status is "awaiting_approval".
    pending: dict[str, Any] | None


@contextmanager
def sqlite_checkpointer(path: str | Path) -> Iterator[SqliteSaver]:
    db = Path(path)
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db, check_same_thread=False)
    try:
        saver = SqliteSaver(conn)
        saver.setup()
        yield saver
    finally:
        conn.close()


def _drive(
    graph: CompiledStateGraph,
    graph_input: Any,
    thread_id: str,
    max_steps: int,
    on_part: Callable[[StreamPart], None] | None,
) -> RunResult:
    config = run_config(thread_id, max_steps)
    # durability="sync": the checkpoint is on disk before stream() returns, so a
    # process killed while waiting for approval can always be resumed.
    for part in graph.stream(
        graph_input,
        config,
        stream_mode=["updates", "custom"],
        # Without subgraphs=True, custom events from inside the specialist
        # subgraphs (live tool calls) are silently dropped.
        subgraphs=True,
        version="v2",
        durability="sync",
    ):
        if on_part is not None:
            on_part(part)
    return load_run(graph, thread_id)


def load_run(graph: CompiledStateGraph, thread_id: str) -> RunResult:
    snapshot = graph.get_state(run_config(thread_id))
    if snapshot.interrupts:
        return RunResult(
            thread_id, "awaiting_approval", snapshot.values, snapshot.interrupts[0].value
        )
    return RunResult(thread_id, "done", snapshot.values, None)


def start(
    graph: CompiledStateGraph,
    alert: dict[str, Any],
    *,
    thread_id: str,
    max_steps: int = DEFAULT_MAX_STEPS,
    on_part: Callable[[StreamPart], None] | None = None,
) -> RunResult:
    return _drive(graph, {"alert": alert, "max_steps": max_steps}, thread_id, max_steps, on_part)


def resume(
    graph: CompiledStateGraph,
    thread_id: str,
    decision: Any,
    *,
    on_part: Callable[[StreamPart], None] | None = None,
) -> RunResult:
    if decision is None:
        # langgraph 1.2.12 crashes on Command(resume=None) (see DECISIONS.md).
        raise ValueError("a decision is required to resume")
    current = load_run(graph, thread_id)
    if current.status != "awaiting_approval":
        raise NotWaitingError(f"thread {thread_id!r} is not waiting for approval")
    max_steps = current.state.get("max_steps", DEFAULT_MAX_STEPS)
    return _drive(graph, Command(resume=decision), thread_id, max_steps, on_part)
