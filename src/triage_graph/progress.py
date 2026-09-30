"""Progress events for `stream_mode="custom"` consumers such as the CLI."""

from __future__ import annotations

from typing import Any

from langgraph.config import get_stream_writer


def emit(event: str, **data: Any) -> None:
    """Send a progress event. Does nothing when called outside a graph run."""
    try:
        writer = get_stream_writer()
    except RuntimeError:
        return
    writer({"event": event, **data})
