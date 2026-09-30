"""Model selection. The fake model is the default so nothing needs an API key."""

from __future__ import annotations

import os

from langchain_core.language_models import BaseChatModel

from triage_graph.fake_model import FakeTriageModel

DEFAULT_ANTHROPIC_MODEL = "claude-opus-5-5"


def make_model(provider: str | None = None, model: str | None = None) -> BaseChatModel:
    """Build the chat model named by `provider`, or by TRIAGE_PROVIDER (default "fake").

    TRIAGE_MODEL overrides the model name for real providers.
    """
    provider = (provider or os.environ.get("TRIAGE_PROVIDER") or "fake").lower()
    if provider == "fake":
        return FakeTriageModel()
    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise RuntimeError(
                "TRIAGE_PROVIDER=anthropic needs the extra: uv sync --extra anthropic"
            ) from exc
        # No temperature: the current Claude models reject sampling parameters.
        return ChatAnthropic(  # type: ignore[call-arg]
            model=model or os.environ.get("TRIAGE_MODEL") or DEFAULT_ANTHROPIC_MODEL,
            max_tokens=16000,
        )
    raise ValueError(f"unknown TRIAGE_PROVIDER {provider!r}; expected 'fake' or 'anthropic'")
