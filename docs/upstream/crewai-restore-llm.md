# Upstream report: checkpoint restore silently swaps a custom LLM for an OpenAI client

Status as of 2026-10-01: **not filed by this project.**

> **Search before filing.** crewAIInc/crewAI issues and PRs were searched on
> 2026-10-01 (`from_checkpoint llm`, `checkpoint custom LLM`, `llm_type base`,
> `_validate_llm_ref`, `Crew.from_checkpoint`, and others). No issue reports this.
> The behaviour comes from
> [PR #6117](https://github.com/crewAIInc/crewAI/pull/6117), "fix(checkpoint): rebuild
> custom BaseLLM as concrete LLM on restore" (merged 2026-06-11). That PR replaced a
> crash with a deliberate "best-effort" fallback: "A custom subclass can't be
> reconstructed exactly (its class/`__init__` aren't recorded), so this is best-effort
> — restore succeeds and a real model resumes."
>
> So the issue below does not claim an unnoticed bug. It argues that doing this
> silently is the wrong failure mode, and asks for it to be loud or configurable.

---

**Title:** Restoring a checkpoint silently replaces a custom `BaseLLM` with a default-provider client

### Description

When an agent uses a custom `BaseLLM` subclass, `Crew.from_checkpoint` restores the
agent with a generic `LLM(model=...)` instead. For a model name without a provider
prefix, that is an `OpenAICompletion`. Nothing is logged or warned. The next call on
the resumed crew sends the prompt to OpenAI, not to the backend the custom class
wrapped.

This comes from the fallback added in #6117. `_validate_llm_ref`
(`lib/crewai/src/crewai/agents/agent_builder/base_agent.py`) maps the serialized
`llm_type: "base"` to the abstract `BaseLLM` and, since it can't be instantiated,
builds a concrete `LLM` from the saved fields (lines 121-124 on `main` at
`1b9bbccbed`). The checkpoint JSON stores only
`{"llm_type": "base", "model": "...", ...}`, with nothing about the original class.

Replacing a crash with a working restore makes sense. But a silent change of provider
matters for anyone whose custom LLM exists to *avoid* a public provider: an in-house
endpoint, a gateway that redacts data, a model under a data-processing agreement.
With `OPENAI_API_KEY` set in the environment, which is common on developer machines,
the restored crew sends its prompts, including task context and tool output, to
OpenAI. The request would probably then fail on the unknown model name, but by then
the data has left.

### Steps to reproduce

```python
import glob
import os
import tempfile

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.pop("OPENAI_API_KEY", None)  # makes the swap visible without a network call

from crewai import Agent, BaseLLM, Crew, Task
from crewai.state.checkpoint_config import CheckpointConfig


class PrivateLLM(BaseLLM):
    """Stands in for any custom LLM, e.g. a wrapper around an in-house endpoint."""

    def call(
        self,
        messages,
        tools=None,
        callbacks=None,
        available_functions=None,
        from_task=None,
        from_agent=None,
        response_model=None,
    ):
        return "done"


location = tempfile.mkdtemp()
agent = Agent(role="writer", goal="write", backstory="b", llm=PrivateLLM(model="in-house-model"))
task = Task(description="Say done.", expected_output="done", agent=agent)
Crew(agents=[agent], tasks=[task], checkpoint=CheckpointConfig(location=location)).kickoff()

checkpoint = sorted(glob.glob(f"{location}/**/*.json", recursive=True))[-1]
restored = Crew.from_checkpoint(CheckpointConfig(restore_from=checkpoint))
llm = restored.agents[0].llm
print("restored LLM:", type(llm).__module__ + "." + type(llm).__name__, "model:", llm.model)

try:
    llm.call([{"role": "user", "content": "hello"}])
except Exception as exc:
    print("calling it:", type(exc).__name__, exc)
```

### Output

```
restored LLM: crewai.llms.providers.openai.completion.OpenAICompletion model: in-house-model
calling it: ValueError OPENAI_API_KEY is required
```

The only warnings printed were unrelated `DeprecationWarning`s (`function_calling_llm`,
`reasoning`). Nothing mentioned the LLM being replaced. With `OPENAI_API_KEY` set, the
call goes to the OpenAI API.

### Expected behaviour

Any of these would avoid a silent provider switch, roughly in order of preference:

1. Record the custom class's import path when serializing, and re-instantiate it on
   restore when it is importable (falling back as today only when it isn't).
2. Let the caller supply LLMs on restore, e.g.
   `Crew.from_checkpoint(config, llms={"writer": PrivateLLM(...)})`, and raise if a
   custom LLM would otherwise be replaced.
3. At minimum, emit a warning that names the agent, the lost class, and the class
   that replaced it. Also note on the checkpointing docs page that custom LLMs must
   be re-attached after `from_checkpoint`.

### Workaround

Re-assign `agent.llm` on every restored agent before calling `kickoff()`.

### Environment

- crewai 1.15.23 (latest on PyPI on 2026-10-01); same code on `main` at `1b9bbccbed`
- Python 3.11.14
- macOS 26.5.1 (arm64)

### Additional context

Found while porting a workflow to CrewAI and testing crash recovery; the test that
pins this behaviour is `tests/test_crewai_resume.py` in
https://github.com/ahmed-hashim-pro/triage-graph.
