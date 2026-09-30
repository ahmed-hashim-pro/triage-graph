# Decisions

A running log of what this project was built against and why it is shaped the
way it is. Entries are added as the work happens, not reconstructed afterwards.

## Versions

Pinned exactly in `pyproject.toml`; the full transitive set is in `uv.lock`.

| Package | Version | Why it is listed |
|---|---|---|
| langgraph | 1.2.12 | graph runtime |
| langgraph-prebuilt | 1.1.0 | `ToolNode`, `tools_condition`, `InjectedState` live here, not in `langgraph` |
| langgraph-checkpoint | 4.2.0 | checkpointer base classes and serializer |
| langgraph-checkpoint-sqlite | 3.1.1 | `SqliteSaver` |
| langchain-core | 1.6.6 | `BaseChatModel`, messages, `@tool` |
| langchain-anthropic | 1.7.5 | optional real provider (`[anthropic]` extra) |
| crewai | 1.15.23 | comparison port only (`[crewai]` extra); declares `requires-python <3.14` |
| pydantic | 2.12.5 | boundary validation |

Supported Python: 3.11–3.13 (CI matrix). Local development used 3.11.14.

## Documentation read (2026-09-30)

LangGraph (docs.langchain.com, which documents langgraph 1.x):

- Graph API (StateGraph, reducers, conditional edges, `Command`, recursion limit):
  https://docs.langchain.com/oss/python/langgraph/graph-api
- Interrupts (`interrupt()`, `Command(resume=...)`, re-execution rules):
  https://docs.langchain.com/oss/python/langgraph/interrupts
- Persistence: https://docs.langchain.com/oss/python/langgraph/persistence
- Checkpointers (`SqliteSaver`, `StateSnapshot`, durability modes):
  https://docs.langchain.com/oss/python/langgraph/checkpointers
- Streaming (`stream_mode`, `version="v2"`, `get_stream_writer`):
  https://docs.langchain.com/oss/python/langgraph/streaming
- Event streaming (`stream_events(version="v3")`):
  https://docs.langchain.com/oss/python/langgraph/event-streaming
- Subgraphs: https://docs.langchain.com/oss/python/langgraph/use-subgraphs
- ToolNode usage (quickstart and the interrupts page's tool-approval example):
  https://docs.langchain.com/oss/python/langgraph/quickstart

CrewAI (docs.crewai.com, which documents 1.x):

- Agents: https://docs.crewai.com/en/concepts/agents
- Tasks: https://docs.crewai.com/en/concepts/tasks
- Crews: https://docs.crewai.com/en/concepts/crews
- Processes: https://docs.crewai.com/en/concepts/processes
- Tools: https://docs.crewai.com/en/concepts/tools
- Custom LLM: https://docs.crewai.com/en/learn/custom-llm
- Checkpointing: https://docs.crewai.com/en/concepts/checkpointing
- Human-in-the-loop: https://docs.crewai.com/en/learn/human-in-the-loop
- Human feedback in flows: https://docs.crewai.com/en/learn/human-feedback-in-flows

Where the docs and the installed source disagreed, the source was treated as
the authority and the difference is recorded below.

## Verified by running it, before designing on it

A throwaway probe (not committed) was run against the pinned versions to check
the load-bearing claims:

- `StateGraph(State, input_schema=Input)` drops input keys that are not in the
  input schema. A forged `decision` key passed to `stream()` never reached state.
  This is one of the approval-boundary guarantees, so it has its own test.
- `SqliteSaver(sqlite3.connect(path, check_same_thread=False))` + `interrupt()`:
  a second OS process built a fresh graph on the same file and resumed with
  `Command(resume=...)` to completion.
- On resume, the interrupted node re-runs from its first line. Anything computed
  before `interrupt()` is recomputed, so nothing non-deterministic (ids,
  timestamps) may be created in the approval node before the interrupt.
- `stream(..., stream_mode=["updates", "custom"], version="v2")` yields dicts
  `{"type", "ns", "data"}`; an interrupt arrives as an `updates` part whose data
  is `{"__interrupt__": (Interrupt(...),)}`.
- `ToolNode` + `InjectedState` works inside a subgraph that is invoked from a
  node of a checkpointed parent; the injected argument is hidden from the tool
  schema the model sees.
- A `BaseChatModel` subclass that implements `bind_tools` by binding OpenAI-format
  tool schemas receives them as `tools=` in `_generate`, which is what the fake
  model relies on.

## Where the docs and the source disagreed

1. **`stream_events(version="v3")` is presented as the recommended API but is
   experimental.** The interrupts and event-streaming pages use
   `graph.stream_events(..., version="v3")` (with `stream.interrupted`,
   `stream.interrupts`, `stream.output`) as the primary pattern. The docstring of
   `Pregel.stream_events` in langgraph 1.2.12 says: "The `version="v3"` API is
   experimental and may change." The default for `stream_events` in source is
   `"v2"`, not `"v3"`. **Decision:** the CLI streams with
   `stream(stream_mode=[...], version="v2")` and reads interrupts from
   `get_state()`, and does not use v3.
2. **`stream(version=...)` is not in the `Pregel.stream` docstring.** It exists in
   the signature (`Literal["v1", "v2"]`, default `"v1"`) and is documented only in
   the streaming guide. It is not marked experimental. Its output shape was
   confirmed by the probe above rather than taken from the docs.
3. **The persistence page does not document `SqliteSaver`.** It only names it; the
   import path, `check_same_thread=False`, and `setup()` are on the checkpointers
   page.
4. **The API reference URL `https://reference.langchain.com/python/langgraph/prebuilt/`
   returned 404.** `ToolNode`'s constructor was read from the installed source
   instead: `ToolNode(tools, *, name="tools", tags=None, handle_tool_errors=...,
   messages_key="messages", wrap_tool_call=None, awrap_tool_call=None)`.
5. **CrewAI `BaseLLM.call` has more parameters than the custom-LLM page shows.**
   The page documents `call(messages, tools, callbacks, available_functions)`.
   In 1.15.23 the signature is `call(messages, tools=None, callbacks=None,
   available_functions=None, from_task=None, from_agent=None,
   response_model=None)`. The only abstract method is `call`.

## Design decisions

### State is plain JSON-shaped dicts; pydantic validates at the edges

Graph state is a `TypedDict` whose values are dicts, lists and strings. Pydantic
models (`Alert`, `Proposal`, `HumanDecision`, ...) validate data when it enters a
node and are dumped back to dicts when it leaves. This keeps checkpoints
readable and avoids depending on how the checkpoint serializer handles custom
classes across versions.

### The approval boundary is enforced in code, in layers

- The graph's `input_schema` only accepts the alert and the step limit, so a
  caller cannot pre-seed an approval.
- The proposer validates the model's action against the allow-list. An action
  outside it is not passed on. The proposal becomes `page_human` and the refused
  action is recorded, so a human sees what the model tried.
- The approval node fails closed. A malformed resume payload, or an edit to an
  action outside the allow-list, is treated as a rejection.
- The executor re-checks everything itself: an approval exists, it approves or
  edits *this* proposal (by id), and the final action is on the allow-list.
- The dry-run infrastructure adapter only has methods for the allowed actions.

The allow-list lives in one module (`actions.py`). Prompts mention it, but they
are not what enforces it.

### Expected answers are not visible to the model

Each scenario's expected action lives in `fixtures/expected.json`, which the
graph never reads. Alerts contain only what a real alert would.

### Streaming uses `stream(..., version="v2")`

See discrepancy 1. The v2 dict format (`type`, `ns`, `data`) is uniform
across modes and subgraphs, which keeps the CLI renderer small.

### Checkpoint writes use `durability="sync"`

The default is `"async"`. The crash test kills the process as soon as it reports
that it is waiting for approval, so the checkpoint must already be on disk by
then. `"sync"` makes that true by construction instead of relying on timing.

### CrewAI is an optional extra

`crewai==1.15.23` pulls in chromadb, the OpenAI SDK and many more packages. It is
only needed for `crewai_port/`, so the core package and its tests do not
depend on it.
