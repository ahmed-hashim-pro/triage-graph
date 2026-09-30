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
6. **The default recursion limit is 10007, not the documented 1000.**
   - Docs: https://docs.langchain.com/oss/python/langgraph/graph-api, section
     "Recursion limit" (line 1032 of the page's `.md` export, fetched 2026-09-30):
     "Starting in version 1.0.6, the default recursion limit is set to 1000 steps."
   - Source: `langgraph/_internal/_config.py`, line 32, in the installed
     langgraph 1.2.12:
     `DEFAULT_RECURSION_LIMIT = int(getenv("LANGGRAPH_DEFAULT_RECURSION_LIMIT", "10007"))`.
     The same module applies it as the run default at line 335
     (`recursion_limit=DEFAULT_RECURSION_LIMIT`).
   - Runtime: a one-node graph that loops forever, invoked with no config and with
     `LANGGRAPH_DEFAULT_RECURSION_LIMIT` unset, raises `GraphRecursionError: Recursion
     limit of 10007 reached without hitting a stop condition.`

   Found when a test expecting `GraphRecursionError` at about 42 supersteps did not
   get one. **Consequence:** the framework limit is no practical guard against a
   looping supervisor. The step limit is enforced in the supervisor node itself, and
   `run_config()` sets `recursion_limit = 2 * max_steps + 10` as a backstop above it.
7. **`Command(resume=None)` crashes with `UnboundLocalError`.** In langgraph 1.2.12,
   `PregelLoop._first` (`langgraph/pregel/_loop.py`) assigns `resume_is_map` only
   inside `if (resume := ...) is not None:` (line 904). Line 927 reads it anyway, so
   `Command(resume=None)` and a bare `Command()` raise `UnboundLocalError` instead of the
   `EmptyInputError` on line 928. The thread stays paused, so nothing is lost.
   Reproduced with a standalone script on Python 3.11, 3.12 and 3.13. This is already
   reported upstream as
   [langchain-ai/langgraph#7034](https://github.com/langchain-ai/langgraph/issues/7034)
   (open, fix PRs unmerged). A write-up for the maintainers is in
   [`docs/upstream/resume-none.md`](upstream/resume-none.md). **Workaround:**
   `runner.resume()` rejects `None` before calling LangGraph. Every other malformed
   payload reaches the approval node, which fails closed.

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

### Specialists are subgraphs called from a node, not shared-state subgraphs

Each specialist is a compiled `StateGraph` (model node + `ToolNode`, routed by
`tools_condition`) with its own `messages` channel. The parent calls it from a
node function and keeps only a `Finding`: the final summary plus every tool call
and its output. Tool chatter stays out of the parent state and the checkpoint,
and the proposer sees raw tool output rather than only the specialists' summaries.

A specialist gets `max_tool_rounds` (default 4) tool-calling turns. After that
it is asked for its summary with the tools still bound (Anthropic rejects a
history that contains `tool_use` blocks when no tools are defined) and any tool
call in that reply is dropped.

### Tool errors are split into "the model's fault" and "a bug"

`ToolNode`'s default only reports argument-validation errors back to the model
and re-raises everything else. The tools raise `ToolInputError` for bad input
from the model (an unparseable timestamp, an unknown metric), and the node is
built with `handle_tool_errors=(ToolInputError,)`. The model sees those and can
retry. Any other exception still fails the run.

### The step limit counts every non-final supervisor turn

A turn that dispatches a specialist, or that fails to produce a valid `route`
call, uses one step. When `steps >= max_steps`, the supervisor stops without
calling the model. It sets `outcome="escalated"` with a reason that starts with
"insufficient evidence, escalating", and the graph goes straight to the report.
Counting invalid replies means a model that never calls `route` still ends.

### Proposals may only act on the alerting service

The proposer refuses an allowed action aimed at another service. For example, it
will not restart `inventory-api` because `orders-api` is alerting. The refused
proposal becomes `page_human` in the same way as an off-list action. This keeps
the blast radius to the service the alert is about.

### Forced tool choice is not used

The default real model (`claude-opus-5-5`) returns a 400 for `tool_choice` `any`
or `tool`. Tools are bound with the default (`auto`) and the prompts name the
tool to call. If a model answers in prose instead, the code already treats that
as a failed step (supervisor) or as `page_human` (proposer).

### The approval gate

- `approval` calls `interrupt()` with the proposal and the allow-list. Nothing
  before that call has side effects or randomness, because LangGraph re-runs the
  node from its first line on resume. The proposal id is created earlier, in the
  proposer, so it is fixed in the checkpoint before the pause.
- The resume value must validate as `HumanDecision` (`extra="forbid"`) and name
  the pending proposal's id. An approval is bound to the proposal the human
  actually saw. Anything else sets `outcome="refused"` and executes nothing.
- An edit replaces the action only; the target stays the alerting service. The
  edited action is checked against the allow-list before it is recorded.
- `executor` does not trust the routing. It re-validates the recorded decision,
  the proposal id and the allow-list, and checks that the target is the
  alerting service. If any check fails it raises instead of executing. Tests
  call it directly with forged states to show this.
- `DryRunInfra.execute` accepts only `Action` values and records at most once per
  proposal id. If a crash lands between writing the ledger and writing the
  checkpoint, the re-run executor returns the earlier record instead of acting
  twice.

### What the approval gate does not protect against

The gate assumes the checkpoint database is trusted. Anyone who can write to it
can call `update_state` or edit rows to forge a decision. Signing decisions
(for example, an HMAC over proposal id + decision) would close that gap. It is
out of scope here and is listed under Limitations in the README.

### Kill-and-resume test

`tests/test_crash_resume.py` starts `tests/crash_child.py` in a subprocess. The
child runs a scenario to the approval gate, prints a marker, and blocks on
stdin. The test then:

1. SIGKILLs the child and checks that nothing was executed.
2. Opens the SQLite file from the test process and checks that the thread is
   parked at the gate with the same proposal id.
3. Resumes the thread in a new subprocess, then checks that the approved action
   is in the ledger exactly once, and that token usage and findings are identical
   to the parked state. That shows the investigation was not re-run.

### Known bias in the evidence: runbooks name the answer

Every runbook section ends with `Suggested action: \`...\``, and the runbook
agent passes it on to the proposer. For each scenario, the matching section in
the alerting service's runbook suggests the expected action. A real model can
therefore score well by following the runbook rather than by diagnosing from
logs and metrics. Real runbooks often do say what to do, so this is not
unrealistic. But it means an eval score here says more about "finds and follows
the right runbook" than about independent diagnosis. The eval report will state
this. An ablation that strips the suggested-action lines would measure the
difference; it is not built yet.

### Streaming in the CLI

The runner streams `stream_mode=["updates", "custom"]` with `version="v2"` and
`subgraphs=True`. Nodes send progress events (`route`, `tool_call`, `finding`,
`proposal`, `decision`, `executed`) through `get_stream_writer()`, and the CLI prints
one line per event. Root-level `updates` fill in the nodes that send no events
(intake, report, and the interrupt).

`subgraphs=True` is required, not cosmetic. The specialist subgraphs are invoked
inside a node function, and with `subgraphs=False` their custom events are dropped
silently: no error, and no event reaches the parent stream. A probe against 1.2.12
showed the parent's own custom event arriving without the subgraph's. With
`subgraphs=True` the subgraph's event arrives with `ns=("<node>:<task id>",)`. The
streaming page does say `subgraphs=True` includes subgraph output; what it does not
say is that leaving it off loses custom events without any warning.

### CLI shape

- `triage run` exits at the approval gate and prints the exact `triage resume`
  commands. The pause is a durable state, not a blocked process, so a decision can
  come hours later from another shell.
- `triage run --wait` prompts on stdin instead. The crash test uses this mode: it
  SIGKILLs the real CLI while it waits for input, then resumes with
  `triage resume` in a new process.
- The CLI rejects an `--edit` to an action outside the allow-list before resuming,
  with exit code 2, so a typo doesn't permanently refuse the incident. The approval
  node still fails closed if a bad edit reaches it through the API.
- The graph records which model produced the findings (`state["model"]`). The report
  labels token counts as estimates for the fake model, and stays correct if the run
  is resumed under a different provider.

## Evals

### What is measured

`python -m evals.run` runs every scenario N times in two variants and records, per
run: the proposed action (or `escalated` / `error`), whether it matches
`fixtures/expected.json`, supervisor steps, model calls, and input/output tokens.
A run ends at the approval gate; nothing is executed. Escalation at the step
limit counts as a miss, even for the `page_human` scenario, because it is not a
proposal. A cell whose runs disagree is marked **flaky** in the report, with the
count of each answer. Runs are interleaved (run 1 of every cell, then run 2), so a
budget stop leaves the cells with similar numbers of runs.

### The runbook ablation

The `suggestions-hidden` variant builds the graph with `runbook_suggestions=False`.
The runbook tool then drops the `Suggested action:` lines and the
`suggested_action` field before any agent sees them, so the runbook agent's
summary and the raw evidence given to the proposer can't contain them. A test
checks that no finding or supervisor note in that variant mentions the expected
action. The section prose still describes the fix, so this is a partial ablation.
The report says so.

### Cost control for real-model runs

- Before running, the script prints the plan and an estimate: tokens per run are
  scaled up from a fake-model dry run, with an assumed number of output tokens per
  call (thinking included) that depends on effort. The assumptions are constants
  in `evals/harness.py`, and they are guesses.
- The number of runs is cut so the estimate fits the budget (default $2.00). If
  even one run per cell doesn't fit, the script refuses and says why.
- A paid run asks for confirmation unless `--yes` is given.
- `BudgetGuard`, a LangChain callback with `raise_error = True`, checks each model
  call before it is sent. If the call's worst case (prompt characters / 2 plus a
  tool-schema allowance as input, `max_tokens` as output) could take spend past
  the cap, it raises and the eval stops. Spend is summed from the real
  `usage_metadata` of each response at list price. A test shows the guard stopping
  a run before the cap, and another shows it also sees the calls inside the
  specialist subgraphs. Callbacks reach those calls through LangGraph's config
  propagation; nothing passes them explicitly.

At list prices (hard-coded from 2026-09-25, `PRICES` in `evals/harness.py`), the
planner fits the following under $2 for four scenarios × two variants:

| Model | Effort | Est. $/run | Runs per cell |
|---|---|---|---|
| claude-opus-5-5 | default (medium) | 0.29 | 0: refuses |
| claude-opus-5-5 | low | 0.17 | 1 |
| claude-sonnet-5-5 | low | 0.09 | 2 |
| claude-haiku-4-5 | none (no thinking by default) | 0.04 | 6 |

One run per cell cannot show flakiness; the report says that when N < 3.

### Credentials

- Only `ANTHROPIC_API_KEY`, read by `ChatAnthropic` from the environment, is used.
  Nothing in this repo reads, stores or prints it. `ChatAnthropic` passes an empty
  key when the variable is unset; whether the SDK then falls back to an
  `ant auth login` profile was not checked, so the eval treats the variable as the
  only credential.
- Eval runs use the in-memory checkpointer. The only files written are the
  report and a JSON of the records, both built field by field. A test sets a
  sentinel key in the environment, writes results for a `ChatAnthropic` plan, and
  checks that the sentinel is not in either file.

### The real-model test needs an explicit opt-in

`evals/test_evals.py::test_real_model_eval` is skipped unless both
`ANTHROPIC_API_KEY` and `TRIAGE_EVAL_REAL=1` are set. This is one step stricter
than "skipped when no key": a developer with the key exported for other work
should not spend money by running `pytest`.

### No real-model results are committed

No API key was available while this was built, so the real-model eval was built
and tested offline but never run. `evals/results/fake.md` is the fake-model report.
It is there to show the format, and its 100% is by construction.
