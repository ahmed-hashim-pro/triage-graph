# LangGraph vs CrewAI: notes from building the same workflow twice

This is what I ran into while building triage-graph on **langgraph 1.2.12** and porting
its investigation and proposal steps to **crewai 1.15.23**. Every behaviour described
here was observed in this repo. Most of it is pinned by a test, and the source lines
are cited in [DECISIONS.md](DECISIONS.md).

**Scope, so this isn't read as more than it is:**

- The LangGraph version is the whole workflow: supervisor loop, specialists, proposer,
  approval pause, executor, report, CLI, evals. The CrewAI port covers the
  investigation and the proposal only (steps 1 to 6). Its approval and crash
  behaviour was tested with two small experiments (`crewai_port/experiments/`).
- Everything ran against deterministic fake models. I never ran either framework
  against a real LLM here, so nothing below says how CrewAI's generated prompts, or my
  own, behave with a real model.

## At a glance

| | LangGraph | CrewAI |
|---|---|---|
| Where control flow lives | In the graph: nodes, conditional edges | Mostly in the model: a delegating agent picks coworkers through tool calls |
| Supervisor | A node that calls the model with a `route` tool; code validates the choice | An agent with `allow_delegation=True`; CrewAI adds `delegate_work_to_coworker` and `ask_question_to_coworker` tools, and the coworker is chosen by role name |
| Step limit | A counter in state, checked by the supervisor node; ends with a clean `escalated` outcome | `max_iter` ends with a forced final answer and no signal; took a tool hook plus a `ConditionalTask` |
| What the proposer sees | Raw tool output from every specialist | The supervisor's prose; raw tool output stays inside each coworker's run |
| Prompts | All written in this repo | Assembled by CrewAI from role/goal/backstory and its own templates |
| Tool exceptions | `ToolNode` re-raises anything but argument errors unless told otherwise | Every exception becomes `"Error executing tool: …"` text for the model |
| Human approval | `interrupt()` inside any node; resume with any JSON value | Crews: a blocking `input()` review loop. Flows: `@human_feedback` with a pending state; resume with a string |
| Crash recovery | Checkpoint after every step; exact resume, tested with SIGKILL | Checkpoint per finished task. Restore skips finished tasks, but custom LLMs and task conditions come back wrong |
| Token accounting | `usage_metadata` on each message; summed in state | `CrewOutput.token_usage`. A custom LLM reports zero unless it tracks usage, and an LLM shared by agents is counted once per agent |
| Writes outside the repo | Nothing | A key file under the user data dir on import; a SQLite file there on every kickoff; telemetry on by default |

## What each made easy

**LangGraph**

- **The safety rules are ordinary code, in one place.** The approval gate is a node
  and the step limit is an `if` in another. That made them easy to test by calling
  the nodes directly with forged state (`tests/test_approval.py`), and easy to break
  on purpose to confirm the tests notice. Ten deliberate breakages, all caught.
- **A durable pause took very little code.** A checkpointer, `interrupt()` in the
  approval node, and `Command(resume=...)` from any process gave a pause that
  survives SIGKILL. The test kills the CLI while it waits on stdin and resumes from a
  new process.
- **`input_schema` removed a whole class of bugs.** Keys outside the input schema are
  dropped, so a forged `decision` in the input never reaches state.
- **The fake model needed no special path.** A `BaseChatModel` subclass with
  `bind_tools` receives the same messages and tool schemas a real model would, and
  its tool calls go through the same parsing and validation.

**CrewAI**

- **Defining agents and tools is short.** A role, a goal, a backstory, a list of
  pydantic tools. There are no nodes, edges or state schema to write for the
  investigation loop.
- **Delegation comes for free.** Setting `allow_delegation=True` gives an agent tools
  to hand work to every other agent in the crew, with no routing code.
- **Tool hooks are a good enforcement point.** A before-tool-call hook runs inline
  and can block the call. The port's step limit is built on it, and it also stops the
  supervisor from asking the proposer for an answer mid-investigation.
- **Flows have a real pause.** A provider that raises `HumanFeedbackPending` stops the
  flow, the state is persisted, and `Flow.from_pending(id).resume("approve")` in
  another process continues without re-running the paused step.

## What each hid

**LangGraph**

- **The default recursion limit.** The docs say 1000; the installed source and a
  runtime check say 10007. A looping supervisor would run about ten thousand steps
  before LangGraph stopped it, so the limit that matters is the one in the supervisor
  node. (DECISIONS #6.)
- **The interrupted node re-runs from the top on resume.** This is documented, but
  easy to miss, and it drives the design: nothing before `interrupt()` may have side
  effects or randomness. The proposal id is created in the node *before* the
  approval node for exactly this reason.
- **Custom stream events from a subgraph are dropped without warning** unless
  `stream(..., subgraphs=True)`. A probe showed this before the CLI was built; without
  the flag, the CLI could not show tool calls as they happen.
- **The recommended streaming API is experimental.** The interrupts and streaming
  pages lead with `stream_events(version="v3")`, whose docstring says "experimental and
  may change". I used `stream(version="v2")` instead. (DECISIONS #1.)

**CrewAI**

- **The prompts.** Each agent's messages are assembled from role, goal, backstory and
  CrewAI's own templates: "Current Task: …", "This is the expected criteria for your
  final answer: …", "you MUST return the actual complete content as the final answer,
  not a summary", and a coworker's "Your best answer to your coworker asking you
  this". I control fragments, not the prompt.
- **What happens at `max_iter`.** The executor appends a "force final answer" message,
  calls the model once more with no tools, and returns whatever comes back as a normal
  result (`handle_max_iterations_exceeded`). Nothing in the output says the limit was
  hit.
- **Several tool calls in one turn run in parallel on a thread pool.** The docstring
  says only the first runs; the code runs up to 8 at once, and drops all but the first
  only when one of the tools has `result_as_answer` or `max_usage_count` set. Anything
  a tool or hook shares has to be thread-safe; the port's step-limit hook takes a lock.
  (DECISIONS #8.)
- **The custom-LLM contract isn't the documented one.** The docs show `call()`
  executing tools itself. In 1.15.23 the executor passes `available_functions=None`,
  expects `call()` to return the tool calls, and runs them itself. (DECISIONS #8.)
- **`step_callback` doesn't fire for tool calls** in native function-calling mode,
  despite "called after each step of every agent" in the docs. (DECISIONS #9.)
- **"Sync" event handlers run on a thread pool**, so state kept in an event handler
  can lag behind the crew.
- **Token usage is multiplied by the number of agents sharing an LLM instance:** 60
  requests reported for 12 real calls.
- **Side effects outside the project.** Importing crewai created an encryption key
  under `~/Library/Application Support/crewai/credentials/`. Every kickoff writes
  `latest_kickoff_task_outputs.db` under the user data directory. Telemetry is on
  unless disabled by environment variable.

## Where I had to work around it

**LangGraph**

- **`Command(resume=None)` crashes with `UnboundLocalError`.** This is a known open
  issue, langchain-ai/langgraph#7034. The runner rejects `None` before calling
  LangGraph. (DECISIONS #7, `docs/upstream/resume-none.md`.)
- **The step limit and `recursion_limit`.** The step limit is enforced in the
  supervisor, and `run_config()` sets `recursion_limit = 2 * max_steps + 10` so the
  framework limit sits above it and never pre-empts the clean `escalated` outcome.
- **Tool error policy.** The tools raise `ToolInputError` for bad model input, and
  `ToolNode(handle_tool_errors=(ToolInputError,))` returns only those to the model.

**CrewAI**

- **The step limit.** `max_iter` couldn't express it and `step_callback` couldn't see
  it, so it's a before-tool-call hook. Hooks are registered globally, so this one is
  registered for the duration of one `kickoff()`, filtered to this crew and the
  supervisor's role, and removed in a `finally`. A `ConditionalTask` whose condition
  reads the hook's state skips the proposer after an escalation.
- **Getting the proposal out intact.** In `Process.hierarchical` every task goes
  through the manager, including the proposal, so the proposer's JSON would reach me
  only via the manager's retelling. I used a sequential process with a delegating
  supervisor instead, and parse the proposer's JSON with the same
  `validate_proposal` code the LangGraph version uses.
- **Token accounting.** Agents get one LLM instance each, built from a factory.
- **The fake LLM.** A `BaseLLM` subclass must pass `model` through `__init__`, because
  the validator rejects a field default. The specialists reuse the LangGraph fake's
  behaviour through a message adapter; the supervisor and proposer needed their own
  rules because CrewAI owns the delegation protocol.

## The human-approval pause

**LangGraph.** The approval node calls `interrupt(payload)`. The run stops, the
checkpoint is written, and `triage run` exits and prints the resume command. Any later
process builds the same graph on the same SQLite file and calls
`graph.stream(Command(resume=decision), config)`. The approval node runs again from the
top, `interrupt()` returns the decision, and the node validates it: it must be a
well-formed `HumanDecision` for *this* proposal id, or the run is refused. The resume
value can be any JSON, so the decision is structured, and checking it is ordinary
code. Tested: rejection never executes, an edit executes the edited action, malformed
or mismatched decisions fail closed, and a forged approval in the input is ignored.

**CrewAI.** Three mechanisms:

1. `Task(human_input=True)` calls `input()` inside the agent executor, after the agent's
   final answer. Whatever the human types goes back to the agent as feedback; an empty
   line accepts. It's a review loop for the output, not a gate. Nothing is persisted,
   and the process must stay alive and attached to a terminal.
2. Enterprise webhooks, documented for deployed crews. Not tried.
3. **Flows with `@human_feedback`.** A provider that raises `HumanFeedbackPending` pauses
   the flow and persists it (`SQLiteFlowPersistence` here). A different process calls
   `Flow.from_pending(flow_id, persistence).resume("approve")`, and the next step runs
   without re-running the paused one (0 LLM calls on resume). This works. But:
   - The pause is at a flow-method boundary, so a crew can't pause mid-run. The
     crew runs inside one flow step and the approval is the next step.
   - The resume value is a free-text string. With `emit=[...]`, CrewAI asks an LLM to
     map the human's text to an outcome. I didn't want a model interpreting an
     approval, so the experiment compares the text to `"approve"` exactly. A test shows
     that "yes, go ahead" is treated as a rejection.

## Resuming after a crash

**LangGraph.** `SqliteSaver` writes a checkpoint after every superstep (`durability="sync"`
here). The crash test SIGKILLs the CLI while it waits for approval, opens the database
from the test process to confirm the run is parked at the gate, then resumes in a new
process. Token usage and findings are identical before and after, so the
investigation was not re-run. A crash in the middle of a node re-runs that node.

**CrewAI.** A crew with `checkpoint=CheckpointConfig(on_events=["task_completed"])` writes
one JSON file per finished task. The experiment kills the process while the proposer
runs, then restores with `Crew.from_checkpoint` in a new process:

- The finished investigation was skipped. Only the proposer ran again, and it produced
  the same proposal. This matches the docs.
- The custom tools came back as their own classes, with their data.
- **The custom LLM did not come back.** Only `{"llm_type": "base", "model":
  "triage-fake"}` is stored. On restore, CrewAI looks the type up in a fixed registry of
  built-in LLM classes and falls back to a generic `LLM(model="triage-fake")`, which
  resolved to OpenAI. Without an OpenAI key the resumed run failed; with one, it would
  have sent the prompts to OpenAI. Re-attaching the LLMs by hand after restore fixed it.
- **The `ConditionalTask` came back as a plain `Task` without its condition.** After a
  restore, the rule "don't propose after an escalation" would be gone.
- The step-limit hook lives in the process, not in the checkpoint, so it would have
  to be registered again.
- A crash inside a flow step that runs a crew (the Flow approval setup) is not covered
  by flow persistence alone. I didn't test combining flow persistence with crew
  checkpoints.

## Which fits this workflow

For this workflow, LangGraph. Its main requirements are a hard approval gate, a
step limit with a clear outcome, and a pause that survives a crash, and in LangGraph
those are ordinary code at points I choose. In CrewAI each of them meant finding an
extension point (hooks, conditional tasks, a Flow wrapped around the crew). Its
checkpoint restore also silently changes the LLM and drops conditions, which is the
wrong failure mode for something guarding production actions.

CrewAI was quicker for describing agents and their tools, and delegation needed no
routing code. For a workflow where a model *should* decide who does what, and a
human pause is not a hard gate, that trade could go the other way. Neither framework
was tested with a real model here, so this says nothing about how their prompts or
agent loops perform with one.
