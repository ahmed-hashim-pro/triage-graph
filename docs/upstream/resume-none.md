# Upstream report: `Command(resume=None)` raises `UnboundLocalError`

Status as of 2026-09-30: **not filed by this project.**

> **Check before filing:** this bug is already reported as
> [langchain-ai/langgraph#7034](https://github.com/langchain-ai/langgraph/issues/7034)
> (open since 2026-03-05, labelled `bug`). Four fix PRs are linked from it
> (#7035, #7050, #7063, #7171) and none has been merged. The bug is still present in
> langgraph 1.2.12, the latest release on PyPI on 2026-09-30, and on `main`
> (`libs/langgraph/langgraph/pregel/_loop.py`, lines 910, 925 and 927).
>
> A new issue would be a duplicate. The more useful action is the short comment in
> section 1, which adds a newer version and a second trigger (a bare `Command()`).
> Section 2 is the full issue text, in case #7034 is closed without a fix.

---

## 1. Comment for #7034

Still reproduces on **langgraph 1.2.12** (latest on PyPI as of 2026-09-30) and on
current `main`. `_loop.py` still assigns `resume_is_map` only inside
`if (resume := ...) is not None:` (line 904) and reads it unconditionally at line 927.

A bare `Command()` hits the same path. It should raise
`EmptyInputError("Received empty Command input")`, but it raises `UnboundLocalError`:

```python
graph.invoke(Command(), config)
# UnboundLocalError: cannot access local variable 'resume_is_map' where it is not associated with a value
```

The thread is left paused, so nothing is lost. But callers can't tell this apart from a
real bug in their own code. Initialising `resume_is_map = False` before the `if`, as the
linked PRs do, makes both cases fall through to the existing `EmptyInputError`.

---

## 2. Full issue text

**Title:** `Command(resume=None)` and `Command()` raise `UnboundLocalError: resume_is_map` instead of `EmptyInputError`

### Checked other resources

- [x] This is a bug, not a usage question.
- [x] I added a clear and descriptive title that summarizes this issue.
- [x] I searched GitHub and found #7034, which reports the `resume=None` case; this
      adds the bare `Command()` case and confirms the bug on 1.2.12.
- [x] I am sure that this is a bug in LangGraph rather than my code.
- [x] The bug is not resolved by updating to the latest stable version of LangGraph.
- [x] I posted a self-contained, minimal, reproducible example.

### Example code

```python
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class State(TypedDict, total=False):
    answer: object


def ask(state: State) -> State:
    return {"answer": interrupt("approve?")}


builder = StateGraph(State)
builder.add_node("ask", ask)
builder.add_edge(START, "ask")
builder.add_edge("ask", END)
graph = builder.compile(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "1"}}

graph.invoke({}, config)  # pauses at interrupt()
graph.invoke(Command(resume=None), config)  # UnboundLocalError
```

Replacing the last line with `graph.invoke(Command(), config)` fails the same way.

### Error message and stack trace

```
Traceback (most recent call last):
  File "repro_resume_none.py", line 24, in <module>
    graph.invoke(Command(resume=None), config)  # UnboundLocalError
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File ".../site-packages/langgraph/pregel/main.py", line 3913, in invoke
    for chunk in self.stream(
  File ".../site-packages/langgraph/pregel/main.py", line 2899, in stream
    with SyncPregelLoop(
  File ".../site-packages/langgraph/pregel/_loop.py", line 1703, in __enter__
    self.updated_channels = self._first(
                            ^^^^^^^^^^^^
  File ".../site-packages/langgraph/pregel/_loop.py", line 927, in _first
    if not writes and not resume_is_map:
                          ^^^^^^^^^^^^^
UnboundLocalError: cannot access local variable 'resume_is_map' where it is not associated with a value
```

### Description

In `PregelLoop._first` (`langgraph/pregel/_loop.py`, line 848, shared by the sync and
async loops), `resume_is_map` is only assigned inside
`if (resume := cast(Command, self.input).resume) is not None:` (line 904). When the
input is a `Command` whose `resume` is `None` (explicitly, or by default in
`Command()`), that block is skipped. `map_command` yields no writes, and line 927
evaluates `not resume_is_map` on an unassigned local.

**Expected:** `EmptyInputError("Received empty Command input")`, which the code on line
928 is clearly meant to raise. Alternatively, if `None` should be a legal resume
value, the value is passed to `interrupt()`. Either way, the error should not be an
`UnboundLocalError`, because callers can't tell that apart from a bug in their own code.

**Actual:** `UnboundLocalError`. The thread stays paused (`get_state().interrupts` is
still set), so no state is lost.

**Suggested fix:** initialise `resume_is_map = False` before the `if` block.

Since `None` is also `Command.resume`'s default, it can't be used as a resume value.
It may be worth saying so on the interrupts page
(https://docs.langchain.com/oss/python/langgraph/interrupts).

### System info

```
OS: macOS 26.5.1 (Darwin 25.5.0, arm64)
Python: 3.11.14
langgraph: 1.2.12
langgraph-checkpoint: 4.2.0
langchain-core: 1.6.6
langgraph-sdk: 0.4.5
```

Also reproduced on Python 3.12.4 and 3.13.9 with the same package versions.
