"""Kill a crew mid-run, then try to resume it from CrewAI's checkpoint in a new process.

    python -m crewai_port.experiments.crash_resume run DIR       # dies during the proposal
    python -m crewai_port.experiments.crash_resume resume DIR    # restores and continues

This mirrors tests/test_crash_resume.py for the LangGraph version. Findings are in
docs/LANGGRAPH_VS_CREWAI.md.
"""

from __future__ import annotations

import json
import os
import signal
import sys
from pathlib import Path
from typing import Any, ClassVar

from crewai import Crew
from crewai.state.checkpoint_config import CheckpointConfig

import crewai_port  # noqa: F401  (telemetry opt-outs)
from crewai_port.crew import build_crew
from crewai_port.fake_llm import FakeCrewLLM
from crewai_port.roles import PROPOSER
from triage_graph.data import data_root, load_alert


class CountingLLM(FakeCrewLLM):
    """Counts calls per agent role; with DIE_IN_PROPOSER=1 the process kills itself there."""

    calls: ClassVar[dict[str, int]] = {}

    def call(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
        role = getattr(kwargs.get("from_agent"), "role", None) or "forced-final"
        CountingLLM.calls[role] = CountingLLM.calls.get(role, 0) + 1
        if role == PROPOSER and os.environ.get("DIE_IN_PROPOSER") == "1":
            os.kill(os.getpid(), signal.SIGKILL)
        return super().call(messages, *args, **kwargs)


def main() -> None:
    mode, directory = sys.argv[1], Path(sys.argv[2])
    if mode == "run":
        alert = load_alert(data_root() / "fixtures" / "alerts" / "high-latency.json")
        checkpoint = CheckpointConfig(location=str(directory), on_events=["task_completed"])
        crew, _, _ = build_crew(alert, CountingLLM, checkpoint=checkpoint)
        crew.kickoff()
        print("finished without dying")
        return

    files = sorted(p for p in directory.rglob("*") if p.is_file())
    print(json.dumps({"checkpoint_files": [str(p.relative_to(directory)) for p in files]}))
    crew = Crew.from_checkpoint(CheckpointConfig(restore_from=str(files[-1])))
    restored = {
        "llm_classes": {a.role: type(a.llm).__name__ for a in crew.agents},
        "tool_classes": {a.role: [type(t).__name__ for t in a.tools or []] for a in crew.agents},
        "task_types": [type(t).__name__ for t in crew.tasks],
        "task_outputs_present": [t.output is not None for t in crew.tasks],
        "condition_present": [getattr(t, "condition", None) is not None for t in crew.tasks],
    }
    print(json.dumps({"restored": restored}))
    if os.environ.get("REATTACH_LLMS") == "1":
        # The checkpoint keeps only the model name, so custom LLMs must be put back.
        for agent in crew.agents:
            agent.llm = CountingLLM()
    output = crew.kickoff()
    print(json.dumps({"calls_after_resume": CountingLLM.calls, "final": output.raw[:300]}))


if __name__ == "__main__":
    main()
