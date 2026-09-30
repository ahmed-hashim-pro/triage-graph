"""What CrewAI 1.15.23 does when a run is paused or killed and resumed in a new process.

These pin the behaviour docs/LANGGRAPH_VS_CREWAI.md describes. If a CrewAI upgrade
changes it, these fail and the document needs updating.
"""

import importlib.util
import json
import os
import signal
import subprocess
import sys

import pytest

if importlib.util.find_spec("crewai") is None:
    pytest.skip("the crewai extra is not installed", allow_module_level=True)

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="uses SIGKILL")

ENV = {**os.environ, "CREWAI_DISABLE_TELEMETRY": "true", "OTEL_SDK_DISABLED": "true"}
TIMEOUT = 120


def experiment(name, *args, **env):
    return subprocess.run(
        [sys.executable, "-m", f"crewai_port.experiments.{name}", *map(str, args)],
        capture_output=True,
        text=True,
        env={**ENV, **env},
        timeout=TIMEOUT,
    )


def json_lines(stdout):
    return [json.loads(line) for line in stdout.splitlines() if line.startswith("{")]


def test_crew_checkpoint_skips_finished_tasks_but_loses_llms_and_conditions(tmp_path):
    died = experiment("crash_resume", "run", tmp_path, DIE_IN_PROPOSER="1")
    assert died.returncode == -signal.SIGKILL, died.stderr

    resumed = experiment("crash_resume", "resume", tmp_path, REATTACH_LLMS="1")
    assert resumed.returncode == 0, resumed.stderr
    _, restored, after = json_lines(resumed.stdout)
    restored = restored["restored"]

    # The finished investigation is kept and skipped...
    assert restored["task_outputs_present"] == [True, False]
    assert after["calls_after_resume"] == {"Remediation proposer": 1}
    assert json.loads(after["final"])["action"] == "scale_up"
    # ...custom tools come back as their own classes...
    assert restored["tool_classes"]["Log investigator"] == ["SearchLogs"]
    # ...but the custom LLM is rebuilt from its model name as an OpenAI client,
    assert set(restored["llm_classes"].values()) == {"OpenAICompletion"}
    # and the ConditionalTask returns as a plain Task with no condition.
    assert restored["task_types"] == ["Task", "Task"]
    assert restored["condition_present"] == [False, False]


def test_flow_pauses_for_feedback_and_resumes_in_a_new_process(tmp_path):
    db, ledger = tmp_path / "flows.db", tmp_path / "ledger.jsonl"

    started = experiment("flow_approval", "start", db, ledger)
    assert started.returncode == 0, started.stderr
    [paused] = json_lines(started.stdout)
    assert paused["paused"] is True
    assert paused["llm_calls"] == 12
    assert not ledger.exists()

    resumed = experiment("flow_approval", "resume", db, ledger, paused["flow_id"], "approve")
    assert resumed.returncode == 0, resumed.stderr
    [done] = json_lines(resumed.stdout)
    assert done == {"outcome": "executed scale_up", "llm_calls_in_resume": 0}
    assert [json.loads(line)["action"] for line in ledger.read_text().splitlines()] == ["scale_up"]


def test_flow_feedback_is_free_text_so_the_check_is_ours(tmp_path):
    db, ledger = tmp_path / "flows.db", tmp_path / "ledger.jsonl"
    [paused] = json_lines(experiment("flow_approval", "start", db, ledger).stdout)
    resumed = experiment("flow_approval", "resume", db, ledger, paused["flow_id"], "yes, go ahead")
    [done] = json_lines(resumed.stdout)
    assert done["outcome"] == "rejected"
    assert not ledger.exists()
