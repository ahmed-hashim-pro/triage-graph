import json
import os
import queue
import signal
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from conftest import alert_path

from triage_graph.fake_model import FakeTriageModel
from triage_graph.graph import build_graph
from triage_graph.infra import DryRunInfra
from triage_graph.runner import load_run, sqlite_checkpointer

CHILD = Path(__file__).with_name("crash_child.py")
ENV = {**os.environ, "PYTHONUNBUFFERED": "1"}
DEADLINE_SECONDS = 60

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="uses SIGKILL")


def first_line(proc: subprocess.Popen, timeout: float) -> str:
    lines: queue.Queue[str] = queue.Queue()
    threading.Thread(target=lambda: lines.put(proc.stdout.readline()), daemon=True).start()
    try:
        return lines.get(timeout=timeout)
    except queue.Empty:
        proc.kill()
        pytest.fail(f"child did not reach the approval gate within {timeout}s")


def test_run_killed_while_waiting_for_approval_resumes_in_a_new_process(tmp_path):
    db, ledger, thread = tmp_path / "checkpoints.db", tmp_path / "ledger.jsonl", "crash-1"

    child = subprocess.Popen(
        [
            sys.executable,
            str(CHILD),
            "run",
            str(db),
            str(ledger),
            str(alert_path("high-latency")),
            thread,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=ENV,
    )
    line = first_line(child, DEADLINE_SECONDS)
    assert line.startswith("WAITING-FOR-APPROVAL"), line + child.stderr.read()
    proposal_id = line.split()[1]

    child.send_signal(signal.SIGKILL)
    assert child.wait(timeout=10) == -signal.SIGKILL
    assert not ledger.exists()

    # A third party opening the checkpoint sees the run parked at the gate.
    with sqlite_checkpointer(db) as saver:
        graph = build_graph(FakeTriageModel(), checkpointer=saver, infra=DryRunInfra(ledger))
        parked = load_run(graph, thread)
    assert parked.status == "awaiting_approval"
    assert parked.pending["proposal"]["id"] == proposal_id

    resumed = subprocess.run(
        [sys.executable, str(CHILD), "resume", str(db), str(ledger), thread],
        capture_output=True,
        text=True,
        env=ENV,
        timeout=DEADLINE_SECONDS,
    )
    assert resumed.returncode == 0, resumed.stderr
    result = json.loads(resumed.stdout)
    assert result["status"] == "done"
    assert result["state"]["outcome"] == "executed"
    assert result["state"]["decision"]["approver"] == "test"
    assert "## Execution" in result["state"]["report"]

    [record] = DryRunInfra(ledger).history()
    assert (record.proposal_id, record.action, record.target) == (
        proposal_id,
        "scale_up",
        "checkout-api",
    )
    # Resuming made no model calls: the investigation was not re-run.
    assert result["state"]["usage"] == parked.state["usage"]
    assert result["state"]["findings"] == parked.state["findings"]
