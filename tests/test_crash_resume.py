import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time

import pytest
from conftest import alert_path

from triage_graph.fake_model import FakeTriageModel
from triage_graph.graph import build_graph
from triage_graph.runner import load_run, sqlite_checkpointer

CLI = [sys.executable, "-m", "triage_graph.cli"]
ENV = {**os.environ, "PYTHONUNBUFFERED": "1", "TRIAGE_PROVIDER": "fake"}
DEADLINE_SECONDS = 60

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="uses SIGKILL")


def read_until(proc: subprocess.Popen, marker: str, timeout: float) -> list[str]:
    """Lines printed by `proc` up to and including the first one starting with `marker`."""
    lines: queue.Queue[str] = queue.Queue()

    def pump() -> None:
        for line in proc.stdout:
            lines.put(line)
        lines.put("")

    threading.Thread(target=pump, daemon=True).start()
    seen: list[str] = []
    deadline = time.monotonic() + timeout
    while True:
        try:
            line = lines.get(timeout=max(deadline - time.monotonic(), 0))
        except queue.Empty:
            proc.kill()
            pytest.fail(f"no {marker!r} within {timeout}s; output so far:\n{''.join(seen)}")
        if not line:
            pytest.fail(f"process exited before {marker!r}:\n{''.join(seen)}{proc.stderr.read()}")
        seen.append(line)
        if line.startswith(marker):
            return seen


def test_run_killed_while_waiting_for_approval_resumes_in_a_new_process(tmp_path):
    state_dir, thread = tmp_path / "state", "crash-1"
    ledger = state_dir / "dry_run_ledger.jsonl"

    run = subprocess.Popen(
        [*CLI, "run", "--alert", str(alert_path("high-latency")), "--thread-id", thread,
         "--state-dir", str(state_dir), "--wait"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=ENV,
    )  # fmt: skip
    output = read_until(run, "Decision:", DEADLINE_SECONDS)
    assert any(line.startswith("=== Waiting for approval (thread crash-1)") for line in output)

    # The CLI is now blocked reading the decision from stdin.
    run.send_signal(signal.SIGKILL)
    assert run.wait(timeout=10) == -signal.SIGKILL
    assert not ledger.exists()

    with sqlite_checkpointer(state_dir / "checkpoints.db") as saver:
        parked = load_run(build_graph(FakeTriageModel(), checkpointer=saver), thread)
    assert parked.status == "awaiting_approval"
    proposal = parked.pending["proposal"]
    assert f"Proposal {proposal['id']}: scale_up on checkout-api\n" in output

    resumed = subprocess.run(
        [*CLI, "resume", thread, "--approve", "--approver", "test", "--state-dir", str(state_dir)],
        capture_output=True,
        text=True,
        env=ENV,
        timeout=DEADLINE_SECONDS,
    )
    assert resumed.returncode == 0, resumed.stderr
    assert "=== Outcome: executed" in resumed.stdout

    records = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert [(r["proposal_id"], r["action"], r["target"]) for r in records] == [
        (proposal["id"], "scale_up", "checkout-api")
    ]

    with sqlite_checkpointer(state_dir / "checkpoints.db") as saver:
        final = load_run(build_graph(FakeTriageModel(), checkpointer=saver), thread)
    assert final.state["outcome"] == "executed"
    assert final.state["decision"]["approver"] == "test"
    # Resuming made no model calls: the investigation was not re-run.
    assert final.state["usage"] == parked.state["usage"]
    assert final.state["findings"] == parked.state["findings"]
