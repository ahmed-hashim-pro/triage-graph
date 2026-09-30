"""Child process for test_crash_resume.py.

crash_child.py run DB LEDGER ALERT THREAD      run to the approval gate, then block
crash_child.py resume DB LEDGER THREAD         approve the pending proposal and finish
"""

import json
import sys
from pathlib import Path

from triage_graph.data import load_alert
from triage_graph.fake_model import FakeTriageModel
from triage_graph.graph import build_graph
from triage_graph.infra import DryRunInfra
from triage_graph.runner import load_run, resume, sqlite_checkpointer, start

WAITING = "WAITING-FOR-APPROVAL"


def main() -> None:
    mode, db, ledger = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    with sqlite_checkpointer(db) as saver:
        graph = build_graph(FakeTriageModel(), checkpointer=saver, infra=DryRunInfra(ledger))
        if mode == "run":
            alert, thread = sys.argv[4], sys.argv[5]
            result = start(graph, load_alert(alert).model_dump(mode="json"), thread_id=thread)
            print(WAITING, result.pending["proposal"]["id"], flush=True)
            sys.stdin.read()  # Block until the test kills this process.
        else:
            thread = sys.argv[4]
            proposal = load_run(graph, thread).pending["proposal"]
            decision = {"decision": "approve", "proposal_id": proposal["id"], "approver": "test"}
            result = resume(graph, thread, decision)
            print(json.dumps({"status": result.status, "state": result.state}), flush=True)


if __name__ == "__main__":
    main()
