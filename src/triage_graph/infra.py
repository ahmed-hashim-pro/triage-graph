"""A fake infrastructure adapter. It never changes anything; it records what it would do."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from triage_graph.actions import Action, require_allowed
from triage_graph.schemas import ExecutionRecord


class DryRunInfra:
    """Dry-run adapter with one method per allowed action and an append-only ledger.

    With `ledger_path`, records go to a JSON-lines file so another process (or a
    test) can see exactly what was "executed". Executing the same proposal twice
    returns the first record instead of acting again.
    """

    def __init__(self, ledger_path: Path | None = None) -> None:
        self.ledger_path = ledger_path
        self._records: list[ExecutionRecord] = []
        self._handlers: dict[Action, Callable[[str], str]] = {
            Action.RESTART_SERVICE: self.restart_service,
            Action.SCALE_UP: self.scale_up,
            Action.ROLLBACK_DEPLOY: self.rollback_deploy,
            Action.PAGE_HUMAN: self.page_human,
        }

    def restart_service(self, target: str) -> str:
        return f"would perform a rolling restart of {target}"

    def scale_up(self, target: str) -> str:
        return f"would raise the replica ceiling of {target} by 50%"

    def rollback_deploy(self, target: str) -> str:
        return f"would roll {target} back to its previous release"

    def page_human(self, target: str) -> str:
        return f"would page the {target} on-call with this report"

    def execute(self, action: Action | str, target: str, proposal_id: str) -> ExecutionRecord:
        allowed = require_allowed(action)
        for record in self.history():
            if record.proposal_id == proposal_id:
                return record
        record = ExecutionRecord(
            proposal_id=proposal_id,
            action=allowed,
            target=target,
            dry_run=True,
            detail=self._handlers[allowed](target),
            recorded_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        )
        self._records.append(record)
        if self.ledger_path is not None:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with self.ledger_path.open("a") as ledger:
                ledger.write(record.model_dump_json() + "\n")
        return record

    def history(self) -> list[ExecutionRecord]:
        if self.ledger_path is None:
            return list(self._records)
        if not self.ledger_path.exists():
            return []
        return [
            ExecutionRecord.model_validate(json.loads(line))
            for line in self.ledger_path.read_text().splitlines()
            if line.strip()
        ]
