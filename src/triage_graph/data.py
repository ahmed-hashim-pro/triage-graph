"""Alerts, runbooks, and the fixture datasets that stand in for log and metrics backends."""

from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from triage_graph.schemas import Alert

_REPO_ROOT = Path(__file__).resolve().parents[2]


def data_root() -> Path:
    """Directory holding `fixtures/` and `runbooks/`. Override with TRIAGE_DATA_DIR."""
    return Path(os.environ.get("TRIAGE_DATA_DIR", _REPO_ROOT))


@dataclass(frozen=True)
class LogRecord:
    ts: datetime
    level: str
    service: str
    source: str
    msg: str


@dataclass(frozen=True)
class MetricPoint:
    ts: datetime
    service: str
    metric: str
    value: float


@dataclass(frozen=True)
class Dataset:
    name: str
    logs: tuple[LogRecord, ...]
    metrics: tuple[MetricPoint, ...]


@dataclass(frozen=True)
class RunbookSection:
    file: str
    title: str
    heading: str
    text: str


def load_alert(path: str | Path) -> Alert:
    return Alert.model_validate_json(Path(path).read_text())


def load_dataset(name: str) -> Dataset:
    return _load_dataset(name, data_root())


@lru_cache(maxsize=16)
def _load_dataset(name: str, root: Path) -> Dataset:
    fixtures = root / "fixtures"
    logs = tuple(
        LogRecord(
            ts=datetime.fromisoformat(row["ts"]),
            level=row["level"],
            service=row["service"],
            source=row["source"],
            msg=row["msg"],
        )
        for row in map(json.loads, (fixtures / "logs" / f"{name}.jsonl").read_text().splitlines())
    )
    with (fixtures / "metrics" / f"{name}.csv").open(newline="") as f:
        metrics = tuple(
            MetricPoint(
                ts=datetime.fromisoformat(row["ts"]),
                service=row["service"],
                metric=row["metric"],
                value=float(row["value"]),
            )
            for row in csv.DictReader(f)
        )
    return Dataset(name=name, logs=logs, metrics=metrics)


def load_runbooks() -> tuple[RunbookSection, ...]:
    return _load_runbooks(data_root())


@lru_cache(maxsize=4)
def _load_runbooks(root: Path) -> tuple[RunbookSection, ...]:
    sections: list[RunbookSection] = []
    for path in sorted((root / "runbooks").glob("*.md")):
        title = path.stem
        heading: str | None = None
        body: list[str] = []
        for line in [*path.read_text().splitlines(), "## "]:
            if line.startswith("# "):
                title = line[2:].strip()
            elif line.startswith("## "):
                if heading is not None:
                    sections.append(
                        RunbookSection(path.name, title, heading, "\n".join(body).strip())
                    )
                heading, body = line[3:].strip(), []
            elif heading is not None:
                body.append(line)
    return tuple(sections)


SUGGESTED_ACTION = re.compile(r"Suggested action:\s*`([a-z_]+)`")
