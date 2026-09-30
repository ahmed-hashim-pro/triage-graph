"""Investigation tools as plain functions over fixture data.

Both the LangGraph graph and the CrewAI port wrap these same functions, so the
two implementations see identical evidence.
"""

from __future__ import annotations

import math
import re
import statistics
from collections import Counter
from datetime import UTC, datetime, timedelta
from typing import Any

from triage_graph.data import SUGGESTED_ACTION, Dataset, MetricPoint, RunbookSection

LEVELS = {"DEBUG": 10, "INFO": 20, "WARN": 30, "ERROR": 40}
METRICS = ("cpu_pct", "memory_pct", "latency_p99_ms", "error_rate_pct", "rps")

# A metric is anomalous when its peak is both this many times its baseline and
# at least this far from it in absolute terms. The absolute floor stops tiny
# baselines (an error rate of 0.2%) from turning noise into "3x anomalies".
MIN_RATIO = 1.5
MIN_DELTA = {
    "cpu_pct": 15.0,
    "memory_pct": 15.0,
    "latency_p99_ms": 150.0,
    "error_rate_pct": 1.0,
    "rps": 50.0,
}


class ToolInputError(ValueError):
    """Bad arguments from the model. Reported back to it instead of failing the run."""


def parse_time(value: str, name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ToolInputError(f"{name}={value!r} is not an ISO-8601 timestamp") from exc
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _iso(ts: datetime) -> str:
    return ts.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _window(start: str, end: str) -> tuple[datetime, datetime]:
    lo, hi = parse_time(start, "start"), parse_time(end, "end")
    if lo >= hi:
        raise ToolInputError(f"start ({start}) must be before end ({end})")
    return lo, hi


_TEMPLATE_SUBS = (
    (re.compile(r"\b[0-9a-f]{8}-[0-9a-f-]{27}\b"), "<id>"),
    (re.compile(r"\b[A-Z]+-\d+\b"), "<id>"),
    (re.compile(r"\d+(\.\d+)?"), "<n>"),
)


def _template(msg: str) -> str:
    for pattern, placeholder in _TEMPLATE_SUBS:
        msg = pattern.sub(placeholder, msg)
    return msg


def search_logs(
    dataset: Dataset,
    service: str,
    start: str,
    end: str,
    min_level: str = "WARN",
    pattern: str | None = None,
    max_groups: int = 15,
) -> dict[str, Any]:
    """Log lines for `service` in [start, end], grouped by message template."""
    lo, hi = _window(start, end)
    level = min_level.upper()
    if level not in LEVELS:
        raise ToolInputError(f"min_level must be one of {sorted(LEVELS)}, got {min_level!r}")
    matcher = None
    if pattern:
        try:
            matcher = re.compile(pattern, re.IGNORECASE)
        except re.error:
            matcher = re.compile(re.escape(pattern), re.IGNORECASE)

    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    matched = 0
    for record in dataset.logs:
        if record.service != service or not lo <= record.ts <= hi:
            continue
        if LEVELS[record.level] < LEVELS[level]:
            continue
        if matcher and not matcher.search(record.msg):
            continue
        matched += 1
        key = (record.level, record.source, _template(record.msg))
        group = groups.get(key)
        if group is None:
            groups[key] = {
                "level": record.level,
                "source": record.source,
                "template": key[2],
                "count": 1,
                "first_seen": _iso(record.ts),
                "last_seen": _iso(record.ts),
                "example": record.msg,
            }
        else:
            group["count"] += 1
            group["last_seen"] = _iso(record.ts)

    ordered = sorted(
        groups.values(), key=lambda g: (-LEVELS[g["level"]], -g["count"], g["first_seen"])
    )
    return {
        "service": service,
        "start": _iso(lo),
        "end": _iso(hi),
        "min_level": level,
        "pattern": pattern,
        "matched_lines": matched,
        "groups": ordered[:max_groups],
        "omitted_groups": max(0, len(ordered) - max_groups),
    }


def _series(
    dataset: Dataset, service: str, metric: str, lo: datetime, hi: datetime
) -> list[MetricPoint]:
    return [
        p
        for p in dataset.metrics
        if p.service == service and p.metric == metric and lo <= p.ts <= hi
    ]


def detect_metric_anomalies(
    dataset: Dataset,
    service: str,
    start: str,
    end: str,
    baseline_minutes: int = 60,
) -> dict[str, Any]:
    """Compare each metric in [start, end] against its median in the preceding baseline."""
    lo, hi = _window(start, end)
    if not 10 <= baseline_minutes <= 240:
        raise ToolInputError("baseline_minutes must be between 10 and 240")
    base_lo = lo - timedelta(minutes=baseline_minutes)

    anomalies: list[dict[str, Any]] = []
    normal: list[dict[str, Any]] = []
    for metric in METRICS:
        baseline_points = _series(dataset, service, metric, base_lo, lo - timedelta(seconds=1))
        window_points = _series(dataset, service, metric, lo, hi)
        if not baseline_points or not window_points:
            continue
        baseline = statistics.median(p.value for p in baseline_points)
        peak_point = max(window_points, key=lambda p: p.value)
        delta = peak_point.value - baseline
        ratio = peak_point.value / baseline if baseline > 0 else math.inf
        summary = {
            "metric": metric,
            "baseline": round(baseline, 2),
            "peak": round(peak_point.value, 2),
            "peak_at": _iso(peak_point.ts),
        }
        if ratio >= MIN_RATIO and delta >= MIN_DELTA[metric]:
            halfway = baseline + delta / 2
            onset = next(p.ts for p in window_points if p.value >= halfway)
            anomalies.append({**summary, "ratio": round(ratio, 2), "onset": _iso(onset)})
        else:
            normal.append(summary)

    if not anomalies and not normal:
        raise ToolInputError(f"no metrics for {service} in or before that window")
    return {
        "service": service,
        "start": _iso(lo),
        "end": _iso(hi),
        "baseline_start": _iso(base_lo),
        "anomalies": sorted(anomalies, key=lambda a: a["onset"]),
        "normal": normal,
    }


def get_metric_series(
    dataset: Dataset,
    service: str,
    metric: str,
    start: str,
    end: str,
    step_minutes: int = 5,
) -> dict[str, Any]:
    """One metric averaged into `step_minutes` buckets (at most 60 points)."""
    lo, hi = _window(start, end)
    if metric not in METRICS:
        raise ToolInputError(f"metric must be one of {list(METRICS)}, got {metric!r}")
    step = max(step_minutes, 1, math.ceil((hi - lo).total_seconds() / 60 / 60))
    buckets: dict[datetime, list[float]] = {}
    for point in _series(dataset, service, metric, lo, hi):
        offset = int((point.ts - lo).total_seconds() // 60) // step * step
        buckets.setdefault(lo + timedelta(minutes=offset), []).append(point.value)
    return {
        "service": service,
        "metric": metric,
        "step_minutes": step,
        "points": [
            [_iso(ts), round(statistics.fmean(vs), 2)] for ts, vs in sorted(buckets.items())
        ],
    }


_STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it of on or the to was were with "
    "this that when what which above below than".split()
)
_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_SUGGESTION_LINE = re.compile(r"^.*Suggested action:.*$\n?", re.MULTILINE)


def _stem(token: str) -> str:
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _tokens(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


GENERAL_RUNBOOK = "general.md"


def search_runbooks(
    sections: tuple[RunbookSection, ...],
    query: str,
    service: str | None = None,
    top_k: int = 3,
    include_suggestions: bool = True,
) -> dict[str, Any]:
    """Rank runbook sections against `query` with TF-IDF; headings count double.

    With `service`, only that service's runbook and the general policy are searched.
    With `include_suggestions=False`, the "Suggested action" lines are removed from
    the returned text and no `suggested_action` field is returned (used by the eval's
    ablation).
    """
    terms = set(_tokens(query))
    if not terms:
        raise ToolInputError("query has no searchable words")
    if service:
        sections = tuple(s for s in sections if s.title == service or s.file == GENERAL_RUNBOOK)
    docs = [Counter(_tokens(f"{s.title} {s.heading}") * 2 + _tokens(s.text)) for s in sections]
    n = len(docs)
    scored: list[tuple[float, RunbookSection]] = []
    for section, counts in zip(sections, docs, strict=True):
        score = 0.0
        for term in terms:
            tf = counts.get(term, 0)
            if tf:
                df = sum(1 for d in docs if term in d)
                score += (1 + math.log(tf)) * (math.log((n + 1) / (df + 1)) + 1)
        if score > 0:
            scored.append((score, section))
    scored.sort(key=lambda pair: (-pair[0], pair[1].file, pair[1].heading))

    results: list[dict[str, Any]] = []
    for score, section in scored[:top_k]:
        result: dict[str, Any] = {
            "file": section.file,
            "heading": section.heading,
            "score": round(score, 2),
        }
        if include_suggestions:
            suggested = SUGGESTED_ACTION.search(section.text)
            result["suggested_action"] = suggested.group(1) if suggested else None
            result["text"] = section.text
        else:
            result["text"] = _SUGGESTION_LINE.sub("", section.text).strip()
        results.append(result)
    return {"query": query, "service": service, "results": results}
