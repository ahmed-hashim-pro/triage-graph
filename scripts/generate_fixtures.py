"""Regenerate fixtures/logs/*.jsonl and fixtures/metrics/*.csv.

    uv run python scripts/generate_fixtures.py

The output is deterministic (fixed seeds), so regenerating produces no diff
unless this script changed. Alerts, runbooks and expected answers are written
by hand and are not touched.
"""

from __future__ import annotations

import csv
import json
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("cpu_pct", "memory_pct", "latency_p99_ms", "error_rate_pct", "rps")
START_MIN, END_MIN = -240, 10

# (minutes relative to alert, level, source, message)
Event = tuple[float, str, str, str]
MetricFn = Callable[[str, float, random.Random], float]


@dataclass
class Scenario:
    name: str
    service: str
    fired_at: datetime
    metric: MetricFn
    events: Callable[[random.Random], list[Event]]
    request_paths: list[str] = field(default_factory=list)


def ramp(t: float, start: float, minutes: float) -> float:
    return min(max((t - start) / minutes, 0.0), 1.0)


def jitter(rng: random.Random, spread: float) -> float:
    return rng.uniform(-spread, spread)


# --- high-latency: traffic surge saturates CPU; autoscaler already at max -------


def high_latency_metric(metric: str, t: float, rng: random.Random) -> float:
    surge = ramp(t, -45, 30)
    cpu = min(38 + 58 * surge + jitter(rng, 3), 99.0)
    return {
        "rps": 400 + 600 * surge + jitter(rng, 15),
        "cpu_pct": cpu,
        "latency_p99_ms": 180 + max(cpu - 85, 0) * 110 + jitter(rng, 15),
        "error_rate_pct": 0.2 + (0.4 if cpu > 90 else 0) + jitter(rng, 0.05),
        "memory_pct": 55 + jitter(rng, 2),
    }[metric]


def high_latency_events(rng: random.Random) -> list[Event]:
    events: list[Event] = [
        (-40, "INFO", "autoscaler", "scaling checkout-api from 4 to 6 replicas (cpu 78%)"),
        (-28, "WARN", "autoscaler", "checkout-api at max replicas (6/6); cannot scale further"),
    ]
    for t in range(-22, 6):
        depth = 20 + (t + 22) * 4 + rng.randint(0, 6)
        events.append((t, "WARN", "app", f"worker pool saturated: 64/64 busy, queue depth {depth}"))
    for t in range(-15, 6, 3):
        ms = 1100 + rng.randint(0, 500)
        events.append((t + 0.5, "WARN", "app", f"slow request POST /api/checkout took {ms}ms"))
    for _ in range(8):
        t = rng.uniform(-12, 3)
        events.append((t, "ERROR", "app", "upstream timeout calling pricing-service after 2000ms"))
    return events


# --- error-spike-after-deploy: v3.8.0 breaks refunds and charges ----------------


def deploy_spike_metric(metric: str, t: float, rng: random.Random) -> float:
    after = t >= -7
    return {
        "rps": 250 + jitter(rng, 10),
        "cpu_pct": 45 + (3 if after else 0) + jitter(rng, 3),
        "latency_p99_ms": (300 if after else 220) + jitter(rng, 25),
        "error_rate_pct": (11 + jitter(rng, 1.5)) if after else 0.3 + jitter(rng, 0.08),
        "memory_pct": 60 + jitter(rng, 2),
    }[metric]


def deploy_spike_events(rng: random.Random) -> list[Event]:
    events: list[Event] = [
        (
            -9.5,
            "INFO",
            "deployer",
            "deploy started: payments-api v3.8.0 (previous v3.7.2), triggered by ci-pipeline",
        ),
        (-8, "INFO", "deployer", "deploy finished: payments-api v3.8.0 rolled out to 8/8 pods"),
    ]
    for minute in range(-7, 10):
        for _ in range(rng.randint(20, 28)):
            events.append(
                (
                    minute + rng.random(),
                    "ERROR",
                    "app",
                    "POST /v1/refunds 500: KeyError: 'currency' in refund_mapper.py:88",
                )
            )
        for _ in range(rng.randint(3, 6)):
            events.append(
                (
                    minute + rng.random(),
                    "ERROR",
                    "app",
                    "POST /v1/charges 500: KeyError: 'currency' in charge_mapper.py:41",
                )
            )
    return events


# --- memory-leak: unbounded cache grows until the heap is exhausted -------------


def memory_leak_metric(metric: str, t: float, rng: random.Random) -> float:
    memory = min(50 + 47 * ramp(t, START_MIN, -START_MIN) + jitter(rng, 1), 98.5)
    return {
        "rps": 180 + jitter(rng, 8),
        "memory_pct": memory,
        "cpu_pct": 35 + max(memory - 85, 0) * 2.2 + jitter(rng, 3),
        "latency_p99_ms": 150 + max(memory - 88, 0) * 90 + jitter(rng, 15),
        "error_rate_pct": (1.6 + jitter(rng, 0.3)) if t > -8 else 0.1 + jitter(rng, 0.03),
    }[metric]


def memory_leak_events(rng: random.Random) -> list[Event]:
    events: list[Event] = []
    for i, t in enumerate(range(-230, 1, 30)):
        entries = 120 + i * 95
        events.append(
            (t, "INFO", "app", f"cache stats: stock_snapshot_cache entries={entries}k evictions=0")
        )
    for t in range(-120, 6, 5):
        full = min(80 + (t + 120) * 0.14, 99)
        pause = int(200 + (t + 120) * 14 + rng.randint(0, 150))
        events.append((t + 0.2, "WARN", "app", f"GC pause {pause}ms (old gen {full:.0f}% full)"))
    for t in (-6, -3, -1):
        events.append(
            (
                t,
                "ERROR",
                "app",
                "java.lang.OutOfMemoryError: Java heap space in StockSnapshotCache.load",
            )
        )
    for _ in range(12):
        sku = f"SKU-{rng.randint(10000, 99999)}"
        events.append(
            (rng.uniform(-7, 5), "ERROR", "app", f"GET /v2/stock/{sku} 503: heap exhausted")
        )
    return events


# --- intermittent-5xx: bursts with two plausible causes and a config change -----

BURSTS = [(-30 + 6 * i, -28 + 6 * i) for i in range(7)]


def in_burst(t: float) -> bool:
    return any(lo <= t < hi for lo, hi in BURSTS)


def intermittent_metric(metric: str, t: float, rng: random.Random) -> float:
    burst = in_burst(t)
    return {
        "rps": 320 + jitter(rng, 12),
        "cpu_pct": 42 + jitter(rng, 3),
        "memory_pct": 58 + jitter(rng, 2),
        "latency_p99_ms": (820 + jitter(rng, 120)) if burst else 240 + jitter(rng, 20),
        "error_rate_pct": (3.2 + jitter(rng, 0.7)) if burst else 0.2 + jitter(rng, 0.05),
    }[metric]


def intermittent_events(rng: random.Random) -> list[Event]:
    events: list[Event] = [
        (
            -33,
            "INFO",
            "config",
            "feature flag 'split-shipments' changed from 10% to 50% by ops-console",
        ),
    ]
    for lo, hi in BURSTS:
        for _ in range(6):
            sku = f"SKU-{rng.randint(10000, 99999)}"
            events.append(
                (
                    rng.uniform(lo, hi),
                    "ERROR",
                    "app",
                    f"upstream inventory-api responded 503 for GET /v2/stock/{sku}",
                )
            )
        for _ in range(4):
            events.append(
                (
                    rng.uniform(lo, hi),
                    "ERROR",
                    "app",
                    "db: timeout acquiring connection from pool orders-db after 5000ms",
                )
            )
        for _ in range(3):
            events.append(
                (rng.uniform(lo, hi), "WARN", "app", "retrying inventory-api request (attempt 2/3)")
            )
    return events


SCENARIOS = [
    Scenario(
        "high-latency",
        "checkout-api",
        datetime(2026, 9, 14, 10, 30, tzinfo=UTC),
        high_latency_metric,
        high_latency_events,
        ["GET /api/cart", "POST /api/checkout", "GET /api/promotions"],
    ),
    Scenario(
        "error-spike-after-deploy",
        "payments-api",
        datetime(2026, 9, 14, 14, 10, tzinfo=UTC),
        deploy_spike_metric,
        deploy_spike_events,
        ["POST /v1/charges", "POST /v1/refunds", "GET /v1/payments"],
    ),
    Scenario(
        "memory-leak",
        "inventory-api",
        datetime(2026, 9, 14, 12, 40, tzinfo=UTC),
        memory_leak_metric,
        memory_leak_events,
        ["GET /v2/stock", "PUT /v2/stock", "GET /v2/warehouses"],
    ),
    Scenario(
        "intermittent-5xx",
        "orders-api",
        datetime(2026, 9, 14, 16, 25, tzinfo=UTC),
        intermittent_metric,
        intermittent_events,
        ["POST /v1/orders", "GET /v1/orders", "GET /v1/orders/status"],
    ),
]


def iso(ts: datetime) -> str:
    return ts.isoformat().replace("+00:00", "Z")


def write_scenario(scenario: Scenario, root: Path = ROOT) -> None:
    rng = random.Random(f"triage-graph:{scenario.name}")
    at = scenario.fired_at

    with (root / "fixtures" / "metrics" / f"{scenario.name}.csv").open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["ts", "service", "metric", "value"])
        for minute in range(START_MIN, END_MIN + 1):
            for metric in METRICS:
                value = scenario.metric(metric, minute, rng)
                writer.writerow(
                    [
                        iso(at + timedelta(minutes=minute)),
                        scenario.service,
                        metric,
                        f"{max(value, 0):.2f}",
                    ]
                )

    events = scenario.events(rng)
    for seconds in range(START_MIN * 60, END_MIN * 60, 30):
        path = rng.choice(scenario.request_paths)
        events.append((seconds / 60, "INFO", "app", f"{path} 200 in {rng.randint(20, 90)}ms"))
    events.sort(key=lambda e: e[0])

    with (root / "fixtures" / "logs" / f"{scenario.name}.jsonl").open("w") as f:
        for minutes, level, source, msg in events:
            ts = (at + timedelta(minutes=minutes)).replace(microsecond=0)
            record = {
                "ts": iso(ts),
                "level": level,
                "service": scenario.service,
                "source": source,
                "msg": msg,
            }
            f.write(json.dumps(record) + "\n")


def main(root: Path = ROOT) -> None:
    (root / "fixtures" / "logs").mkdir(parents=True, exist_ok=True)
    (root / "fixtures" / "metrics").mkdir(parents=True, exist_ok=True)
    for scenario in SCENARIOS:
        write_scenario(scenario, root)
        print(f"wrote {scenario.name}")


if __name__ == "__main__":
    main()
