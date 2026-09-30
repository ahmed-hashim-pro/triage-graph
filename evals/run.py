"""Run the eval.

    uv run python -m evals.run                                   # fake model, offline
    TRIAGE_PROVIDER=anthropic uv run python -m evals.run         # real model, asks first

Real-model runs print the plan and estimated cost, ask for confirmation (or take
--yes), and stop before any model call that could take spend past --budget-usd.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from evals.harness import (
    NO_EFFORT_MODELS,
    PRICES,
    VARIANTS,
    BudgetGuard,
    Plan,
    Price,
    RunRecord,
    estimate_tokens,
    fit_runs,
    load_expected,
    run_eval,
    write_results,
)
from triage_graph.llm import describe_model, make_model

RESULTS = Path(__file__).with_name("results")
# Six runs per scenario and variant fit the default $2 cap on this model; see DECISIONS.md.
EVAL_DEFAULT_MODEL = "claude-haiku-4-5"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m evals.run", description=__doc__)
    parser.add_argument("--provider", help="fake (default) or anthropic; overrides TRIAGE_PROVIDER")
    parser.add_argument(
        "--model",
        help=f"real-provider model; overrides TRIAGE_MODEL (default: {EVAL_DEFAULT_MODEL})",
    )
    parser.add_argument("--effort", help="low, medium, high, xhigh or max; model default if unset")
    parser.add_argument("--runs", type=int, default=6, help="runs per scenario and variant")
    parser.add_argument("--budget-usd", type=float, default=2.0, help="hard spending cap")
    parser.add_argument("--max-tokens", type=int, default=8000, help="per model call")
    parser.add_argument("--scenarios", help="comma-separated subset (default: all)")
    parser.add_argument(
        "--variants", default=",".join(VARIANTS), help=f"subset of {', '.join(VARIANTS)}"
    )
    parser.add_argument("--price-in", type=float, help="USD per million input tokens")
    parser.add_argument("--price-out", type=float, help="USD per million output tokens")
    parser.add_argument("--yes", action="store_true", help="do not ask before a paid run")
    parser.add_argument("--out", type=Path, default=RESULTS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    provider = (args.provider or os.environ.get("TRIAGE_PROVIDER") or "fake").lower()
    scenarios = args.scenarios.split(",") if args.scenarios else sorted(load_expected())
    variants = args.variants.split(",")
    unknown = [s for s in scenarios if s not in load_expected()] + [
        v for v in variants if v not in VARIANTS
    ]
    if unknown:
        print(f"error: unknown scenario or variant: {', '.join(unknown)}", file=sys.stderr)
        return 2

    model_name = args.model or os.environ.get("TRIAGE_MODEL") or EVAL_DEFAULT_MODEL
    model = make_model(provider, model_name, effort=args.effort, max_tokens=args.max_tokens)
    label = describe_model(model)
    price: Price | None = None
    runs = args.runs
    est_in, est_out = estimate_tokens(scenarios, label.split(":", 1)[-1], args.effort)

    if provider != "fake":
        name = label.split(":", 1)[1]
        if args.effort and name in NO_EFFORT_MODELS:
            print(f"error: {name} does not accept --effort", file=sys.stderr)
            return 2
        if args.price_in is not None and args.price_out is not None:
            price = Price(args.price_in, args.price_out)
        elif name in PRICES:
            price = PRICES[name]
        else:
            print(
                f"error: no price known for {name}; pass --price-in and --price-out",
                file=sys.stderr,
            )
            return 2
        cells, per_run = len(scenarios) * len(variants), price.cost(est_in, est_out)
        runs = fit_runs(args.runs, cells, per_run, args.budget_usd)
        if runs < 1:
            print(
                f"error: one run of each of the {cells} scenario/variant cells is estimated at "
                f"${per_run * cells:.2f} (${per_run:.3f} per run), over the "
                f"${args.budget_usd:.2f} budget. Use fewer --scenarios or --variants, a lower "
                "--effort, or another model.",
                file=sys.stderr,
            )
            return 2

    plan = Plan(
        model=label,
        scenarios=scenarios,
        variants=variants,
        runs=runs,
        requested_runs=args.runs,
        price=price,
        cap_usd=args.budget_usd if price else None,
        est_input_tokens=est_in,
        est_output_tokens=est_out,
    )
    print(plan.describe())
    if price is not None and not os.environ.get("ANTHROPIC_API_KEY"):
        print("\nNo ANTHROPIC_API_KEY in the environment; nothing was run.", file=sys.stderr)
        return 1
    if (
        price is not None
        and not args.yes
        and input("\nProceed with this paid run? [y/N] ").strip().lower() != "y"
    ):
        print("Nothing was run.")
        return 1

    guard = BudgetGuard(price, args.budget_usd, args.max_tokens) if price else None

    def progress(r: RunRecord) -> None:
        mark = "match" if r.matched else "miss"
        print(f"  {r.scenario:26} {r.variant:18} run {r.run}: {r.label} ({mark})", flush=True)

    result = run_eval(model, plan, guard=guard, progress=progress)
    stem = (
        "fake"
        if label == "fake"
        else f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{label.replace(':', '-')}"
    )
    report, raw = write_results(result, args.out, stem)
    print(f"\nReport: {report}\nRecords: {raw}")
    if guard:
        print(f"Spent at list price: ${guard.spent_usd:.2f} of ${args.budget_usd:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
