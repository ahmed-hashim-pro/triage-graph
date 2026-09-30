"""Run the CrewAI port on one alert with the fake LLM.

uv run python -m crewai_port --alert fixtures/alerts/high-latency.json
"""

from __future__ import annotations

import argparse
import json
import sys

from crewai_port.crew import run_triage
from crewai_port.fake_llm import FakeCrewLLM
from triage_graph.data import load_alert
from triage_graph.state import DEFAULT_MAX_STEPS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m crewai_port", description=__doc__)
    parser.add_argument("--alert", required=True)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    args = parser.parse_args(argv)
    result = run_triage(load_alert(args.alert), FakeCrewLLM, max_steps=args.max_steps)
    print(
        json.dumps(
            {
                "status": result.status,
                "reason": result.reason,
                "delegations": result.delegations,
                "proposal": result.proposal.model_dump(mode="json") if result.proposal else None,
                "token_usage": result.token_usage,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
