"""Steps 1-6 of the triage workflow (investigate, then propose) as a CrewAI crew.

Mapping from the LangGraph version:

- supervisor loop  -> a delegating "Incident supervisor" agent owning the first task
- specialists      -> coworker agents with the same tools and system prompts
- step limit       -> a before-tool-call hook that counts the supervisor's delegations
                      and blocks the one past `max_steps`
- proposer         -> a ConditionalTask, skipped when the step limit was hit
- allow-list       -> the same `validate_proposal` code, applied to the proposer's JSON
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from crewai import Agent, BaseLLM, Crew, Process, Task
from crewai.hooks import (
    ToolCallHookContext,
    register_before_tool_call_hook,
    unregister_before_tool_call_hook,
)
from crewai.state.checkpoint_config import CheckpointConfig
from crewai.tasks.conditional_task import ConditionalTask

from crewai_port.roles import LOGS, METRICS, PROPOSER, RUNBOOKS, SPECIALISTS, SUPERVISOR
from crewai_port.tools import DetectMetricAnomalies, GetMetricSeries, SearchLogs, SearchRunbooks
from triage_graph.actions import ALLOWED_ACTIONS
from triage_graph.nodes import INSUFFICIENT_EVIDENCE
from triage_graph.prompts import PROPOSER_SYSTEM, SPECIALIST_SYSTEM, SUPERVISOR_SYSTEM
from triage_graph.proposals import validate_proposal
from triage_graph.schemas import Alert, Proposal
from triage_graph.state import DEFAULT_MAX_STEPS

COWORKER_TOOLS = ("delegate_work_to_coworker", "ask_question_to_coworker")
MAX_TOOL_ROUNDS = 4


@dataclass
class StepLimit:
    """Counts the supervisor's uses of its coworkers and blocks the one past the limit.

    Registered as a CrewAI before-tool-call hook. Hooks run inline in the agent
    executor; CrewAI's event bus runs "sync" handlers on a thread pool, so a counter
    kept there could lag behind the crew.
    """

    crew: Crew
    max_steps: int
    delegations: list[str] = field(default_factory=list)
    exceeded: bool = False
    refused: list[str] = field(default_factory=list)

    def __call__(self, context: ToolCallHookContext) -> bool | None:
        if context.crew is not self.crew or getattr(context.agent, "role", None) != SUPERVISOR:
            return None
        if context.tool_name not in COWORKER_TOOLS:
            return None
        coworker = str(context.tool_input.get("coworker", "")).strip()
        if len(self.delegations) + len(self.refused) >= self.max_steps:
            self.exceeded = True
            return False
        if coworker not in SPECIALISTS:
            self.refused.append(coworker)
            return False
        self.delegations.append(coworker)
        return None


@dataclass
class CrewTriageResult:
    status: Literal["proposed", "escalated"]
    proposal: Proposal | None
    reason: str | None
    delegations: list[str]
    evidence: str
    token_usage: dict[str, Any]


def _specialist(role: str, llm: BaseLLM | str, tools: list[Any]) -> Agent:
    name = SPECIALISTS[role]
    return Agent(
        role=role,
        goal=f"Report the {name.split('_')[0]} evidence relevant to the alert.",
        backstory=SPECIALIST_SYSTEM[name],
        tools=tools,
        llm=llm,
        allow_delegation=False,
        max_iter=MAX_TOOL_ROUNDS + 1,
    )


LLMFactory = Callable[[], BaseLLM | str]


def build_crew(
    alert: Alert,
    llm: LLMFactory,
    max_steps: int = DEFAULT_MAX_STEPS,
    checkpoint: CheckpointConfig | None = None,
) -> tuple[Crew, StepLimit, Task]:
    """Build the crew. `llm` is called once per agent.

    CrewAI's `Crew.calculate_usage_metrics` adds up each agent's LLM usage, so one LLM
    instance shared by five agents has its tokens counted five times.
    """
    alert_json = alert.model_dump_json()
    supervisor = Agent(
        role=SUPERVISOR,
        goal="Gather enough evidence from the specialists to recommend one action.",
        backstory=SUPERVISOR_SYSTEM,
        llm=llm(),
        allow_delegation=True,
        # One call per delegation, one blocked attempt, then CrewAI forces a final answer.
        max_iter=max_steps + 2,
    )
    specialists = [
        _specialist(LOGS, llm(), [SearchLogs(alert=alert)]),
        _specialist(
            METRICS, llm(), [DetectMetricAnomalies(alert=alert), GetMetricSeries(alert=alert)]
        ),
        _specialist(RUNBOOKS, llm(), [SearchRunbooks()]),
    ]
    proposer = Agent(
        role=PROPOSER,
        goal="Propose exactly one allowed remediation for a human to approve.",
        backstory=PROPOSER_SYSTEM,
        llm=llm(),
        allow_delegation=False,
        max_iter=2,
    )
    investigate = Task(
        description=(
            f"An alert fired: <alert>{alert_json}</alert>\n"
            f"Delegate to {LOGS}, {METRICS} and {RUNBOOKS} as needed. Include the alert "
            "JSON in each delegated task. Stop when the evidence is enough to recommend one "
            "action."
        ),
        expected_output="Every specialist's findings, quoted in full, one section per specialist.",
        agent=supervisor,
    )
    step_limit = StepLimit(crew=None, max_steps=max_steps)  # type: ignore[arg-type]
    propose = ConditionalTask(
        description=(
            f"Alert: <alert>{alert_json}</alert>\n"
            "Using the findings in the context, reply with only a JSON object with the keys "
            '"diagnosis", "action", "target" and "rationale". '
            f"action must be one of: {', '.join(sorted(ALLOWED_ACTIONS))}. "
            f'target must be "{alert.service}".'
        ),
        expected_output="A single JSON object and nothing else.",
        agent=proposer,
        context=[investigate],
        condition=lambda _output: not step_limit.exceeded,
    )
    crew = Crew(
        agents=[supervisor, *specialists, proposer],
        tasks=[investigate, propose],
        process=Process.sequential,
        verbose=False,
        **({"checkpoint": checkpoint} if checkpoint else {}),
    )
    step_limit.crew = crew
    return crew, step_limit, propose


_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def run_triage(
    alert: Alert, llm: LLMFactory, max_steps: int = DEFAULT_MAX_STEPS
) -> CrewTriageResult:
    crew, step_limit, propose = build_crew(alert, llm, max_steps)
    register_before_tool_call_hook(step_limit)
    try:
        output = crew.kickoff()
    finally:
        unregister_before_tool_call_hook(step_limit)

    evidence = output.tasks_output[0].raw if output.tasks_output else ""
    usage = output.token_usage.model_dump() if output.token_usage else {}
    if step_limit.exceeded:
        reason = (
            f"{INSUFFICIENT_EVIDENCE}: the supervisor asked for more after {max_steps} steps "
            f"(limit {max_steps})"
        )
        return CrewTriageResult("escalated", None, reason, step_limit.delegations, evidence, usage)

    raw_answer = propose.output.raw if propose.output else ""
    match = _JSON_OBJECT.search(raw_answer)
    try:
        args = json.loads(match.group()) if match else None
    except json.JSONDecodeError:
        args = None
    proposal = validate_proposal(
        args if isinstance(args, dict) else None,
        alert.service,
        missing_note="the proposer did not return a JSON object",
        fallback_diagnosis=raw_answer[:500],
    )
    return CrewTriageResult("proposed", proposal, None, step_limit.delegations, evidence, usage)
