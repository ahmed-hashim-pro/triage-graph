"""Agent roles. CrewAI routes delegation by role name, so these strings are an interface."""

SUPERVISOR = "Incident supervisor"
LOGS = "Log investigator"
METRICS = "Metrics investigator"
RUNBOOKS = "Runbook agent"
PROPOSER = "Remediation proposer"

# role -> the matching LangGraph specialist name
SPECIALISTS = {
    LOGS: "log_investigator",
    METRICS: "metrics_investigator",
    RUNBOOKS: "runbook_agent",
}
