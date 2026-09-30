"""The investigation and proposal steps of triage-graph, rebuilt as a CrewAI crew.

Importing this package turns CrewAI's telemetry and tracing off unless the
environment already says otherwise, so the port runs offline.
"""

import os

for _name, _value in {
    "CREWAI_DISABLE_TELEMETRY": "true",
    "OTEL_SDK_DISABLED": "true",
    "CREWAI_TRACING_ENABLED": "false",
    "CREWAI_STORAGE_DIR": "triage-graph",
}.items():
    os.environ.setdefault(_name, _value)
