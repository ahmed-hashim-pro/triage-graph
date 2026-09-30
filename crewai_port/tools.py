"""CrewAI tools wrapping the same functions the LangGraph graph uses.

LangGraph tools read the alert from injected graph state; these are built per
crew with the alert bound, which is the natural shape in CrewAI.
"""

from __future__ import annotations

import json

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from triage_graph import lc_tools, tools
from triage_graph.data import load_dataset, load_runbooks
from triage_graph.schemas import Alert


class SearchLogsArgs(BaseModel):
    start: str = Field(description="ISO-8601 UTC timestamp")
    end: str = Field(description="ISO-8601 UTC timestamp")
    min_level: str = Field(default="WARN", description="DEBUG, INFO, WARN or ERROR")
    pattern: str | None = Field(default=None, description="case-insensitive regex")


class SearchLogs(BaseTool):
    name: str = "search_logs"
    description: str = lc_tools.search_logs.description
    args_schema: type[BaseModel] = SearchLogsArgs
    alert: Alert

    def _run(
        self, start: str, end: str, min_level: str = "WARN", pattern: str | None = None
    ) -> str:
        dataset = load_dataset(self.alert.dataset)
        return json.dumps(
            tools.search_logs(dataset, self.alert.service, start, end, min_level, pattern)
        )


class AnomalyArgs(BaseModel):
    start: str = Field(description="ISO-8601 UTC timestamp")
    end: str = Field(description="ISO-8601 UTC timestamp")
    baseline_minutes: int = 60


class DetectMetricAnomalies(BaseTool):
    name: str = "detect_metric_anomalies"
    description: str = lc_tools.detect_metric_anomalies.description
    args_schema: type[BaseModel] = AnomalyArgs
    alert: Alert

    def _run(self, start: str, end: str, baseline_minutes: int = 60) -> str:
        dataset = load_dataset(self.alert.dataset)
        return json.dumps(
            tools.detect_metric_anomalies(dataset, self.alert.service, start, end, baseline_minutes)
        )


class SeriesArgs(BaseModel):
    metric: str
    start: str
    end: str
    step_minutes: int = 5


class GetMetricSeries(BaseTool):
    name: str = "get_metric_series"
    description: str = lc_tools.get_metric_series.description
    args_schema: type[BaseModel] = SeriesArgs
    alert: Alert

    def _run(self, metric: str, start: str, end: str, step_minutes: int = 5) -> str:
        dataset = load_dataset(self.alert.dataset)
        return json.dumps(
            tools.get_metric_series(dataset, self.alert.service, metric, start, end, step_minutes)
        )


class RunbookArgs(BaseModel):
    query: str
    service: str | None = None


class SearchRunbooks(BaseTool):
    name: str = "search_runbooks"
    description: str = lc_tools.make_search_runbooks(True).description
    args_schema: type[BaseModel] = RunbookArgs

    def _run(self, query: str, service: str | None = None) -> str:
        return json.dumps(tools.search_runbooks(load_runbooks(), query, service))
