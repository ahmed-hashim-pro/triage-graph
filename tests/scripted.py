"""Fake models that misbehave in one specific, scripted way."""

from langchain_core.messages import AIMessage

from triage_graph.fake_model import FakeTriageModel


class StubbornSupervisor(FakeTriageModel):
    """Never satisfied: always asks the log investigator for more."""

    def supervise(self, messages):
        return self.tool_call(
            "route", {"next": "log_investigator", "focus": "more", "reason": "need more evidence"}
        )


class MutedSupervisor(FakeTriageModel):
    """Answers in prose and never calls `route`."""

    def supervise(self, messages):
        return AIMessage("I think we should look at the logs.")


class ProposesAction(FakeTriageModel):
    """Proposes a fixed action and target regardless of the evidence."""

    action: str = "drop_database"
    target: str | None = None

    def propose(self, messages):
        reply = super().propose(messages)
        args = dict(reply.tool_calls[0]["args"])
        args["action"] = self.action
        if self.target is not None:
            args["target"] = self.target
        return self.tool_call("propose_action", args)


class SilentProposer(FakeTriageModel):
    def propose(self, messages):
        return AIMessage("Probably restart it?")


class EndlessLogSearch(FakeTriageModel):
    """The log investigator keeps calling its tool and never summarizes."""

    def investigate_logs(self, messages):
        return self.tool_call(
            "search_logs",
            {"start": "2026-09-14T10:00:00Z", "end": "2026-09-14T10:30:00Z", "min_level": "WARN"},
        )


class BadTimestamps(FakeTriageModel):
    """Passes an unparseable timestamp once, then behaves."""

    def investigate_logs(self, messages):
        if not any(m.type == "tool" for m in messages):
            return self.tool_call("search_logs", {"start": "an hour ago", "end": "now"})
        return AIMessage("the log search failed")
