"""Command-line interface: `triage run`, `triage resume`, `triage show`."""

from __future__ import annotations

import argparse
import getpass
import shlex
import sys
import uuid
from pathlib import Path
from typing import Any, TextIO

from langchain_core.language_models import BaseChatModel
from pydantic import ValidationError

from triage_graph.actions import ALLOWED_ACTIONS
from triage_graph.data import load_alert
from triage_graph.graph import MAX_STEPS_CEILING, build_graph
from triage_graph.infra import DryRunInfra
from triage_graph.llm import describe_model, make_model
from triage_graph.runner import RunResult, load_run, resume, sqlite_checkpointer, start
from triage_graph.state import DEFAULT_MAX_STEPS

DEFAULT_STATE_DIR = Path(".triage")
ACTIONS = sorted(ALLOWED_ACTIONS)


class CliError(Exception):
    def __init__(self, message: str, code: int = 1) -> None:
        super().__init__(message)
        self.code = code


class Progress:
    """Prints one line per stream event, so a run can be watched as it happens."""

    def __init__(self, out: TextIO) -> None:
        self.out = out

    def line(self, who: str, text: str) -> None:
        print(f"{who:<21}{text}", file=self.out, flush=True)

    def __call__(self, part: dict[str, Any]) -> None:
        if part["type"] == "custom":
            self.event(part["data"])
        elif part["type"] == "updates" and not part["ns"]:
            self.update(part["data"])

    def update(self, data: dict[str, Any]) -> None:
        if "intake" in data:
            alert = data["intake"]["alert"]
            self.line(
                "intake",
                f"{alert['id']} [{alert['severity']}] {alert['service']}: {alert['title']}",
            )
        elif "report" in data:
            self.line("report", "incident report written")
        elif "__interrupt__" in data:
            self.line("approval", "paused: waiting for a human decision")

    def event(self, e: dict[str, Any]) -> None:
        kind = e["event"]
        if kind == "route":
            if e["next"] == "retry":
                self.line(
                    "supervisor",
                    f"step {e['step']}/{e['max_steps']}: no valid route call (counts as a step)",
                )
            elif e["next"] == "propose":
                self.line("supervisor", f"enough evidence -> proposer ({e['reason']})")
            else:
                focus = f": {e['focus']}" if e.get("focus") else ""
                self.line("supervisor", f"step {e['step']}/{e['max_steps']} -> {e['next']}{focus}")
        elif kind == "escalate":
            self.line("supervisor", f"stop: {e['reason']}")
        elif kind == "tool_call":
            args = ", ".join(f"{k}={v!r}" for k, v in e["args"].items())
            self.line(e["specialist"], f"{e['tool']}({args})")
        elif kind == "finding":
            lines = e["summary"].splitlines() or [""]
            self.line(e["specialist"], f"found: {lines[0]}")
            for extra in lines[1:4]:
                self.line("", f"       {extra}")
        elif kind == "proposal":
            self.line("proposer", f"proposes {e['action']} on {e['target']} (proposal {e['id']})")
            if e.get("refused_action"):
                self.line("", f"refused: {e['note']}")
        elif kind == "decision":
            text = {"approve": "approved", "reject": "rejected", "edit": "edited"}.get(
                e["decision"], e["decision"]
            )
            if e.get("approver"):
                text += f" by {e['approver']}"
            if e.get("action"):
                text += f", action changed to {e['action']}"
            if e.get("reason"):
                text += f": {e['reason']}"
            self.line("approval", text)
        elif kind == "executed":
            self.line("executor", f"{e['detail']} (dry run)" if e["dry_run"] else e["detail"])


def _resume_hint(thread_id: str, state_dir: Path) -> str:
    extra = "" if state_dir == DEFAULT_STATE_DIR else f" --state-dir {shlex.quote(str(state_dir))}"
    return f"triage resume {thread_id}{extra}"


def print_pending(result: RunResult, state_dir: Path, out: TextIO) -> None:
    proposal = result.pending["proposal"] if result.pending else {}
    hint = _resume_hint(result.thread_id, state_dir)
    print(
        "\n".join(
            [
                "",
                f"=== Waiting for approval (thread {result.thread_id}) ===",
                f"Proposal {proposal['id']}: {proposal['action']} on {proposal['target']}",
                f"Diagnosis: {proposal['diagnosis']}",
                f"Rationale: {proposal['rationale']}",
                *([f"Note: {proposal['note']}"] if proposal.get("note") else []),
                "",
                f"  {hint} --approve",
                f'  {hint} --reject --reason "..."',
                f"  {hint} --edit {{{'|'.join(ACTIONS)}}}",
            ]
        ),
        file=out,
    )


def finish(result: RunResult, state_dir: Path, out: TextIO) -> int:
    if result.status == "awaiting_approval":
        print_pending(result, state_dir, out)
        return 0
    report_path = state_dir / "reports" / f"{result.thread_id}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(result.state["report"])
    reason = result.state.get("outcome_reason")
    print(
        f"\n=== Outcome: {result.state['outcome']}" + (f" ({reason})" if reason else ""), file=out
    )
    print(f"Report: {report_path}\n", file=out)
    print(result.state["report"], file=out)
    return 0


def _model(args: argparse.Namespace) -> BaseChatModel:
    try:
        return make_model(args.provider, args.model)
    except (RuntimeError, ValueError) as exc:
        raise CliError(str(exc)) from exc


def _decision(kind: str, proposal_id: str, approver: str, **extra: Any) -> dict[str, Any]:
    fields = {k: v for k, v in extra.items() if v is not None}
    return {"decision": kind, "proposal_id": proposal_id, "approver": approver, **fields}


def ask_for_decision(
    result: RunResult, approver: str, stdin: TextIO, out: TextIO
) -> dict[str, Any] | None:
    """Prompt on stdin. Returns None to leave the run paused (empty line or EOF)."""
    proposal_id = result.pending["proposal"]["id"] if result.pending else ""
    while True:
        print(
            "\nDecision: approve | reject [reason] | edit ACTION | Enter to decide later\n> ",
            end="",
            file=out,
            flush=True,
        )
        words = stdin.readline().strip().split(maxsplit=1)
        if not words:
            return None
        verb, rest = words[0].lower(), (words[1] if len(words) > 1 else None)
        if verb in ("a", "approve"):
            return _decision("approve", proposal_id, approver)
        if verb in ("r", "reject"):
            return _decision("reject", proposal_id, approver, reason=rest)
        if verb in ("e", "edit") and rest in ALLOWED_ACTIONS:
            return _decision("edit", proposal_id, approver, action=rest)
        print(f"not understood; edit takes one of: {', '.join(ACTIONS)}", file=out)


def cmd_run(args: argparse.Namespace, stdin: TextIO, out: TextIO) -> int:
    try:
        alert = load_alert(args.alert)
    except FileNotFoundError as exc:
        raise CliError(f"no such alert file: {args.alert}") from exc
    except ValidationError as exc:
        raise CliError(f"invalid alert file {args.alert}:\n{exc}") from exc
    if not 1 <= args.max_steps <= MAX_STEPS_CEILING:
        raise CliError(f"--max-steps must be between 1 and {MAX_STEPS_CEILING}", code=2)

    model = _model(args)
    thread_id = args.thread_id or f"{alert.id.lower()}-{uuid.uuid4().hex[:6]}"
    state_dir = args.state_dir
    with sqlite_checkpointer(state_dir / "checkpoints.db") as saver:
        graph = build_graph(
            model, checkpointer=saver, infra=DryRunInfra(state_dir / "dry_run_ledger.jsonl")
        )
        if load_run(graph, thread_id).state:
            raise CliError(f"thread {thread_id!r} already exists; use `triage resume` or `show`")
        print(
            f"triage run: thread {thread_id}, model {describe_model(model)}, state in {state_dir}",
            file=out,
        )
        progress = Progress(out)
        result = start(
            graph,
            alert.model_dump(mode="json"),
            thread_id=thread_id,
            max_steps=args.max_steps,
            on_part=progress,
        )
        if result.status == "awaiting_approval" and args.wait:
            print_pending(result, state_dir, out)
            decision = ask_for_decision(result, args.approver, stdin, out)
            if decision is not None:
                result = resume(graph, thread_id, decision, on_part=progress)
        return finish(result, state_dir, out)


def cmd_resume(args: argparse.Namespace, stdin: TextIO, out: TextIO) -> int:
    if args.edit is not None and args.edit not in ALLOWED_ACTIONS:
        raise CliError(f"--edit must be one of: {', '.join(ACTIONS)}", code=2)
    model = _model(args)
    state_dir = args.state_dir
    with sqlite_checkpointer(state_dir / "checkpoints.db") as saver:
        graph = build_graph(
            model, checkpointer=saver, infra=DryRunInfra(state_dir / "dry_run_ledger.jsonl")
        )
        current = load_run(graph, args.thread_id)
        if not current.state:
            raise CliError(f"no run with thread id {args.thread_id!r} in {state_dir}")
        if current.status != "awaiting_approval":
            raise CliError(
                f"thread {args.thread_id!r} is not waiting for approval "
                f"(outcome: {current.state.get('outcome')})"
            )
        proposal_id = current.pending["proposal"]["id"] if current.pending else ""
        if args.approve:
            decision = _decision("approve", proposal_id, args.approver, reason=args.reason)
        elif args.reject:
            decision = _decision("reject", proposal_id, args.approver, reason=args.reason)
        else:
            decision = _decision(
                "edit", proposal_id, args.approver, action=args.edit, reason=args.reason
            )
        result = resume(graph, args.thread_id, decision, on_part=Progress(out))
        return finish(result, state_dir, out)


def cmd_show(args: argparse.Namespace, stdin: TextIO, out: TextIO) -> int:
    with sqlite_checkpointer(args.state_dir / "checkpoints.db") as saver:
        # show only reads checkpoints; the model is needed to compile the graph, not called.
        graph = build_graph(make_model("fake"), checkpointer=saver)
        result = load_run(graph, args.thread_id)
    if not result.state:
        raise CliError(f"no run with thread id {args.thread_id!r} in {args.state_dir}")
    if result.status == "awaiting_approval":
        print_pending(result, args.state_dir, out)
    elif "report" in result.state:
        print(result.state["report"], file=out)
    else:
        print(f"thread {args.thread_id} has no report yet", file=out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--state-dir",
        type=Path,
        default=DEFAULT_STATE_DIR,
        help="where checkpoints, the dry-run ledger and reports live (default: .triage)",
    )
    model_opts = argparse.ArgumentParser(add_help=False)
    model_opts.add_argument(
        "--provider", help="fake (default) or anthropic; overrides TRIAGE_PROVIDER"
    )
    model_opts.add_argument("--model", help="model name for real providers; overrides TRIAGE_MODEL")
    who = argparse.ArgumentParser(add_help=False)
    who.add_argument(
        "--approver",
        default=_current_user(),
        help="name recorded with the decision (default: your login name)",
    )

    parser = argparse.ArgumentParser(
        prog="triage", description="Multi-agent incident triage with a human approval gate."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser(
        "run", parents=[common, model_opts, who], help="investigate an alert and propose a fix"
    )
    run.add_argument("--alert", required=True, type=Path, help="alert JSON file")
    run.add_argument("--thread-id", help="id for this run (default: derived from the alert)")
    run.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    run.add_argument(
        "--wait", action="store_true", help="prompt for the approval decision instead of exiting"
    )
    run.set_defaults(handler=cmd_run)

    res = sub.add_parser(
        "resume", parents=[common, model_opts, who], help="approve, reject or edit a proposal"
    )
    res.add_argument("thread_id")
    choice = res.add_mutually_exclusive_group(required=True)
    choice.add_argument("--approve", action="store_true")
    choice.add_argument("--reject", action="store_true")
    choice.add_argument("--edit", metavar="ACTION", help=f"run this instead: {', '.join(ACTIONS)}")
    res.add_argument("--reason", help="why; shown in the report")
    res.set_defaults(handler=cmd_resume)

    show = sub.add_parser(
        "show", parents=[common], help="print a pending proposal or a finished report"
    )
    show.add_argument("thread_id")
    show.set_defaults(handler=cmd_show)
    return parser


def _current_user() -> str:
    try:
        return getpass.getuser()
    except (OSError, KeyError):
        return "unknown"


def main(
    argv: list[str] | None = None, stdin: TextIO | None = None, out: TextIO | None = None
) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args, stdin or sys.stdin, out or sys.stdout)
    except CliError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.code


if __name__ == "__main__":
    sys.exit(main())
