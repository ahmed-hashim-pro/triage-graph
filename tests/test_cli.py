import io
import json

import pytest
from conftest import alert_path

from triage_graph.cli import main


@pytest.fixture
def cli(tmp_path, capsys):
    def invoke(*argv, stdin=""):
        out = io.StringIO()
        code = main([*argv, "--state-dir", str(tmp_path)], stdin=io.StringIO(stdin), out=out)
        return code, out.getvalue(), capsys.readouterr().err

    invoke.state_dir = tmp_path
    return invoke


def ledger(tmp_path):
    path = tmp_path / "dry_run_ledger.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def run(cli, scenario="high-latency", thread="t1", *extra, stdin=""):
    return cli(
        "run", "--alert", str(alert_path(scenario)), "--thread-id", thread, *extra, stdin=stdin
    )


def test_run_streams_each_node_in_order_and_stops_at_the_gate(cli):
    code, out, _ = run(cli)
    assert code == 0
    who = [line.split()[0] for line in out.splitlines()[1:] if line and not line.startswith(" ")]
    expected = [
        "intake",
        "supervisor",
        "log_investigator",
        "supervisor",
        "metrics_investigator",
        "supervisor",
        "runbook_agent",
        "supervisor",
        "proposer",
        "approval",
    ]
    seen = [w for i, w in enumerate(who) if i == 0 or w != who[i - 1]]
    assert seen[: len(expected)] == expected
    assert "search_logs(start=" in out
    assert "=== Waiting for approval (thread t1) ===" in out
    assert "triage resume t1" in out
    assert ledger(cli.state_dir) == []


def test_resume_approve_executes_and_writes_the_report(cli):
    run(cli)
    code, out, _ = cli("resume", "t1", "--approve", "--approver", "ahmed")
    assert code == 0
    assert "approved by ahmed" in out
    assert "=== Outcome: executed" in out
    assert [r["action"] for r in ledger(cli.state_dir)] == ["scale_up"]
    report = (cli.state_dir / "reports" / "t1.md").read_text()
    assert "Decided by: ahmed" in report


def test_resume_reject_executes_nothing(cli):
    run(cli)
    code, out, _ = cli("resume", "t1", "--reject", "--reason", "known load test")
    assert code == 0
    assert "=== Outcome: rejected (known load test)" in out
    assert ledger(cli.state_dir) == []


def test_resume_edit_runs_the_edited_action(cli):
    run(cli)
    code, out, _ = cli("resume", "t1", "--edit", "page_human")
    assert code == 0
    assert "action changed to page_human" in out
    assert [r["action"] for r in ledger(cli.state_dir)] == ["page_human"]


def test_edit_to_an_unknown_action_is_rejected_before_resuming(cli):
    run(cli)
    code, _, err = cli("resume", "t1", "--edit", "drop_database")
    assert code == 2
    assert "--edit must be one of" in err
    code, out, _ = cli("show", "t1")
    assert "=== Waiting for approval (thread t1) ===" in out


def test_resume_errors_for_unknown_or_finished_threads(cli):
    code, _, err = cli("resume", "nope", "--approve")
    assert (code, "no run with thread id 'nope'" in err) == (1, True)
    run(cli)
    cli("resume", "t1", "--approve")
    code, _, err = cli("resume", "t1", "--approve")
    assert (code, "is not waiting for approval" in err) == (1, True)
    assert len(ledger(cli.state_dir)) == 1


def test_run_refuses_to_reuse_a_thread_id(cli):
    run(cli)
    code, _, err = run(cli)
    assert code == 1
    assert "already exists" in err


def test_wait_mode_takes_the_decision_from_stdin(cli):
    code, out, _ = run(cli, "memory-leak", "t2", "--wait", stdin="edit nonsense\napprove\n")
    assert code == 0
    assert "not understood" in out
    assert "=== Outcome: executed" in out
    assert [r["action"] for r in ledger(cli.state_dir)] == ["restart_service"]


def test_wait_mode_can_defer_the_decision(cli):
    code, out, _ = run(cli, "memory-leak", "t2", "--wait", stdin="\n")
    assert code == 0
    assert "=== Outcome" not in out
    code, out, _ = cli("resume", "t2", "--approve")
    assert "=== Outcome: executed" in out


def test_step_limit_escalates_and_the_cli_says_so(cli):
    # The fake model needs three dispatches; two are not enough.
    code, out, _ = run(cli, "high-latency", "t3", "--max-steps", "2")
    assert code == 0
    assert "supervisor           stop: insufficient evidence, escalating" in out
    assert "=== Outcome: escalated (insufficient evidence, escalating" in out
    assert "Waiting for approval" not in out
    assert ledger(cli.state_dir) == []


def test_max_steps_out_of_range_is_a_usage_error(cli):
    code, _, err = run(cli, "high-latency", "t4", "--max-steps", "21")
    assert code == 2
    assert "--max-steps must be between 1 and 20" in err


def test_show_prints_the_report_once_finished(cli):
    run(cli)
    cli("resume", "t1", "--approve")
    code, out, _ = cli("show", "t1")
    assert code == 0
    assert out.startswith("# Incident report: checkout-api p99 latency above 1s")


def test_bad_alert_file_is_reported(cli, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"id": "x"}')
    code, _, err = cli("run", "--alert", str(bad))
    assert code == 1
    assert "invalid alert file" in err
    code, _, err = cli("run", "--alert", str(tmp_path / "missing.json"))
    assert "no such alert file" in err


def test_unknown_provider_is_reported(cli):
    code, _, err = run(cli, "high-latency", "t5", "--provider", "openai")
    assert code == 1
    assert "unknown TRIAGE_PROVIDER 'openai'" in err
