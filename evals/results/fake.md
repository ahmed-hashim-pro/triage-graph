# Eval report: fake

- Started: 2026-09-30T22:37:44+00:00
- Runs per scenario and variant: 6
- Runs completed: 48 of 48; errors: 0
- Tokens in/out: 426,270/34,068 (chars / 4 estimates from the fake model, not real counts)

## Read this first

- **This is the fake model.** Its rules were written against these fixtures, so matching every scenario is by construction. It says nothing about how an LLM would do. It exists so the eval harness is exercised by the offline test suite.
- A run matches when the proposed action equals the scenario's expected action. A run that ends in escalation (step limit) is a miss, even where the expected action is page_human.
- **Runbook bias.** Each service's runbook has a section whose `Suggested action` is the expected answer. The `with-suggestions` column partly measures "found and followed the runbook".
- The `suggestions-hidden` column removes the `Suggested action` lines at the source (the runbook tool), so no agent sees them. The sections' prose still describes the fix (for example "Roll back to the previous version"), so this is a partial ablation, not a runbook-free baseline.

## Match rate

| Scenario | Expected | with-suggestions | suggestions-hidden |
|---|---|---|---|
| error-spike-after-deploy | rollback_deploy | 6/6 | 6/6 |
| high-latency | scale_up | 6/6 | 6/6 |
| intermittent-5xx | page_human | 6/6 | 6/6 |
| memory-leak | restart_service | 6/6 | 6/6 |
| **all** | | 24/24 (100%) | 24/24 (100%) |

## Averages per run

| Scenario | Variant | Supervisor steps | Model calls | Input tokens | Output tokens | Seconds |
|---|---|---|---|---|---|---|
| error-spike-after-deploy | with-suggestions | 3.0 | 12.0 | 8,613 | 684 | 0.0 |
| error-spike-after-deploy | suggestions-hidden | 3.0 | 12.0 | 8,520 | 674 | 0.0 |
| high-latency | with-suggestions | 3.0 | 12.0 | 9,536 | 747 | 0.0 |
| high-latency | suggestions-hidden | 3.0 | 12.0 | 9,449 | 739 | 0.0 |
| intermittent-5xx | with-suggestions | 3.0 | 12.0 | 8,853 | 705 | 0.0 |
| intermittent-5xx | suggestions-hidden | 3.0 | 12.0 | 8,790 | 697 | 0.0 |
| memory-leak | with-suggestions | 3.0 | 12.0 | 8,688 | 721 | 0.0 |
| memory-leak | suggestions-hidden | 3.0 | 12.0 | 8,596 | 711 | 0.0 |

## Every run

- error-spike-after-deploy / with-suggestions / run 1: rollback_deploy, match
- error-spike-after-deploy / suggestions-hidden / run 1: rollback_deploy, match
- high-latency / with-suggestions / run 1: scale_up, match
- high-latency / suggestions-hidden / run 1: scale_up, match
- intermittent-5xx / with-suggestions / run 1: page_human, match
- intermittent-5xx / suggestions-hidden / run 1: page_human, match
- memory-leak / with-suggestions / run 1: restart_service, match
- memory-leak / suggestions-hidden / run 1: restart_service, match
- error-spike-after-deploy / with-suggestions / run 2: rollback_deploy, match
- error-spike-after-deploy / suggestions-hidden / run 2: rollback_deploy, match
- high-latency / with-suggestions / run 2: scale_up, match
- high-latency / suggestions-hidden / run 2: scale_up, match
- intermittent-5xx / with-suggestions / run 2: page_human, match
- intermittent-5xx / suggestions-hidden / run 2: page_human, match
- memory-leak / with-suggestions / run 2: restart_service, match
- memory-leak / suggestions-hidden / run 2: restart_service, match
- error-spike-after-deploy / with-suggestions / run 3: rollback_deploy, match
- error-spike-after-deploy / suggestions-hidden / run 3: rollback_deploy, match
- high-latency / with-suggestions / run 3: scale_up, match
- high-latency / suggestions-hidden / run 3: scale_up, match
- intermittent-5xx / with-suggestions / run 3: page_human, match
- intermittent-5xx / suggestions-hidden / run 3: page_human, match
- memory-leak / with-suggestions / run 3: restart_service, match
- memory-leak / suggestions-hidden / run 3: restart_service, match
- error-spike-after-deploy / with-suggestions / run 4: rollback_deploy, match
- error-spike-after-deploy / suggestions-hidden / run 4: rollback_deploy, match
- high-latency / with-suggestions / run 4: scale_up, match
- high-latency / suggestions-hidden / run 4: scale_up, match
- intermittent-5xx / with-suggestions / run 4: page_human, match
- intermittent-5xx / suggestions-hidden / run 4: page_human, match
- memory-leak / with-suggestions / run 4: restart_service, match
- memory-leak / suggestions-hidden / run 4: restart_service, match
- error-spike-after-deploy / with-suggestions / run 5: rollback_deploy, match
- error-spike-after-deploy / suggestions-hidden / run 5: rollback_deploy, match
- high-latency / with-suggestions / run 5: scale_up, match
- high-latency / suggestions-hidden / run 5: scale_up, match
- intermittent-5xx / with-suggestions / run 5: page_human, match
- intermittent-5xx / suggestions-hidden / run 5: page_human, match
- memory-leak / with-suggestions / run 5: restart_service, match
- memory-leak / suggestions-hidden / run 5: restart_service, match
- error-spike-after-deploy / with-suggestions / run 6: rollback_deploy, match
- error-spike-after-deploy / suggestions-hidden / run 6: rollback_deploy, match
- high-latency / with-suggestions / run 6: scale_up, match
- high-latency / suggestions-hidden / run 6: scale_up, match
- intermittent-5xx / with-suggestions / run 6: page_human, match
- intermittent-5xx / suggestions-hidden / run 6: page_human, match
- memory-leak / with-suggestions / run 6: restart_service, match
- memory-leak / suggestions-hidden / run 6: restart_service, match
