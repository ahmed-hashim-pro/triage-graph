# General incident policy

## Escalation

When the evidence does not point to a single cause, or the fix is outside the
allowed automated actions, page a human with what has been found so far.

Suggested action: `page_human`

## Allowed automated actions

Only four actions can be proposed for automated execution: restart_service,
scale_up, rollback_deploy, and page_human. Each one runs only after a human
approves it. Anything else must be done by a person.
