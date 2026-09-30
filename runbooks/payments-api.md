# payments-api

## Overview

payments-api charges cards and issues refunds through the card processor. Deploys
go out through ci-pipeline as rolling updates across 8 pods.

## Error rate spike after a deploy

Symptoms: the 5xx rate jumps within minutes of a rollout, and stack traces point
at code that changed in the release. Roll back to the previous version. Do not
attempt a forward fix during the incident.

Suggested action: `rollback_deploy`

## Card processor degradation

Symptoms: timeouts or 5xx responses from the card processor client, with no
recent deploy of payments-api. The fault is outside our systems; page the
payments on-call so they can contact the processor.

Suggested action: `page_human`
