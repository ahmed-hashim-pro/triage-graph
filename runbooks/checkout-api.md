# checkout-api

## Overview

checkout-api turns a cart into an order. It calls pricing-service for totals and
payments-api to charge the card. It runs 4 to 6 replicas under the autoscaler.

## High latency with CPU saturation

Symptoms: p99 latency climbs together with CPU above 85%, the worker pool reports
saturation, and queue depth keeps growing. The usual cause is more traffic than
the current replicas can serve (a campaign, a partner batch job, bots).

Check the request rate against its baseline and whether the autoscaler has hit
its replica ceiling. If traffic is genuinely higher and the autoscaler is capped,
raise capacity.

Suggested action: `scale_up`

Do not restart saturated pods: in-flight checkouts are dropped and the queue on
the remaining pods gets longer.

## Elevated errors after a release

If 5xx errors start within minutes of a checkout-api deploy, roll back first and
investigate afterwards.

Suggested action: `rollback_deploy`
