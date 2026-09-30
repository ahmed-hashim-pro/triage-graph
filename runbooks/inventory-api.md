# inventory-api

## Overview

inventory-api serves stock levels for orders-api and the storefront. It is a JVM
service with a 4 GiB heap per pod.

## Memory growth and OutOfMemoryError

Symptoms: heap usage rises steadily over hours, GC pauses get longer, and the
service eventually throws OutOfMemoryError and returns 503. Known cause:
stock_snapshot_cache has no eviction policy (ticket INV-311), so its entry count
only grows.

Mitigation: a rolling restart clears the heap and buys roughly four to six hours.
Add the incident to INV-311 afterwards.

Suggested action: `restart_service`

## Stale stock counts

Symptoms: stock levels lag the warehouse feed by more than 10 minutes. This is
usually the upstream feed, not inventory-api; page the inventory on-call.

Suggested action: `page_human`
