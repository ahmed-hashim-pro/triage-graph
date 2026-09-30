# orders-api

## Overview

orders-api creates and tracks orders. It depends on inventory-api for stock
checks and on orders-db (Postgres) for storage. Feature flags are managed from
ops-console.

## Intermittent 5xx errors

Intermittent errors almost always come from a dependency. Work out which one:
inventory-api 503s belong to the inventory team; orders-db connection-pool
timeouts belong to the database on-call.

If the errors point at more than one dependency, or line up with a feature-flag
change, do not guess. Page the orders on-call with the evidence. None of the
automated actions fixes a failing dependency, and restarting or scaling
orders-api adds load to the dependencies.

Suggested action: `page_human`
