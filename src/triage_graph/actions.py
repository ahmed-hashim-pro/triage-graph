"""The remediation allow-list. This module, not any prompt, decides what may run."""

from __future__ import annotations

from enum import StrEnum


class Action(StrEnum):
    RESTART_SERVICE = "restart_service"
    SCALE_UP = "scale_up"
    ROLLBACK_DEPLOY = "rollback_deploy"
    PAGE_HUMAN = "page_human"


ALLOWED_ACTIONS: frozenset[str] = frozenset(action.value for action in Action)


class ActionNotAllowedError(ValueError):
    pass


def require_allowed(action: object) -> Action:
    """Return the `Action` for `action`, or raise if it is not on the allow-list."""
    if not isinstance(action, str) or action not in ALLOWED_ACTIONS:
        raise ActionNotAllowedError(
            f"{action!r} is not an allowed action; allowed: {sorted(ALLOWED_ACTIONS)}"
        )
    return Action(action)
