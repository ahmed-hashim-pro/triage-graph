import pytest

from triage_graph.actions import ALLOWED_ACTIONS, Action, ActionNotAllowedError, require_allowed


def test_allow_list_is_exactly_the_four_actions():
    assert {"restart_service", "scale_up", "rollback_deploy", "page_human"} == ALLOWED_ACTIONS


@pytest.mark.parametrize("action", sorted(ALLOWED_ACTIONS))
def test_allowed_actions_pass(action):
    assert require_allowed(action) == Action(action)


@pytest.mark.parametrize(
    "action",
    ["delete_database", "RESTART_SERVICE", "restart_service ", "", None, 3, ["scale_up"], {}],
)
def test_anything_else_is_refused(action):
    with pytest.raises(ActionNotAllowedError):
        require_allowed(action)
