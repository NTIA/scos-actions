#from scos_actions.actions import action_classes
from scos_actions.actions import action_classes
from scos_actions.actions.sync_gps import SyncGps
from scos_actions.discover.yaml import load_from_yaml

actions = {
    "sync_gps": SyncGps(),
}
test_actions = {
    "sync_gps": SyncGps(),
}


def init():
    for key, value in load_from_yaml(action_classes).items():
        if key.startswith("test_"):
            test_actions[key] = value
        else:
            actions[key] = value


init()

def get_action_with_summary(action):
    """Given an action, return the string 'action_name - summary'."""
    if action in actions:
        action_fn = actions[action]
    elif action in test_actions:
        action_fn = test_actions[action]
    else:
        raise Exception("Unknown action")
    summary = get_summary(action_fn)
    action_with_summary = action
    if summary:
        action_with_summary += " - {}".format(summary)

    return action_with_summary


def get_summary(action_fn):
    """Extract the first line of the action's description as a summary."""
    description = action_fn.description
    summary = None
    if description:
        summary = description.splitlines()[0]

    return summary
