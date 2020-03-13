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
