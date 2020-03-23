#from scos_actions.actions import action_classes
from scos_actions.actions import action_classes
from scos_actions.actions.monitor_radio import RadioMonitor
from scos_actions.actions.sync_gps import SyncGps
from scos_actions.discover.yaml import load_from_yaml
from scos_actions.hardware import radio
from scos_actions.settings import ACTION_DEFINITIONS_DIR

actions = {
}
test_actions = {
    "sync_gps": SyncGps(),
    "monitor_radio": RadioMonitor(radio)
}


def init(action_classes=action_classes, radio=radio, yaml_dir=ACTION_DEFINITIONS_DIR):
    yaml_actions = {}
    yaml_test_actions = {}
    for key, value in load_from_yaml(action_classes, radio=radio, yaml_dir=yaml_dir).items():
        if key.startswith("test_"):
            yaml_test_actions[key] = value
        else:
            yaml_actions[key] = value
    return yaml_actions, yaml_test_actions


yaml_actions, yaml_test_actions = init()
actions.update(yaml_actions)
test_actions.update(yaml_test_actions)
