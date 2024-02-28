from scos_actions.actions import action_classes
from scos_actions.actions.logger import Logger
from scos_actions.actions.monitor_sigan import MonitorSignalAnalyzer
from scos_actions.actions.runtime_exception import RuntimeException
from scos_actions.actions.sync_gps import SyncGps
from scos_actions.actions.system_exit_action import SystemExitAction
from scos_actions.discover.yaml import load_from_yaml
from scos_actions.settings import ACTION_DEFINITIONS_DIR

actions = {
    "logger": Logger(),
    "RuntimeException": RuntimeException(),
    "SystemExit": SystemExitAction(),
}
test_actions = {
    "test_sync_gps": SyncGps(parameters={"name": "test_sync_gps"}),
    "test_monitor_sigan": MonitorSignalAnalyzer(
        parameters={"name": "test_monitor_sigan"}
    ),
    "logger": Logger(),
}


def init(
    action_classes=action_classes,
    yaml_dir=ACTION_DEFINITIONS_DIR,
):
    yaml_actions = {}
    yaml_test_actions = {}
    for key, value in load_from_yaml(action_classes, yaml_dir=yaml_dir).items():
        if key.startswith("test_"):
            yaml_test_actions[key] = value
        else:
            yaml_actions[key] = value
    return yaml_actions, yaml_test_actions


yaml_actions, yaml_test_actions = init()
actions.update(yaml_actions)
test_actions.update(yaml_test_actions)
