from scos_actions.actions import action_classes
from scos_actions.actions.interfaces.signals import register_component_with_status
from scos_actions.actions.logger import Logger
from scos_actions.actions.monitor_sigan import MonitorSignalAnalyzer
from scos_actions.actions.sync_gps import SyncGps
from scos_actions.discover.yaml import load_from_yaml
from scos_actions.hardware.mocks.mock_gps import MockGPS
from scos_actions.hardware.mocks.mock_sigan import MockSignalAnalyzer
from scos_actions.settings import ACTION_DEFINITIONS_DIR, MOCK_SIGAN

mock_sigan = MockSignalAnalyzer(randomize_values=True)
mock_gps = MockGPS()
if MOCK_SIGAN:
    register_component_with_status.send(mock_sigan.__class__, component=mock_sigan)
actions = {"logger": Logger()}
test_actions = {
    "test_sync_gps": SyncGps(
        parameters={"name": "test_sync_gps"}, sigan=mock_sigan, gps=mock_gps
    ),
    "test_monitor_sigan": MonitorSignalAnalyzer(
        parameters={"name": "test_monitor_sigan"}, sigan=mock_sigan
    ),
    "logger": Logger(),
}


def init(
    action_classes=action_classes,
    sigan=mock_sigan,
    gps=mock_gps,
    yaml_dir=ACTION_DEFINITIONS_DIR,
):
    yaml_actions = {}
    yaml_test_actions = {}
    for key, value in load_from_yaml(
        action_classes, sigan=sigan, gps=gps, yaml_dir=yaml_dir
    ).items():
        if key.startswith("test_"):
            yaml_test_actions[key] = value
        else:
            yaml_actions[key] = value
    return yaml_actions, yaml_test_actions


yaml_actions, yaml_test_actions = init()
actions.update(yaml_actions)
test_actions.update(yaml_test_actions)
