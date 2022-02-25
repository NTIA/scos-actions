from scos_actions.actions.interfaces.signals import monitor_action_completed
from scos_actions.actions.tests.utils import SENSOR_DEFINITION
from scos_actions.discover import test_actions as actions

MONITOR_SIGAN_SCHEDULE = {
    "name": "test_monitor",
    "start": None,
    "stop": None,
    "interval": None,
    "action": "test_monitor_sigan",
}


def test_monitor_sigan_not_available():
    _sigan_healthy = None

    def callback(sender, **kwargs):
        nonlocal _sigan_healthy
        _sigan_healthy = kwargs["sigan_healthy"]

    monitor_action_completed.connect(callback)
    action = actions["test_monitor_sigan"]
    sigan = action.sigan
    sigan._is_available = False
    action(MONITOR_SIGAN_SCHEDULE, 1, SENSOR_DEFINITION)
    assert _sigan_healthy == False
    sigan._is_available = True


def test_monitor_sigan_not_healthy():
    _sigan_healthy = None

    def callback(sender, **kwargs):
        nonlocal _sigan_healthy
        _sigan_healthy = kwargs["sigan_healthy"]

    monitor_action_completed.connect(callback)
    action = actions["test_monitor_sigan"]
    sigan = action.sigan
    sigan._healthy = False
    action(MONITOR_SIGAN_SCHEDULE, 1, SENSOR_DEFINITION)
    assert _sigan_healthy == False
    sigan._healthy = True


def test_monitor_sigan_healthy():
    _sigan_healthy = None

    def callback(sender, **kwargs):
        nonlocal _sigan_healthy
        _sigan_healthy = kwargs["sigan_healthy"]

    monitor_action_completed.connect(callback)
    action = actions["test_monitor_sigan"]
    sigan = action.sigan
    sigan._is_available = True
    sigan.set_times_to_fail_recv(0)
    action(MONITOR_SIGAN_SCHEDULE, 1, SENSOR_DEFINITION)
    assert _sigan_healthy == True
