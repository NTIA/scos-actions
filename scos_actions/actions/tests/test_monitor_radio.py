from scos_actions.actions.interfaces.signals import monitor_action_completed
from scos_actions.actions.tests.utils import SENSOR_DEFINITION
from scos_actions.discover import test_actions as actions

MONITOR_RADIO_SCHEDULE = {
    "name": "test_monitor",
    "start": None,
    "stop": None,
    "interval": None,
    "action": "test_monitor_radio",
}


def test_monitor_radio_not_available():
    _radio_healthy = None

    def callback(sender, **kwargs):
        nonlocal _radio_healthy
        _radio_healthy = kwargs["radio_healthy"]

    monitor_action_completed.connect(callback)
    action = actions["test_monitor_radio"]
    radio = action.radio
    radio._is_available = False
    action(MONITOR_RADIO_SCHEDULE, 1, SENSOR_DEFINITION)
    assert _radio_healthy == False
    radio._is_available = True


def test_monitor_radio_not_healthy():
    _radio_healthy = None

    def callback(sender, **kwargs):
        nonlocal _radio_healthy
        _radio_healthy = kwargs["radio_healthy"]

    monitor_action_completed.connect(callback)
    action = actions["test_monitor_radio"]
    radio = action.radio
    radio._healthy = False
    action(MONITOR_RADIO_SCHEDULE, 1, SENSOR_DEFINITION)
    assert _radio_healthy == False
    radio._healthy = True


def test_monitor_radio_healthy():
    _radio_healthy = None

    def callback(sender, **kwargs):
        nonlocal _radio_healthy
        _radio_healthy = kwargs["radio_healthy"]

    monitor_action_completed.connect(callback)
    action = actions["test_monitor_radio"]
    radio = action.radio
    radio._is_available = True
    radio.set_times_to_fail_recv(0)
    action(MONITOR_RADIO_SCHEDULE, 1, SENSOR_DEFINITION)
    assert _radio_healthy == True
