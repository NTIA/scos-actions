from scos_actions.discover import test_actions as actions
from scos_actions.signals import trigger_api_restart

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

    trigger_api_restart.connect(callback)
    action = actions["test_monitor_sigan"]
    sigan = action.sigan
    sigan._is_available = False
    action(MONITOR_SIGAN_SCHEDULE, 1)
    assert _sigan_healthy == False
    sigan._is_available = True


def test_monitor_sigan_not_healthy():
    _sigan_healthy = None

    def callback(sender, **kwargs):
        nonlocal _sigan_healthy
        _sigan_healthy = kwargs["sigan_healthy"]

    trigger_api_restart.connect(callback)
    action = actions["test_monitor_sigan"]
    sigan = action.sigan
    sigan._healthy = False
    action(MONITOR_SIGAN_SCHEDULE, 1)
    assert _sigan_healthy == False
    sigan._healthy = True


def test_monitor_sigan_healthy():
    _sigan_healthy = None

    def callback(sender, **kwargs):
        nonlocal _sigan_healthy
        _sigan_healthy = kwargs["sigan_healthy"]

    trigger_api_restart.connect(callback)
    action = actions["test_monitor_sigan"]
    sigan = action.sigan
    sigan._is_available = True
    sigan.set_times_to_fail_recv(0)
    action(MONITOR_SIGAN_SCHEDULE, 1)
    assert _sigan_healthy == True
