from scos_actions.discover import test_actions as actions
from scos_actions.hardware.mocks.mock_sensor import MockSensor
from scos_actions.hardware.mocks.mock_sigan import MockSignalAnalyzer
from scos_actions.signals import trigger_api_restart

MONITOR_SIGAN_SCHEDULE = {
    "name": "test_monitor",
    "start": None,
    "stop": None,
    "interval": None,
    "action": "test_monitor_sigan",
}


def test_monitor_sigan_not_available():
    _api_restart_triggered = False

    def callback(sender, **kwargs):
        nonlocal _api_restart_triggered
        _api_restart_triggered = True

    trigger_api_restart.connect(callback)
    action = actions["test_monitor_sigan"]
    mock_sigan = MockSignalAnalyzer()
    mock_sigan._is_available = False
    sensor = MockSensor(signal_analyzer=mock_sigan)
    action(sensor, MONITOR_SIGAN_SCHEDULE, 1)
    assert _api_restart_triggered == True  # signal sent
    sensor.signal_analyzer._is_available = True


def test_monitor_sigan_not_healthy():
    _api_restart_triggered = False

    def callback(sender, **kwargs):
        nonlocal _api_restart_triggered
        _api_restart_triggered = True

    trigger_api_restart.connect(callback)
    action = actions["test_monitor_sigan"]
    mock_sigan = MockSignalAnalyzer()
    mock_sigan.times_to_fail_recv = 6
    sensor = MockSensor(signal_analyzer=mock_sigan)
    action(sensor, MONITOR_SIGAN_SCHEDULE, 1)
    assert _api_restart_triggered == True  # signal sent


def test_monitor_sigan_healthy():
    _api_restart_triggered = False

    def callback(sender, **kwargs):
        nonlocal _api_restart_triggered
        _api_restart_triggered = True

    trigger_api_restart.connect(callback)
    action = actions["test_monitor_sigan"]
    mock_sigan = MockSignalAnalyzer()
    mock_sigan._is_available = True
    mock_sigan.set_times_to_fail_recv(0)
    sensor = MockSensor(signal_analyzer=mock_sigan)
    action(sensor, MONITOR_SIGAN_SCHEDULE, 1)
    assert _api_restart_triggered == False  # signal not sent
