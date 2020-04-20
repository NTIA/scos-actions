from scos_actions.actions.interfaces.signals import location_action_completed
from scos_actions.actions.tests.utils import SENSOR_DEFINITION
from scos_actions.discover import test_actions

SYNC_GPS = {
    "name": "sync_gps",
    "start": None,
    "stop": None,
    "interval": None,
    "action": "sync_gps",
}


def test_detector():
    _latitude = None
    _longitude = None

    def callback(sender, **kwargs):
        nonlocal _latitude
        nonlocal _longitude
        _latitude = kwargs["latitude"]
        _longitude = kwargs["longitude"]

    location_action_completed.connect(callback)
    action = test_actions["sync_gps"]
    action(SYNC_GPS, 1, SENSOR_DEFINITION)
    assert _latitude
    assert _longitude
