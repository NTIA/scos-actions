import subprocess
import sys

import pytest

from scos_actions.discover import test_actions
from scos_actions.signals import location_action_completed

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
    action = test_actions["test_sync_gps"]
    if sys.platform == "linux":
        action(SYNC_GPS, 1)
        assert _latitude
        assert _longitude
    elif sys.platform == "win32":
        with pytest.raises(subprocess.CalledProcessError):
            action(SYNC_GPS, 1)
    else:
        raise NotImplementedError("Test not implemented for current OS.")
