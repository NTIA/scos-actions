from scos_actions.actions.interfaces.signals import measurement_action_completed
from scos_actions.actions.tests.utils import SENSOR_DEFINITION, check_metadata_fields
from scos_actions.discover import test_actions as actions

SINGLE_FREQUENCY_FFT_ACQUISITION = {
    "name": "test_acq",
    "start": None,
    "stop": None,
    "interval": None,
    "action": "test_single_frequency_m4s_action",
}


def test_detector():
    _data = None
    _metadata = None
    _task_id = 0

    def callback(sender, **kwargs):
        nonlocal _data
        nonlocal _metadata
        nonlocal _task_id
        _task_id = kwargs["task_id"]
        _data = kwargs["data"]
        _metadata = kwargs["metadata"]

    measurement_action_completed.connect(callback)
    action = actions["test_single_frequency_m4s_action"]
    action(SINGLE_FREQUENCY_FFT_ACQUISITION, 1, SENSOR_DEFINITION)
    assert _task_id
    assert _data.any()
    assert _metadata
    check_metadata_fields(
        _metadata,
        SINGLE_FREQUENCY_FFT_ACQUISITION["name"],
        SINGLE_FREQUENCY_FFT_ACQUISITION["action"],
        1,
    )
