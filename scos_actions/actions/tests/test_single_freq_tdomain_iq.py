import pytest

from scos_actions.actions.tests.utils import check_metadata_fields
from scos_actions.discover import test_actions as actions
from scos_actions.hardware.mocks.mock_sensor import MockSensor
from scos_actions.hardware.mocks.mock_sigan import MockSignalAnalyzer
from scos_actions.signals import measurement_action_completed

SINGLE_TIMEDOMAIN_IQ_ACQUISITION = {
    "name": "test_acq",
    "start": None,
    "stop": None,
    "interval": None,
    "action": "test_single_frequency_iq_action",
}


def test_metadata_timedomain_iq_single_acquisition():
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
    action = actions["test_single_frequency_iq_action"]
    assert action.description
    sensor = MockSensor()
    action(sensor, SINGLE_TIMEDOMAIN_IQ_ACQUISITION, 1)
    assert _data.any()
    assert _metadata
    assert _task_id == 1
    check_metadata_fields(
        _metadata,
        action,
        SINGLE_TIMEDOMAIN_IQ_ACQUISITION["name"],
        SINGLE_TIMEDOMAIN_IQ_ACQUISITION["action"],
        1,
    )
    assert len(_metadata["captures"]) == 1
    assert all(
        [
            k in _metadata["captures"][0]
            for k in [
                "core:frequency",
                "core:datetime",
                "ntia-sensor:duration",
                "ntia-sensor:overload",
                "core:sample_start",
            ]
        ]
    )


def test_required_components():
    action = actions["test_single_frequency_m4s_action"]
    mock_sigan = MockSignalAnalyzer()
    mock_sigan._is_available = False
    sensor = MockSensor(signal_analyzer=mock_sigan)
    with pytest.raises(RuntimeError):
        action(sensor, SINGLE_TIMEDOMAIN_IQ_ACQUISITION, 1)
    mock_sigan._is_available = True


def test_num_samples_skip():
    action = actions["test_single_frequency_iq_action"]
    assert action.description
    action(
        sensor=MockSensor(), schedule_entry=SINGLE_TIMEDOMAIN_IQ_ACQUISITION, task_id=1
    )
    assert action.sensor.signal_analyzer._num_samples_skip == action.parameters["nskip"]
