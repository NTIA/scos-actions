from scos_actions.actions.tests.utils import check_metadata_fields
from scos_actions.discover import test_actions as actions
from scos_actions.hardware.mocks.mock_sensor import MockSensor
from scos_actions.signals import measurement_action_completed

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
    assert action.description
    action(
        sensor=MockSensor(),
        schedule_entry=SINGLE_FREQUENCY_FFT_ACQUISITION,
        task_id=1,
    )
    assert _task_id
    assert _data.any()
    assert _metadata
    check_metadata_fields(
        _metadata,
        action,
        SINGLE_FREQUENCY_FFT_ACQUISITION["name"],
        SINGLE_FREQUENCY_FFT_ACQUISITION["action"],
        1,
    )
    assert "ntia-algorithm:processing" in _metadata["global"]
    assert len(_metadata["global"]["ntia-algorithm:processing"]) == 1
    assert _metadata["global"]["ntia-algorithm:processing"][0] == "fft_1"
    assert "ntia-algorithm:processing_info" in _metadata["global"]
    assert len(_metadata["global"]["ntia-algorithm:processing_info"]) == 1
    assert all(
        [
            k in _metadata["global"]["ntia-algorithm:processing_info"][0]
            for k in [
                "id",
                "equivalent_noise_bandwidth",
                "samples",
                "dfts",
                "window",
                "baseband",
                "description",
            ]
        ]
    )
    assert _metadata["global"]["ntia-algorithm:processing_info"][0]["id"] == "fft_1"
    assert "ntia-algorithm:data_products" in _metadata["global"]
    assert len(_metadata["global"]["ntia-algorithm:data_products"]) == 1
    assert all(
        [
            k in _metadata["global"]["ntia-algorithm:data_products"][0]
            for k in [
                "name",
                "series",
                "length",
                "x_units",
                "x_start",
                "x_stop",
                "x_step",
                "y_units",
                "reference",
                "description",
            ]
        ]
    )
