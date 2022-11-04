from scos_actions.discover import test_actions as actions
from scos_actions.signals import measurement_action_completed

SINGLE_TIMEDOMAIN_IQ_MULTI_RECORDING_ACQUISITION = {
    "name": "test_multirec_acq",
    "start": None,
    "stop": None,
    "interval": None,
    "action": "test_multi_frequency_iq_action",
}


def test_metadata_timedomain_iq_multiple_acquisition():
    _datas = []
    _metadatas = []
    _task_ids = []
    _count = 0
    _recording_ids = []

    def callback(sender, **kwargs):
        nonlocal _datas
        nonlocal _metadatas
        nonlocal _task_ids
        nonlocal _count
        nonlocal _recording_ids
        _task_ids.append(kwargs["task_id"])
        _datas.append(kwargs["data"])
        _metadatas.append(kwargs["metadata"])
        _count += 1
        _recording_ids.append(kwargs["metadata"]["global"]["ntia-scos:recording"])

    measurement_action_completed.connect(callback)
    action = actions["test_multi_frequency_iq_action"]
    assert action.description
    action(SINGLE_TIMEDOMAIN_IQ_MULTI_RECORDING_ACQUISITION, 1)
    for i in range(_count):
        assert _datas[i].any()
        assert _metadatas[i]
        assert _task_ids[i] == 1
        assert _recording_ids[i] == i + 1
    assert _count == 10


def test_num_samples_skip():
    action = actions["test_multi_frequency_iq_action"]
    assert action.description
    action(SINGLE_TIMEDOMAIN_IQ_MULTI_RECORDING_ACQUISITION, 1)
    if isinstance(action.parameters["nskip"], list):
        assert action.sigan._num_samples_skip == action.parameters["nskip"][-1]
    else:
        assert action.sigan._num_samples_skip == action.parameters["nskip"]
