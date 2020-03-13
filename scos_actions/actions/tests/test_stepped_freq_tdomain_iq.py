import json

from scos_actions.actions.interfaces.signals import measurement_action_completed
from scos_actions.actions.tests.utils import SENSOR_DEFINITION
from scos_actions.discover import test_actions as actions


SINGLE_TIMEDOMAIN_IQ_ACQUISITION = {
    "name": "test_acq",
    "start": None,
    "stop": None,
    "interval": None,
    "action": "test_single_frequency_iq_action",
}

SINGLE_TIMEDOMAIN_IQ_MULTI_RECORDING_ACQUISITION = {
    "name": "test_multirec_acq",
    "start": None,
    "stop": None,
    "interval": None,
    "action": "test_multi_frequency_iq_action",
}

# SCHEMA_DIR = path.join(settings.REPO_ROOT, "schemas")
# SCHEMA_FNAME = "scos_transfer_spec_schema.json"
# SCHEMA_PATH = path.join(SCHEMA_DIR, SCHEMA_FNAME)

# with open(SCHEMA_PATH, "r") as f:
#     schema = json.load(f)


# def test_metadata():
#     action = actions["test_multi_frequency_iq_action"]
#     action_results = action()
#     for action_result in action_results:
#         assert action_result.annotations
#         for annotation in action_result.annotations:
#             assert "start_index" in annotation
#             assert annotation["start_index"] >= 0
#             assert annotation["length"]
#             assert annotation["metadata"]
#         #assert sigmf_validate(acquisition.metadata)
#         # FIXME: update schema so that this passes
#         # schema_validate(sigmf_metadata, schema)
#
#
# def test_data():
#     action = actions["test_multi_frequency_iq_action"]
#     action_results = action()
#     for action_result in action_results:
#         assert action_result.acq_data.any()
#     # entry_name = simulate_multirec_acquisition(user_client)
#     # tr = TaskResult.objects.get(schedule_entry__name=entry_name, task_id=1)
#     # acquisitions = Acquisition.objects.filter(task_result=tr)
#     # for acquisition in acquisitions:
#     #     assert acquisition.data
#     #     assert path.exists(acquisition.data.path)
#     #     os.remove(acquisition.data.path)


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
    action(SINGLE_TIMEDOMAIN_IQ_ACQUISITION, 1, SENSOR_DEFINITION)
    assert _data.any()
    assert _metadata
    assert _task_id
    # entry_name = simulate_timedomain_iq_acquisition(user_client)
    # tr = TaskResult.objects.get(schedule_entry__name=entry_name, task_id=1)
    # acquisition = Acquisition.objects.get(task_result=tr)
    # check_metadata_fields(
    #     acquisition,
    #     entry_name,
    #     SINGLE_TIMEDOMAIN_IQ_ACQUISITION,
    #     is_multirecording=False,
    # )


def test_metadata_timedomain_iq_multiple_acquisition():
    _data = None
    _metadata = None
    _task_id = 0
    _count = 0

    def callback(sender,**kwargs):
        nonlocal _data
        nonlocal _metadata
        nonlocal _task_id
        nonlocal _count
        _task_id = kwargs["task_id"]
        _data = kwargs["data"]
        _metadata = kwargs["metadata"]
        _count += 1
    measurement_action_completed.connect(callback)
    action = actions["test_multi_frequency_iq_action"]
    action(SINGLE_TIMEDOMAIN_IQ_MULTI_RECORDING_ACQUISITION, 1, SENSOR_DEFINITION)
    assert _data.any()
    assert _metadata
    assert _task_id
    assert _count == 10
