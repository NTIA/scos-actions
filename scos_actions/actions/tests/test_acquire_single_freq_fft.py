import json
import os
from os import path

# SCHEMA_DIR = path.join(settings.REPO_ROOT, "schemas")
# SCHEMA_FNAME = "scos_transfer_spec_schema.json"
# SCHEMA_PATH = path.join(SCHEMA_DIR, SCHEMA_FNAME)

# with open(SCHEMA_PATH, "r") as f:
#     schema = json.load(f)
from scos_actions.actions.tests.utils import SENSOR_DEFINITION
from scos_actions.discover import test_actions as actions
from scos_actions.actions.interfaces.signals import measurement_action_completed

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
    # entry_name = simulate_frequency_fft_acquisitions(user_client)
    # tr = TaskResult.objects.get(schedule_entry__name=entry_name, task_id=1)
    # acquisition = Acquisition.objects.get(task_result=tr)
    # assert sigmf_validate(acquisition.metadata)
    # # FIXME: update schema so that this passes
    # # schema_validate(sigmf_metadata, schema)
    # os.remove(acquisition.data.path)

