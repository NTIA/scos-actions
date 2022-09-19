"""
This is a sample file showing how an action be created and called for debugging purposes
using a mock signal analyzer.
"""
import json

from scos_actions.actions.acquire_single_freq_fft import SingleFrequencyFftAcquisition
from scos_actions.actions.interfaces.signals import measurement_action_completed
from scos_actions.hardware.mocks.mock_sigan import MockSignalAnalyzer

parameters = {
    "name": "test_single_frequency_m4s_action",
    "frequency": 739e6,
    "gain": 40,
    "sample_rate": 15.36e6,
    "fft_size": 1024,
    "nffts": 300,
    "nskip": 0,
    "classification": "UNCLASSIFIED"
}
schedule_entry_json = {
    "name": "test_m4s_multi_1",
    "start": "2020-04-10T18:47:00.000Z",
    "stop": "2020-04-10T18:47:02.000Z",
    "interval": 1,
    "priority": 10,
    "id": "test_m4s_multi_1",
}
sensor = {
    "id": "",
    "sensor_spec": {"id": "", "model": "greyhound"},
    "antenna": {"antenna_spec": {"id": "", "model": "L-com HG3512UP-NF"}},
    "signal_analyzer": {"sigan_spec": {"id": "", "model": "Ettus USRP B210"}},
    "computer_spec": {"id": "", "model": "Intel NUC"},
}

_data = None
_metadata = None
_task_id = 0


def callback(sender, **kwargs):
    global _data
    global _metadata
    global _task_id
    _task_id = kwargs["task_id"]
    _data = kwargs["data"]
    _metadata = kwargs["metadata"]


measurement_action_completed.connect(callback)

action = SingleFrequencyFftAcquisition(parameters=parameters, sigan=MockSignalAnalyzer(randomize_values=True))
action(schedule_entry_json, 1)
print("metadata:")
print(json.dumps(_metadata, indent=4))
print("finished")
