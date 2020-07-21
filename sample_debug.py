"""
This is a sample file showing how an action be called for debugging purposes.
"""
from scos_actions.actions.acquire_single_freq_fft import SingleFrequencyFftAcquisition
from scos_actions.hardware import radio

parameters = {
    "name": "test_single_frequency_m4s_action",
    "frequency": 739e6,
    "gain": 40,
    "sample_rate": 15.36e6,
    "fft_size": 1024,
    "nffts": 300,
    "radio": radio,
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
action = SingleFrequencyFftAcquisition(**parameters)
action(schedule_entry_json, 1, sensor)
print("finished")
