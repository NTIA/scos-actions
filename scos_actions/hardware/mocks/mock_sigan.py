"""Mock the UHD USRP module."""
import logging
from collections import namedtuple

import numpy as np

from scos_actions.hardware.sigan_iface import SignalAnalyzerInterface
from scos_actions.utils import get_datetime_str_now

logger = logging.getLogger(__name__)

tune_result_params = ["actual_dsp_freq", "actual_rf_freq"]
MockTuneResult = namedtuple("MockTuneResult", tune_result_params)


class MockSignalAnalyzer(SignalAnalyzerInterface):
    """
    MockSignalAnalyzer is mock signal analyzer object for testing.

    The following parameters are required for measurements:
    sample_rate: requested sample rate in samples/second
    frequency: center frequency in Hz
    gain: requested gain in dB
    """

    def __init__(self, randomize_values=False):
        self.auto_dc_offset = False
        self._frequency = 700e6
        self._sample_rate = 10e6
        self.clock_rate = 40e6
        self._gain = 0
        self._overload = False
        self._capture_time = None
        self._is_available = True
        self._healthy = True

        # Simulate returning less than the requested number of samples from
        # self.recv_num_samps
        self.times_to_fail_recv = 0
        self.times_failed_recv = 0

        self.randomize_values = randomize_values

    @property
    def is_available(self):
        return self._is_available

    @property
    def sample_rate(self):
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, sample_rate):
        self._sample_rate = sample_rate

    @property
    def frequency(self):
        return self._frequency

    @frequency.setter
    def frequency(self, frequency):
        self._frequency = frequency

    def configure(self, action_name):
        pass

    @property
    def gain(self):
        return self._gain

    @gain.setter
    def gain(self, gain):
        self._gain = gain

    def acquire_time_domain_samples(self, num_samples, num_samples_skip=0, retries=5, gain_adjust=True):
        self.sigan_overload = False
        self._capture_time = None
        self._num_samples_skip = num_samples_skip

        # Try to acquire the samples
        max_retries = retries
        data = []
        while True:
            if self.times_failed_recv < self.times_to_fail_recv:
                self.times_failed_recv += 1
                data = np.ones(0, dtype=np.complex64)
            else:
                self._capture_time = get_datetime_str_now()
                if self.randomize_values:
                    i = np.random.normal(0.5, 0.5, num_samples)
                    q = np.random.normal(0.5, 0.5, num_samples)
                    rand_iq = np.empty(num_samples, dtype=np.complex64)
                    rand_iq.real = i
                    rand_iq.imag = q
                    data = rand_iq
                else:
                    data = np.ones(num_samples, dtype=np.complex64)

            data_len = len(data)
            if not len(data) == num_samples:
                if retries > 0:
                    msg = "USRP error: requested {} samples, but got {}."
                    logger.warning(msg.format(num_samples + num_samples_skip, data_len))
                    logger.warning("Retrying {} more times.".format(retries))
                    retries = retries - 1
                else:
                    err = "Failed to acquire correct number of samples "
                    err += "{} times in a row.".format(max_retries)
                    raise RuntimeError(err)
            else:
                logger.debug("Successfully acquired {} samples.".format(num_samples))
                return {
                    "data": data,
                    "overload": self._overload,
                    "frequency": self._frequency,
                    "gain": self._gain,
                    "sample_rate": self._sample_rate,
                    "capture_time": self._capture_time,
                    "calibration_annotation": self.create_calibration_annotation(),
                }

    def create_calibration_annotation(self):
        annotation_md = {
            "ntia-core:annotation_type": "CalibrationAnnotation",
            "ntia-sensor:gain_sigan": self.gain,
        }
        return annotation_md

    def set_times_to_fail_recv(self, n):
        self.times_to_fail_recv = n
        self.times_failed_recv = 0

    @property
    def last_calibration_time(self):
        return get_datetime_str_now()

    @property
    def healthy(self):
        return self._healthy
