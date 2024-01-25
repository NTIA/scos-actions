"""Mock a signal analyzer for testing."""
import logging
from collections import namedtuple
from typing import Optional

import numpy as np
from scos_actions.hardware.sigan_iface import SignalAnalyzerInterface
from scos_actions.utils import get_datetime_str_now

from scos_actions import __version__ as SCOS_ACTIONS_VERSION

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

    def __init__(
        self,
        switches: Optional[dict] = None,
        randomize_values: bool = False,
    ):
        super().__init__(switches)
        self._model = "Mock Signal Analyzer"
        self._frequency = 700e6
        self._sample_rate = 10e6
        self.clock_rate = 40e6
        self._gain = 0
        self._attenuation = 0
        self._preamp_enable = False
        self._reference_level = -30
        self._is_available = True
        self._plugin_version = SCOS_ACTIONS_VERSION
        self._firmware_version = "1.2.3"
        self._api_version = "v1.2.3"

        # Simulate returning less than the requested number of samples from
        # self.recv_num_samps
        self.times_to_fail_recv = 0
        self.times_failed_recv = 0

        self.randomize_values = randomize_values

    @property
    def is_available(self):
        return self._is_available

    @property
    def plugin_version(self):
        return self._plugin_version

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

    @property
    def attenuation(self):
        return self._attenuation

    @attenuation.setter
    def attenuation(self, attenuation):
        self._attenuation = attenuation

    @property
    def preamp_enable(self):
        return self._preamp_enable

    @preamp_enable.setter
    def preamp_enable(self, preamp_enable):
        self._preamp_enable = preamp_enable

    @property
    def reference_level(self):
        return self._reference_level

    @reference_level.setter
    def reference_level(self, reference_level):
        self._reference_level = reference_level

    def connect(self):
        pass

    def acquire_time_domain_samples(
        self, num_samples: int, num_samples_skip: int = 0
    ) -> dict:
        logger.warning("Using mock signal analyzer!")
        overload = False
        capture_time = None

        # Try to acquire the samples
        data = []
        if self.times_failed_recv < self.times_to_fail_recv:
            self.times_failed_recv += 1
            data = np.ones(0, dtype=np.complex64)
        else:
            capture_time = get_datetime_str_now()
            if self.randomize_values:
                i = np.random.normal(0.5, 0.5, num_samples)
                q = np.random.normal(0.5, 0.5, num_samples)
                rand_iq = np.empty(num_samples, dtype=np.complex64)
                rand_iq.real = i
                rand_iq.imag = q
                data = rand_iq
            else:
                data = np.ones(num_samples, dtype=np.complex64)

        if (data_len := len(data)) != num_samples:
            err = "Failed to acquire correct number of samples: "
            err += f"got {data_len} instead of {num_samples}"
            raise RuntimeError(err)
        else:
            logger.debug(f"Successfully acquired {num_samples} samples.")
            return {
                "data": data,
                "overload": overload,
                "frequency": self._frequency,
                "gain": self._gain,
                "attenuation": self._attenuation,
                "preamp_enable": self._preamp_enable,
                "reference_level": self._reference_level,
                "sample_rate": self._sample_rate,
                "capture_time": capture_time,
            }

    def set_times_to_fail_recv(self, n):
        self.times_to_fail_recv = n
        self.times_failed_recv = 0
