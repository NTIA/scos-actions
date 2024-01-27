"""Mock a signal analyzer for testing."""
import logging
from collections import namedtuple
from typing import Optional

import numpy as np

from scos_actions import __version__ as SCOS_ACTIONS_VERSION
from scos_actions.calibration.calibration import Calibration
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

    def __init__(
        self,
        sensor_cal: Optional[Calibration] = None,
        sigan_cal: Optional[Calibration] = None,
        switches: Optional[dict] = None,
        randomize_values: bool = False,
    ):
        super().__init__(sensor_cal, sigan_cal, switches)
        # Define the default calibration dicts
        self.DEFAULT_SIGAN_CALIBRATION = {
            "datetime": get_datetime_str_now(),
            "gain": 0,  # Defaults to gain setting
            "enbw": None,  # Defaults to sample rate
            "noise_figure": 0,
            "1db_compression_point": 100,
            "temperature": 26.85,
        }

        self.DEFAULT_SENSOR_CALIBRATION = {
            "datetime": get_datetime_str_now(),
            "gain": 0,  # Defaults to sigan gain
            "enbw": None,  # Defaults to sigan enbw
            "noise_figure": None,  # Defaults to sigan noise figure
            "1db_compression_point": None,  # Defaults to sigan compression + preselector gain
            "temperature": 26.85,
        }
        self.auto_dc_offset = False
        self._frequency = 700e6
        self._sample_rate = 10e6
        self.clock_rate = 40e6
        self._gain = 0
        self._attenuation = 0
        self._preamp_enable = False
        self._reference_level = -30
        self._overload = False
        self._capture_time = None
        self._is_available = True
        self._plugin_version = SCOS_ACTIONS_VERSION
        self._firmware_version = "1.2.3"
        self._api_version = "v1.2.3"

        # Simulate returning less than the requested number of samples from
        # self.recv_num_samps
        self.times_to_fail_recv = 0
        self.times_failed_recv = 0

        self.randomize_values = randomize_values
        self.sensor_calibration_data = self.DEFAULT_SENSOR_CALIBRATION
        self.sigan_calibration_data = self.DEFAULT_SIGAN_CALIBRATION

    @property
    def is_available(self):
        return self._is_available

    @property
    def plugin_version(self):
        return self._plugin_version

    @property
    def firmware_version(self):
        return self._firmware_version

    @property
    def api_version(self):
        return self._api_version

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
        self, num_samples, num_samples_skip=0, retries=5, cal_adjust=True
    ):
        logger.warning("Using mock signal analyzer!")
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
                    msg = "Signal analyzer error: requested {} samples, but got {}."
                    logger.warning(msg.format(num_samples + num_samples_skip, data_len))
                    logger.warning(f"Retrying {retries} more times.")
                    retries = retries - 1
                else:
                    err = "Failed to acquire correct number of samples "
                    err += f"{max_retries} times in a row."
                    raise RuntimeError(err)
            else:
                logger.debug(f"Successfully acquired {num_samples} samples.")
                return {
                    "data": data,
                    "overload": self._overload,
                    "frequency": self._frequency,
                    "gain": self._gain,
                    "attenuation": self._attenuation,
                    "preamp_enable": self._preamp_enable,
                    "reference_level": self._reference_level,
                    "sample_rate": self._sample_rate,
                    "capture_time": self._capture_time,
                }

    def set_times_to_fail_recv(self, n):
        self.times_to_fail_recv = n
        self.times_failed_recv = 0

    @property
    def last_calibration_time(self):
        return get_datetime_str_now()

    def update_calibration(self, params):
        pass

    def recompute_sensor_calibration_data(self, cal_args: list) -> None:
        if self.sensor_calibration is not None:
            self.sensor_calibration_data.update(
                self._sensor_calibration.get_calibration_dict(cal_args)
            )
        else:
            logger.warning("Sensor calibration does not exist.")

    def recompute_sigan_calibration_data(self, cal_args: list) -> None:
        """Set the sigan calibration data based on the current tuning"""
        if self.sigan_calibration is not None:
            self.sigan_calibration_data.update(
                self.sigan_calibration.get_calibration_dict(cal_args)
            )
        else:
            logger.warning("Sigan calibration does not exist.")
