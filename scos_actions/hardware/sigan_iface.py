import copy
from abc import ABC, abstractmethod

from scos_actions.calibration import Calibration
from scos_actions.settings import sensor_calibration, sigan_calibration
from scos_actions.utils import (
    convert_string_to_millisecond_iso_format,
    get_datetime_str_now,
)


class SignalAnalyzerInterface(ABC):
    def __init__(self):
        # Define the default calibration dicts
        self.DEFAULT_SIGAN_CALIBRATION = {
            "gain_sigan": None,  # Defaults to gain setting
            "enbw_sigan": None,  # Defaults to sample rate
            "noise_figure_sigan": 0,
            "1db_compression_sigan": 100,
            "calibration_datetime": get_datetime_str_now(),
        }

        self.DEFAULT_SENSOR_CALIBRATION = {
            "gain_sensor": None,  # Defaults to sigan gain
            "enbw_sensor": None,  # Defaults to sigan enbw
            "noise_figure_sensor": None,  # Defaults to sigan noise figure
            "1db_compression_sensor": None,  # Defaults to sigan compression + preselector gain
            "gain_preselector": 0,
            "noise_figure_preselector": 0,
            "1db_compression_preselector": 100,
            "calibration_datetime": get_datetime_str_now(),
        }
        self.sensor_calibration_data = copy.deepcopy(self.DEFAULT_SENSOR_CALIBRATION)
        self.sigan_calibration_data = copy.deepcopy(self.DEFAULT_SIGAN_CALIBRATION)

    @property
    def last_calibration_time(self) -> str:
        """Returns the last calibration time from calibration data."""
        return convert_string_to_millisecond_iso_format(
            sensor_calibration.calibration_datetime
        )

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Returns True if sigan is initialized and ready for measurements."""
        pass

    @abstractmethod
    def acquire_time_domain_samples(
        self,
        num_samples: int,
        num_samples_skip: int = 0,
        retries: int = 5,
        gain_adjust: bool = True,
    ) -> dict:
        """
        Acquire time domain IQ samples

        :param num_samples: Number of samples to acquire
        :param num_samples_skip: Number of samples to skip
        :param retries: Maximum number of retries on failure
        :param gain_adjust: If True, scale IQ samples based on calibration data.
        :return: dictionary containing data, sample_rate, frequency, capture_time, etc
        """
        pass

    @property
    @abstractmethod
    def healthy(self) -> bool:
        """Perform a health check by collecting IQ samples."""
        pass

    def recompute_calibration_data(self, cal_args: Calibration) -> None:
        """Set the calibration data based on the current tuning"""

        # Try and get the sensor calibration data
        self.sensor_calibration_data = self.DEFAULT_SENSOR_CALIBRATION.copy()
        if sensor_calibration is not None:
            self.sensor_calibration_data.update(
                sensor_calibration.get_calibration_dict(cal_args)
            )

        # Try and get the sigan calibration data
        self.sigan_calibration_data = self.DEFAULT_SIGAN_CALIBRATION.copy()
        if sigan_calibration is not None:
            self.sigan_calibration_data.update(
                sigan_calibration.get_calibration_dict(cal_args)
            )
