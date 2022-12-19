import copy
import logging
import time
from abc import ABC, abstractmethod

from scos_actions.calibration import sensor_calibration, sigan_calibration
from scos_actions.calibration.calibration import Calibration
from scos_actions.capabilities import capabilities
from scos_actions.hardware.hardware_configuration_exception import (
    HardwareConfigurationException,
)
from scos_actions.hardware.utils import power_cycle_sigan
from scos_actions.utils import (
    convert_string_to_millisecond_iso_format,
    get_datetime_str_now,
)

logger = logging.getLogger(__name__)


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

    @abstractmethod
    def connect(self) -> None:
        """
        Establish a connection to the signal analyzer.
        """
        pass

    def healthy(self, num_samples=56000):
        """Perform health check by collecting IQ samples."""
        logger.debug("Performing health check.")
        if not self.is_available:
            return False
        try:
            measurement_result = self.acquire_time_domain_samples(
                num_samples, gain_adjust=False
            )
            data = measurement_result["data"]
        except Exception as e:
            logger.exception("Unable to acquire samples from device.")
            return False

        if not len(data) == num_samples:
            logger.error("Data length doesn't match request.")
            return False

        return True

    def power_cycle_and_connect(self, sleep_time: float = 2.0) -> None:
        """
        Attempt to cycle signal analyzer power then reconnect.

        :param sleep_time: Time (s) to wait for power to cycle, defaults to 2.0
        """
        logger.info("Attempting to power cycle the signal analyzer and reconnect.")
        try:
            power_cycle_sigan()
        except HardwareConfigurationException as hce:
            logger.warn(f"Unable to power cycle sigan: {hce}")
            return
        try:
            # Wait for power cycle to complete
            logger.debug(f"Waiting {sleep_time} seconds before reconnecting...")
            time.sleep(sleep_time)
            logger.info("Power cycled signal analyzer. Reconnecting...")
            self.connect()
        except Exception as e:
            logger.error(
                f"Unable to reconnect to signal analyzer after power cycling: {e}"
            )
        return

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

    def get_status(self):
        try:
            sigan_model = capabilities["sensor"]["signal_analyzer"]["sigan_spec"][
                "model"
            ]
            if sigan_model.lower() in ["default", ""]:
                raise KeyError
        except KeyError:
            sigan_model = str(self.__class__)
        return {"model": sigan_model, "healthy": self.healthy}
