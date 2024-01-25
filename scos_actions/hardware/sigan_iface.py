import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Optional

from its_preselector.web_relay import WebRelay
from scos_actions.hardware.utils import power_cycle_sigan

logger = logging.getLogger(__name__)


# All setting names for all supported sigans
SIGAN_SETTINGS_KEYS = [
    "sample_rate",
    "frequency",
    "gain",
    "attenuation",
    "reference_level",
    "preamp_enable",
]


class SignalAnalyzerInterface(ABC):
    def __init__(
        self,
        switches: Optional[Dict[str, WebRelay]] = None,
    ):
        self._model = "Unknown"
        self._api_version = "Unknown"
        self._firmware_version = "Unknown"
        self.switches = switches

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Returns True if sigan is initialized and ready for measurements."""
        pass

    @property
    @abstractmethod
    def plugin_version(self) -> str:
        """Returns the version of the SCOS plugin defining this interface."""
        pass

    @property
    def firmware_version(self) -> str:
        """Returns the version of the signal analyzer firmware."""
        return self._firmware_version

    @property
    def api_version(self) -> str:
        """Returns the version of the underlying signal analyzer API."""
        return self._api_version

    @abstractmethod
    def acquire_time_domain_samples(
        self,
        num_samples: int,
        num_samples_skip: int = 0,
    ) -> dict:
        """
        Acquire time domain IQ samples, scaled to Volts at
        the signal analyzer input.

        :param num_samples: Number of samples to acquire
        :param num_samples_skip: Number of samples to skip
        :return: dictionary containing data, sample_rate, frequency, capture_time, etc
        """
        pass

    @abstractmethod
    def connect(self) -> None:
        """
        Establish a connection to the signal analyzer.
        """
        pass

    def healthy(self, num_samples: int = 56000) -> bool:
        """Perform health check by collecting IQ samples."""
        logger.debug("Performing health check.")
        if not self.is_available:
            return False
        try:
            measurement_result = self.acquire_time_domain_samples(num_samples)
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
            power_cycle_sigan(self.switches)
        except Exception as hce:
            logger.warning(f"Unable to power cycle sigan: {hce}")
            return
        try:
            # Wait for power cycle to complete
            logger.debug(f"Waiting {sleep_time} seconds before reconnecting...")
            time.sleep(sleep_time)
            logger.info("Power cycled signal analyzer. Reconnecting...")
            self.connect()
        except Exception:
            logger.exception(
                "Unable to reconnect to signal analyzer after power cycling"
            )
        return

    def get_status(self) -> dict:
        return {"model": self._model, "healthy": self.healthy()}

    @property
    def model(self) -> str:
        return self._model

    @model.setter
    def model(self, value: str):
        self._model = value
