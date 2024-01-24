import datetime
import hashlib
import json
import logging
from typing import Dict, Optional

from its_preselector.preselector import Preselector
from its_preselector.web_relay import WebRelay
from scos_actions.calibration.differential_calibration import DifferentialCalibration
from scos_actions.calibration.interfaces.calibration import Calibration
from scos_actions.calibration.sensor_calibration import SensorCalibration
from scos_actions.hardware.gps_iface import GPSInterface
from scos_actions.hardware.sigan_iface import (
    SIGAN_SETTINGS_KEYS,
    SignalAnalyzerInterface,
)
from scos_actions.utils import convert_string_to_millisecond_iso_format

logger = logging.getLogger(__name__)


class Sensor:
    def __init__(
        self,
        signal_analyzer: Optional[SignalAnalyzerInterface] = None,
        gps: Optional[GPSInterface] = None,
        preselector: Optional[Preselector] = None,
        switches: Dict[str, WebRelay] = {},
        location: Optional[dict] = None,
        capabilities: Optional[dict] = None,
        sensor_cal: Optional[SensorCalibration] = None,
        differential_cal: Optional[DifferentialCalibration] = None,
    ):
        self.signal_analyzer = signal_analyzer
        self.gps = gps
        self.preselector = preselector
        self.switches = switches
        self.location = location
        self.capabilities = capabilities
        self.sensor_calibration_data = {}
        self._sensor_calibration = sensor_cal
        self._differential_calibration = differential_cal
        # There is no setter for start_time property
        self._start_time = datetime.datetime.utcnow()

    @property
    def signal_analyzer(self) -> Optional[SignalAnalyzerInterface]:
        return self._signal_analyzer

    @signal_analyzer.setter
    def signal_analyzer(self, sigan: Optional[SignalAnalyzerInterface]):
        self._signal_analyzer = sigan

    @property
    def gps(self) -> Optional[GPSInterface]:
        return self._gps

    @gps.setter
    def gps(self, gps: Optional[GPSInterface]):
        self._gps = gps

    @property
    def preselector(self) -> Optional[Preselector]:
        return self._preselector

    @preselector.setter
    def preselector(self, preselector: Optional[Preselector]):
        self._preselector = preselector

    @property
    def switches(self) -> Dict[str, WebRelay]:
        return self._switches

    @switches.setter
    def switches(self, switches: Dict[str, WebRelay]):
        self._switches = switches

    @property
    def location(self) -> Optional[dict]:
        return self._location

    @location.setter
    def location(self, loc: Optional[dict]):
        self._location = loc

    @property
    def capabilities(self) -> Optional[dict]:
        return self._capabilities

    @capabilities.setter
    def capabilities(self, capabilities: Optional[dict]):
        if capabilities is not None:
            if "sensor_sha512" not in capabilities["sensor"]:
                sensor_def = json.dumps(capabilities["sensor"], sort_keys=True)
                SENSOR_DEFINITION_HASH = hashlib.sha512(
                    sensor_def.encode("UTF-8")
                ).hexdigest()
                capabilities["sensor"]["sensor_sha512"] = SENSOR_DEFINITION_HASH
        self._capabilities = capabilities

    @property
    def has_configurable_preselector(self) -> bool:
        if self._capabilities is None:
            return False
        else:
            sensor_definition = self._capabilities["sensor"]
            if (
                "preselector" in sensor_definition
                and "rf_paths" in sensor_definition["preselector"]
            ):
                return True
            else:
                return False

    @property
    def start_time(self) -> datetime.datetime:
        return self._start_time

    @property
    def sensor_calibration(self) -> Optional[SensorCalibration]:
        return self._sensor_calibration

    @sensor_calibration.setter
    def sensor_calibration(self, cal: Optional[SensorCalibration]):
        self._sensor_calibration = cal

    @property
    def differential_calibration(self) -> Optional[DifferentialCalibration]:
        return self._differential_calibration

    @differential_calibration.setter
    def differential_calibration(self, cal: Optional[DifferentialCalibration]):
        self._differential_calibration = cal

    @property
    def last_calibration_time(self) -> str:
        """Get a datetime string for the most recent sensor calibration."""
        return convert_string_to_millisecond_iso_format(
            self.sensor_calibration.last_calibration_datetime
        )

    def _get_calibration_args_from_sigan(self, calibration: Calibration) -> list:
        """Get current values of signal analyzer settings which are calibration parameters."""
        cal_params = [
            p for p in calibration.calibration_parameters if p in SIGAN_SETTINGS_KEYS
        ]
        if set(cal_params) <= set(calibration.calibration_parameters):
            msg = (
                "One or more required calibration parameters is not a valid signal "
                + f"analyzer property.\nRequired parameters: {calibration.calibration_parameters}"
                + f"\nSignal analyzer properties: {list(vars(self.signal_analyzer).keys())}"
            )
            logger.error(msg)
            raise KeyError
        cal_args = [vars(self.signal_analyzer)[f"_{p}"] for p in cal_params]
        logger.debug(f"Matched calibration params: {cal_args}")
        return cal_args  # Order matches calibration.calibration_parameters

    def recompute_differential_calibration_data(self) -> None:
        """Set the differential calibration data based on the current tuning."""
        self.differential_calibration_data = {}
        if self.differential_calibration is not None:
            cal_args = self._get_calibration_args_from_sigan(
                self.differential_calibration
            )
            self.differential_calibration_data.update(
                self.differential_calibration.get_calibration_dict(cal_args)
            )
        else:
            logger.warning("Differential calibration does not exist.")

    def recompute_sensor_calibration_data(self) -> None:
        """Set the sensor calibration data based on the current tuning."""
        self.sensor_calibration_data = {}
        if self.sensor_calibration is not None:
            cal_args = self._get_calibration_args_from_sigan(self.sensor_calibration)
            self.sensor_calibration_data.update(
                self.sensor_calibration.get_calibration_dict(cal_args)
            )
        else:
            logger.warning("Sensor calibration does not exist.")

    def acquire_time_domain_samples(
        self,
        num_samples: int,
        num_samples_skip: int = 0,
        retries: int = 5,
        cal_adjust: bool = True,
    ) -> dict:
        """
        Acquire time-domain IQ samples from the signal analyzer.

        Signal analyzer settings, preselector state, etc. should already be
        set before calling this function.

        Gain adjustment can be applied to acquired samples using ``cal_adjust``.
        If ``True``, the samples acquired from the signal analyzer will be
        scaled based on the calibrated ``gain`` value in the ``SensorCalibration``,
        if one exists for this sensor, and "calibration terminal" will be the value
        of the "reference" key in the returned dict.

        :param num_samples: Number of samples to acquire
        :param num_samples_skip: Number of samples to skip
        :param retries: Maximum number of retries on failure
        :param cal_adjust: If True, use available calibration data to scale the samples.
        :return: dictionary containing data, sample_rate, frequency, capture_time, etc
        :raises Exception: If the sample acquisition fails, or the sensor has
            no signal analyzer.
        """
        logger.debug("Sensor.acquire_time_domain_samples starting")
        logger.debug(f"Number of retries = {retries}")
        max_retries = retries
        # TODO: Include RF path as a sensor cal argument?
        # Acquire samples from signal analyzer
        if self.signal_analyzer is not None:
            while True:
                try:
                    measurement_result = (
                        self.signal_analyzer.acquire_time_domain_samples(
                            num_samples, num_samples_skip
                        )
                    )
                    break
                except Exception as e:
                    retries -= 1
                    logger.info("Error while acquiring samples from signal analyzer.")
                    if retries == 0:
                        logger.exception(
                            "Failed to acquire samples from signal analyzer. "
                            + f"Tried {max_retries} times."
                        )
                        raise e
        else:
            msg = "Failed to acquire samples: sensor has no signal analyzer"
            logger.error(msg)
            raise Exception(msg)

        # Apply gain adjustment based on calibration
        if cal_adjust:
            if self.sensor_calibration is not None:
                logger.debug("Scaling samples using calibration data")
                self.recompute_sensor_calibration_data()
                calibrated_gain__db = self.sensor_calibration_data["gain"]
                calibrated_nf__db = self.sensor_calibration_data["noise_figure"]
                logger.debug(f"Using sensor gain: {calibrated_gain__db} dB")
                if self.differential_calibration is not None:
                    # Also apply differential calibration correction
                    # TODO recompute functions match to current signal analyzer
                    # settings. should they use the measurement_result instead?
                    self.recompute_differential_calibration_data()
                    differential_loss = self.differential_calibration_data["loss"]
                    logger.debug(f"Using differential loss: {differential_loss} dB")
                    calibrated_gain__db -= differential_loss
                    calibrated_nf__db += differential_loss
                    measurement_result[
                        "reference"
                    ] = self.differential_calibration.reference_point
                else:
                    # No differential calibration exists
                    logger.debug("No differential calibration was applied")
                    measurement_result["reference"] = "calibration terminal"

                linear_gain = 10.0 ** (calibrated_gain__db / 20.0)
                logger.debug(f"Applying total gain of {calibrated_gain__db}")
                measurement_result["data"] /= linear_gain

                # Metadata: record the gain and noise figure based on the actual
                # scaling which was used.
                measurement_result["applied_calibration"] = {
                    "gain": calibrated_gain__db,
                    "noise_figure": calibrated_nf__db,
                }
            else:
                # No sensor calibration exists
                msg = "Unable to scale samples without sensor calibration data"
                logger.error(msg)
                raise Exception(msg)
        else:
            # Set the data reference in the measurement_result
            measurement_result["reference"] = "signal analyzer input"
            measurement_result["calibration"] = None

        return measurement_result
