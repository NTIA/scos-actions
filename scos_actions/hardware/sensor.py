import datetime
import hashlib
import json
import logging
from typing import Any, Dict, List, Optional

import numpy as np
from its_preselector.preselector import Preselector
from its_preselector.web_relay import WebRelay

from scos_actions.calibration.differential_calibration import DifferentialCalibration
from scos_actions.calibration.interfaces.calibration import Calibration
from scos_actions.calibration.sensor_calibration import SensorCalibration
from scos_actions.hardware.gps_iface import GPSInterface
from scos_actions.hardware.sigan_iface import SignalAnalyzerInterface
from scos_actions.utils import convert_string_to_millisecond_iso_format

logger = logging.getLogger(__name__)


class Sensor:
    """
    Software representation of the physical RF sensor. The Sensor may include a GPSInterface,
    Preselector, a dictionary of WebRelays, a location specified in GeoJSON, and a dictionary
    of the sensor capabilities. The capabilities should include a 'sensor' key that maps to
    the metadata definition of the Sensor(
    https://github.com/NTIA/sigmf-ns-ntia/blob/master/ntia-sensor.sigmf-ext.md#01-the-sensor-object),
    and an 'action' key that maps to a list of ntia-scos action objects
    (https://github.com/NTIA/sigmf-ns-ntia/blob/master/ntia-scos.sigmf-ext.md#02-the-action-object)
    The Sensor instance is passed into Actions __call__ methods to perform an action.
    """

    logger = logging.getLogger(__name__)

    def __init__(
        self,
        signal_analyzer: SignalAnalyzerInterface,
        capabilities: dict,
        gps: Optional[GPSInterface] = None,
        preselector: Optional[Preselector] = None,
        switches: Optional[Dict[str, WebRelay]] = {},
        location: Optional[dict] = None,
        sensor_cal: Optional[SensorCalibration] = None,
        differential_cal: Optional[DifferentialCalibration] = None,
    ):
        self.signal_analyzer = signal_analyzer
        self.gps = gps
        self.preselector = preselector
        self.switches = switches
        self.location = location
        self.capabilities = capabilities
        self._sensor_calibration_data = {}
        self._sensor_calibration = sensor_cal
        self._differential_calibration_data = {}
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
    def gps(self) -> GPSInterface:
        """
        The sensor's Global Positioning System.
        """
        return self._gps

    @gps.setter
    def gps(self, gps: GPSInterface):
        """
        Set the sensor's Global Positioning System.
        """
        self._gps = gps

    @property
    def preselector(self) -> Optional[Preselector]:
        """
        RF front end that may include calibration sources, filters, and/or amplifiers.
        """
        return self._preselector

    @preselector.setter
    def preselector(self, preselector: Preselector):
        """
        Set the RF front end that may include calibration sources, filters, and/or amplifiers.
        """
        self._preselector = preselector

    @property
    def switches(self) -> Dict[str, WebRelay]:
        """
        Dictionary of WebRelays, indexed by name. WebRelays may enable/disable other
        components within the sensor and/or provide a variety of sensors.
        """
        return self._switches

    @switches.setter
    def switches(self, switches: Dict[str, WebRelay]):
        self._switches = switches

    @property
    def location(self) -> Optional[dict]:
        """
        The GeoJSON dictionary of the sensor's location.
        """
        return self._location

    @location.setter
    def location(self, loc: dict):
        """
        Set the GeoJSON location of the sensor.
        """
        self._location = loc

    @property
    def capabilities(self) -> Optional[dict]:
        """
        A dictionary of the sensor's capabilities. The dictionary should
        include a 'sensor' key that maps to the ntia-sensor
        (https://github.com/NTIA/sigmf-ns-ntia/blob/master/ntia-sensor.sigmf-ext.md)
        object and an actions key that maps to a list of ntia-scos action objects
        (https://github.com/NTIA/sigmf-ns-ntia/blob/master/ntia-scos.sigmf-ext.md#02-the-action-object)
        """
        return self._capabilities

    @capabilities.setter
    def capabilities(self, capabilities: dict):
        """
        Set the dictionary of the sensor's capabilities. The dictionary should
        include a 'sensor' key that links to the ntia-sensor
        (https://github.com/NTIA/sigmf-ns-ntia/blob/master/ntia-sensor.sigmf-ext.md)
        object and an actions key that links to a list of ntia-scos action objects
        (https://github.com/NTIA/sigmf-ns-ntia/blob/master/ntia-scos.sigmf-ext.md#02-the-action-object)
        """
        if capabilities is not None:
            if (
                "sensor" in capabilities
                and "sensor_sha512" not in capabilities["sensor"]
            ):
                sensor_def = json.dumps(capabilities["sensor"], sort_keys=True)
                sensor_definition_hash = hashlib.sha512(
                    sensor_def.encode("UTF-8")
                ).hexdigest()
                capabilities["sensor"]["sensor_sha512"] = sensor_definition_hash
        self._capabilities = capabilities

    @property
    def has_configurable_preselector(self) -> bool:
        """
        Checks if the preselector has multiple rf paths.
        Returns: True if either the Preselector object or the sensor definition contain multiple rf_paths, False
        otherwise.
        """
        if (
            self.preselector is not None
            and self.preselector.rf_paths is not None
            and len(self.preselector.rf_paths) > 0
        ):
            self.logger.debug(
                "Preselector is configurable: found multiple rf_paths in preselector object."
            )
            return True
        elif (
            self.capabilities
            and len(
                self.capabilities.get("sensor", {})
                .get("preselector", {})
                .get("rf_paths", [])
            )
            > 1
        ):
            self.logger.debug(
                "Preselector is configurable: found multiple rf_paths in sensor definition."
            )
            return True
        else:
            self.logger.debug(
                "Preselector is not configurable: Neither sensor definition or preselector object contained multiple rf_paths."
            )
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
        """A datetime string for the most recent sensor calibration."""
        return convert_string_to_millisecond_iso_format(
            self.sensor_calibration.last_calibration_datetime
        )

    @property
    def sensor_calibration_data(self) -> Dict[str, Any]:
        """Sensor calibration data for the current sensor settings."""
        return self._sensor_calibration_data

    @property
    def differential_calibration_data(self) -> Dict[str, float]:
        """Differential calibration data for the current sensor settings."""
        return self._differential_calibration_data

    def recompute_calibration_data(self, params: dict) -> None:
        """
        Set the differential_calibration_data and sensor_calibration_data
        based on the specified ``params``.
        """
        recomputed = False
        if self.differential_calibration is not None:
            self._differential_calibration_data.update(
                self.differential_calibration.get_calibration_dict(params)
            )
            recomputed = True
        else:
            logger.debug("No differential calibration available to recompute")

        if self.sensor_calibration is not None:
            self._sensor_calibration_data.update(
                self.sensor_calibration.get_calibration_dict(params)
            )
            recomputed = True
        else:
            logger.debug("No sensor calibration available to recompute")

        if not recomputed:
            logger.warning("Failed to recompute calibration data")

    def check_sensor_overload(self, data) -> bool:
        """Check for sensor overload in the measurement data."""
        measured_data = data.astype(np.complex64)

        time_domain_avg_power = 10 * np.log10(np.mean(np.abs(measured_data) ** 2))
        time_domain_avg_power += (
            10 * np.log10(1 / (2 * 50)) + 30
        )  # Convert log(V^2) to dBm
        # explicitly check is not None since 1db compression could be 0
        if self.sensor_calibration_data["compression_point"] is not None:
            return bool(
                time_domain_avg_power
                > self.sensor_calibration_data["compression_point"]
            )
        else:
            logger.debug(
                "Compression point is None, returning False for sensor overload."
            )
            return False

    def acquire_time_domain_samples(
        self,
        num_samples: int,
        num_samples_skip: int = 0,
        retries: int = 5,
        cal_adjust: bool = True,
        cal_params: Optional[dict] = None,
    ) -> dict:
        """
        Acquire time-domain IQ samples from the signal analyzer.

        Signal analyzer settings, preselector state, etc. should already be
        set before calling this function.

        Gain adjustment can be applied to acquired samples using ``cal_adjust``.
        If ``True``, the samples acquired from the signal analyzer will be
        scaled based on the calibrated ``gain`` and ``loss`` values in
        the ``SensorCalibration`` and ``DifferentialCalibration.``
        If no ``DifferentialCalibration`` exists, "calibration terminal"
        will be the value of the "reference" key in the
        returned dict. If a ``DifferentialCalibration`` exists, the gain and
        noise figure will be adjusted with the loss specified in the
        ``DifferentialCalibration`` and the "reference" will be set to the
        calibration_reference of the ``DifferentialCalibration``.

        :param num_samples: Number of samples to acquire
        :param num_samples_skip: Number of samples to skip
        :param retries: Maximum number of retries on failure
        :param cal_adjust: If True, use available calibration data to scale the samples.
        :param cal_params: A dictionary with keys for all of the calibration parameters.
            May contain additional keys. Example: ``{"sample_rate": 14000000.0, "gain": 10.0}``
            Must be specified if ``cal_adjust`` is ``True``. Otherwise, ignored.
        :return: dictionary containing data, sample_rate, frequency, capture_time, etc
        :raises Exception: If the sample acquisition fails, or the sensor has
            no signal analyzer.
        """
        logger.debug("***********************************\n")
        logger.debug("Sensor.acquire_time_domain_samples starting")
        logger.debug(f"Number of retries = {retries}")
        logger.debug("*************************************\n")

        max_retries = retries
        sensor_overload = False
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
                except BaseException as e:
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
            if cal_params is None:
                raise ValueError(
                    "Data scaling cannot occur without specified calibration parameters."
                )
            if self.sensor_calibration is not None:
                logger.debug("Scaling samples. Fetching calibration data.")
                self.recompute_calibration_data(cal_params)
                if self.differential_calibration is not None:
                    logger.debug(
                        f"USING DIFF. CAL: {self.differential_calibration.calibration_data}"
                    )
                if self.sensor_calibration is not None:
                    logger.debug(
                        f"USING SENSOR CAL: {self.sensor_calibration.calibration_data}"
                    )
                calibrated_gain__db = self.sensor_calibration_data["gain"]
                calibrated_nf__db = self.sensor_calibration_data["noise_figure"]
                logger.debug(f"Using sensor gain: {calibrated_gain__db} dB")
                measurement_result["reference"] = (
                    self.sensor_calibration.calibration_reference
                )
                if self.differential_calibration is not None:
                    # Also apply differential calibration correction
                    differential_loss = self.differential_calibration_data["loss"]
                    logger.debug(f"Using differential loss: {differential_loss} dB")
                    calibrated_gain__db -= differential_loss
                    calibrated_nf__db += differential_loss
                    measurement_result["reference"] = (
                        self.differential_calibration.calibration_reference
                    )

                else:
                    # No differential calibration exists
                    logger.debug("No differential calibration was applied")

                linear_gain = 10.0 ** (calibrated_gain__db / 20.0)
                logger.debug(f"Applying total gain of {calibrated_gain__db}")
                measurement_result["data"] /= linear_gain

                # Metadata: record the gain and noise figure based on the actual
                # scaling which was used.
                measurement_result["applied_calibration"] = {
                    "gain": calibrated_gain__db,
                    "noise_figure": calibrated_nf__db,
                }
                if "compression_point" in self.sensor_calibration_data:
                    measurement_result["applied_calibration"]["compression_point"] = (
                        self.sensor_calibration_data["compression_point"]
                    )
                    sensor_overload = self.check_sensor_overload(
                        measurement_result["data"]
                    )
                    if sensor_overload:
                        logger.warning("Sensor overload occurred!")
                    # measurement_result["overload"] could be true based on sigan overload or sensor overload
                    measurement_result["overload"] = (
                        measurement_result["overload"] or sensor_overload
                    )
                applied_cal = measurement_result["applied_calibration"]
                logger.debug(f"Setting applied_calibration to: {applied_cal}")
            else:
                # No sensor calibration exists
                msg = "Unable to scale samples without sensor calibration data"
                logger.error(msg)
                raise Exception(msg)
        else:
            # Set the data reference in the measurement_result
            measurement_result["reference"] = "signal analyzer input"
            measurement_result["applied_calibration"] = None

        return measurement_result
