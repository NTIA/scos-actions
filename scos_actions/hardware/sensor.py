import datetime
import hashlib
import json
import logging
from typing import Dict, Optional

from its_preselector.preselector import Preselector
from its_preselector.web_relay import WebRelay

from .gps_iface import GPSInterface
from .sigan_iface import SignalAnalyzerInterface


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
    ):
        self.signal_analyzer = signal_analyzer
        self.gps = gps
        self.preselector = preselector
        self.switches = switches
        self.location = location
        self.capabilities = capabilities
        # There is no setter for start_time property
        self._start_time = datetime.datetime.utcnow()

    @property
    def signal_analyzer(self) -> SignalAnalyzerInterface:
        return self._signal_analyzer

    @signal_analyzer.setter
    def signal_analyzer(self, sigan: SignalAnalyzerInterface):
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
    def preselector(self) -> Preselector:
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
    def location(self) -> dict:
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
    def capabilities(self) -> dict:
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
    def start_time(self):
        return self._start_time
