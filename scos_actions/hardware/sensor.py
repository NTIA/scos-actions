import hashlib
import json
import datetime
from typing import Dict

from its_preselector.preselector import Preselector
from its_preselector.web_relay import WebRelay

from .gps_iface import GPSInterface
from .mocks.mock_sigan import MockSignalAnalyzer
from .sigan_iface import SignalAnalyzerInterface


class Sensor:
    def __init__(
        self,
        signal_analyzer: SignalAnalyzerInterface = MockSignalAnalyzer,
        gps: GPSInterface = None,
        preselector: Preselector = None,
        switches: Dict[str, WebRelay] = {},
        location: dict = None,
        capabilities: dict = None,
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
        return self._gps

    @gps.setter
    def gps(self, gps: GPSInterface):
        self._gps = gps

    @property
    def preselector(self) -> Preselector:
        return self._preselector

    @preselector.setter
    def preselector(self, preselector: Preselector):
        self._preselector = preselector

    @property
    def switches(self) -> Dict[str, WebRelay]:
        return self._switches

    @switches.setter
    def switches(self, switches: Dict[str, WebRelay]):
        self._switches = switches

    @property
    def location(self) -> dict:
        return self._location

    @location.setter
    def location(self, loc: dict):
        self._location = loc

    @property
    def capabilities(self) -> dict:
        return self._capabilities

    @capabilities.setter
    def capabilities(self, capabilities: dict):
        if capabilities is not None:
            if "sensor_sha512" not in capabilities["sensor"]:
                sensor_def = json.dumps(capabilities["sensor"], sort_keys=True)
                SENSOR_DEFINITION_HASH = hashlib.sha512(
                    sensor_def.encode("UTF-8")
                ).hexdigest()
                capabilities["sensor"]["sensor_sha512"] = SENSOR_DEFINITION_HASH
            self._capabilities = capabilities
        else:
            self._capabilities = None

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
    def start_time(self):
        return self._start_time
