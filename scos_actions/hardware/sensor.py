import hashlib

from .mocks.mock_gps import MockGPS
from .mocks.mock_sigan import MockSignalAnalyzer
from .sigan_iface import SignalAnalyzerInterface


class Sensor:
    def __init__(
        self,
        signal_analyzer=MockSignalAnalyzer,
        gps=MockGPS(),
        preselector=None,
        switches={},
        location=None,
        capabilities=None,
    ):
        self._signal_analyzer = signal_analyzer
        self._gps = gps
        self._preselector = preselector
        self._switches = switches
        self._location = location
        self.capabilities = capabilities

    @property
    def signal_analyzer(self):
        return self._signal_analyzer

    @signal_analyzer.setter
    def signal_analyzer(self, sigan):
        self._signal_analyzer = sigan

    @property
    def gps(self):
        return self._gps

    @gps.setter
    def gps(self, gps):
        self._gps = gps

    @property
    def preselector(self):
        return self._preselector

    @preselector.setter
    def preselector(self, preselector):
        self._preselector = preselector

    @property
    def switches(self):
        return self._switches

    @switches.setter
    def switches(self, switches):
        self._switches = switches

    @property
    def location(self):
        return self._location

    @location.setter
    def location(self, loc):
        self._location = loc

    @property
    def capabilities(self):
        return self._capabilities

    @capabilities.setter
    def capabilities(self, capabilities):
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
    def has_configurable_preselector(self):
        if self._capabilities is None:
            return False
        else:
            sensor_definition = self._capabilities["sensor"]
            if (
                "preselector" in self.sensor_definition
                and "rf_paths" in self.sensor_definition["preselector"]
            ):
                return True
            else:
                return False
