from its_preselector.controlbyweb_web_relay import ControlByWebWebRelay
from its_preselector.web_relay_preselector import WebRelayPreselector

from scos_actions.hardware.mocks.mock_gps import MockGPS
from scos_actions.hardware.mocks.mock_sigan import MockSignalAnalyzer
from scos_actions.hardware.sensor import Sensor


def test_sensor():
    sigan = MockSignalAnalyzer()
    sensor = Sensor(signal_analyzer=sigan, capabilities={}, gps=MockGPS(sigan))
    assert sensor is not None
    assert sensor.signal_analyzer is not None
    assert sensor.gps is not None


def test_set_get_preselector():
    preselector = WebRelayPreselector({}, {"name": "preselector", "base_url": "url"})
    sensor = Sensor(signal_analyzer=MockSignalAnalyzer(), capabilities={})
    sensor.preselector = preselector
    assert sensor.preselector == preselector


def test_set_get_gps():
    sigan = MockSignalAnalyzer()
    gps = MockGPS()
    sensor = Sensor(signal_analyzer=sigan, capabilities={})
    sensor.gps = gps
    assert sensor.gps == gps


def test_set_get_switches():
    switches = {"spu": ControlByWebWebRelay({"name": "spu", "base_url": "url"})}
    sensor = Sensor(signal_analyzer=MockSignalAnalyzer(), capabilities={})
    sensor.switches = switches
    assert sensor.switches == switches


def test_has_configurable_preselector_in_capabilities():
    capabilities = {
        "sensor": {
            "preselector": {"rf_paths": [{"name": "antenna"}, {"name": "noise_diode"}]}
        }
    }
    sensor = Sensor(signal_analyzer=MockSignalAnalyzer(), capabilities=capabilities)
    assert sensor.has_configurable_preselector == True


def test_has_configurable_preselector_in_preselector():
    sensor = Sensor(signal_analyzer=MockSignalAnalyzer(), capabilities={})
    sensor.preselector = WebRelayPreselector(
        {}, {"name": "preselector", "base_url": "url"}
    )
    sensor.preselector.rf_paths = [{"name": "antenna"}, {"name": "noise_diode"}]
    assert sensor.has_configurable_preselector == True


def test_has_configurable_preselector_not_configurable():
    capabilities = {"sensor": {"preselector": {"rf_paths": [{"name": "antenna"}]}}}
    sensor = Sensor(signal_analyzer=MockSignalAnalyzer(), capabilities=capabilities)
    assert sensor.has_configurable_preselector == False


def test_hash_set_when_not_present():
    capabilities = {"sensor": {"preselector": {"rf_paths": [{"name": "antenna"}]}}}
    sensor = Sensor(signal_analyzer=MockSignalAnalyzer(), capabilities=capabilities)
    assert "sensor_sha512" in sensor.capabilities["sensor"]
    assert sensor.capabilities["sensor"]["sensor_sha512"] is not None


def test_hash_not_overwritten():
    capabilities = {"sensor": {"sensor_sha512": "some hash"}}
    sensor = Sensor(signal_analyzer=MockSignalAnalyzer(), capabilities=capabilities)
    assert sensor.capabilities["sensor"]["sensor_sha512"] == "some hash"
