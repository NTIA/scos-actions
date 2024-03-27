import datetime

import pytest
from its_preselector.controlbyweb_web_relay import ControlByWebWebRelay
from its_preselector.web_relay_preselector import WebRelayPreselector

from scos_actions.hardware.mocks.mock_gps import MockGPS
from scos_actions.hardware.mocks.mock_sensor import (
    MockSensor,
    _mock_capabilities,
    _mock_differential_cal_data,
    _mock_location,
    _mock_sensor_cal_data,
)
from scos_actions.hardware.mocks.mock_sigan import MockSignalAnalyzer


@pytest.fixture
def mock_sensor():
    sensor = MockSensor()
    return sensor


def test_mock_sensor_defaults(mock_sensor):
    assert isinstance(mock_sensor.signal_analyzer, MockSignalAnalyzer)
    assert isinstance(mock_sensor.gps, MockGPS)
    assert mock_sensor.preselector is None
    assert mock_sensor.switches == {}
    assert mock_sensor.location == _mock_location
    assert mock_sensor.capabilities == _mock_capabilities
    assert mock_sensor.sensor_calibration is None
    assert mock_sensor.differential_calibration is None
    assert mock_sensor.has_configurable_preselector is False
    assert mock_sensor.has_configurable_preselector is False
    assert mock_sensor.sensor_calibration_data == _mock_sensor_cal_data
    assert mock_sensor.differential_calibration_data == _mock_differential_cal_data
    assert isinstance(mock_sensor.start_time, datetime.datetime)


def test_set_get_sigan(mock_sensor):
    mock_sigan = MockSignalAnalyzer()
    mock_sensor.signal_analyzer = mock_sigan
    assert mock_sensor.signal_analyzer == mock_sigan


def test_set_get_gps(mock_sensor):
    mock_gps = MockGPS()
    mock_sensor.gps = mock_gps
    assert mock_sensor.gps == mock_gps


def test_set_get_preselector(mock_sensor):
    mock_preselector = WebRelayPreselector(
        {}, {"name": "mock_preselector", "base_url": "url"}
    )
    mock_sensor.preselector = mock_preselector
    assert mock_sensor.preselector == mock_preselector


def test_set_get_switches(mock_sensor):
    mock_switches = {
        "mock": ControlByWebWebRelay({"name": "mock_switch", "base_url": "url"})
    }
    mock_sensor.switches = mock_switches
    assert mock_sensor.switches == mock_switches


def test_set_get_location(mock_sensor):
    mock_location = {"x": 0, "y": 0, "z": 0, "description": "Test"}
    mock_sensor.location = mock_location
    assert mock_sensor.location == mock_location


def test_set_get_capabilities(mock_sensor):
    mock_capabilities = {"fake": "capabilities"}
    mock_sensor.capabilities = mock_capabilities
    assert mock_sensor.capabilities == mock_capabilities


def test_set_get_sensor_calibration(mock_sensor):
    assert mock_sensor.sensor_calibration is None


def test_set_get_differential_calibration(mock_sensor):
    assert mock_sensor.differential_calibration is None


def test_has_configurable_preselector_in_capabilities(mock_sensor):
    capabilities = {
        "sensor": {
            "preselector": {"rf_paths": [{"name": "antenna"}, {"name": "noise_diode"}]}
        }
    }
    mock_sensor.capabilities = capabilities
    assert mock_sensor.has_configurable_preselector == True


def test_has_configurable_preselector_in_preselector(mock_sensor):
    mock_sensor.preselector = WebRelayPreselector(
        {}, {"name": "preselector", "base_url": "url"}
    )
    mock_sensor.preselector.rf_paths = [{"name": "antenna"}, {"name": "noise_diode"}]
    assert mock_sensor.has_configurable_preselector == True


def test_has_configurable_preselector_not_configurable(mock_sensor):
    capabilities = {"sensor": {"preselector": {"rf_paths": [{"name": "antenna"}]}}}
    mock_sensor.capabilities = capabilities
    assert mock_sensor.has_configurable_preselector == False


def test_hash_set_when_not_present(mock_sensor):
    capabilities = {"sensor": {"preselector": {"rf_paths": [{"name": "antenna"}]}}}
    mock_sensor.capabilities = capabilities
    assert "sensor_sha512" in mock_sensor.capabilities["sensor"]
    assert mock_sensor.capabilities["sensor"]["sensor_sha512"] is not None


def test_hash_not_overwritten(mock_sensor):
    capabilities = {"sensor": {"sensor_sha512": "some hash"}}
    mock_sensor.capabilities = capabilities
    assert mock_sensor.capabilities["sensor"]["sensor_sha512"] == "some hash"
