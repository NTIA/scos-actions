import datetime

from scos_actions.hardware.mocks.mock_sensor import (
    MockSensor,
    _mock_capabilities,
    _mock_differential_cal_data,
    _mock_location,
    _mock_sensor_cal_data,
)


def test_mock_sensor():
    sensor = MockSensor()
    assert sensor is not None
    assert sensor.signal_analyzer is not None
    assert sensor.gps is not None
    assert sensor.preselector is None
    assert sensor.switches == {}
    assert sensor.location == _mock_location
    assert sensor.capabilities == _mock_capabilities
    assert sensor.sensor_calibration is None
    assert sensor.differential_calibration is None
    assert sensor.has_configurable_preselector is False
    assert sensor.sensor_calibration_data == _mock_sensor_cal_data
    assert sensor.differential_calibration_data == _mock_differential_cal_data
    assert isinstance(sensor.start_time, datetime.datetime)
