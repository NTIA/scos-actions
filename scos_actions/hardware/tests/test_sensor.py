from scos_actions.hardware.mocks.mock_sigan import MockSignalAnalyzer
from scos_actions.hardware.mocks.mock_gps import MockGPS
from scos_actions.hardware.sensor import Sensor


def test_sensor():
    sensor = Sensor(signal_analyzer=MockSignalAnalyzer(), gps=MockGPS())
    assert sensor is not None
    assert sensor.signal_analyzer is not None
    assert sensor.gps is not None
