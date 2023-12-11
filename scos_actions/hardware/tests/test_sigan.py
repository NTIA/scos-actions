from environs import Env

from scos_actions.hardware.mocks.mock_sigan import MockSignalAnalyzer


def test_sigan_default_cal():
    sigan = MockSignalAnalyzer()
    sigan.recompute_sensor_calibration_data([])
    sensor_cal = sigan.sensor_calibration_data
    assert sensor_cal["gain"] == 0