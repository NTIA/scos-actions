import logging

from scos_actions.hardware.mocks.mock_gps import MockGPS
from scos_actions.hardware.mocks.mock_sigan import MockSignalAnalyzer
from scos_actions.hardware.sensor import Sensor
from scos_actions.utils import get_datetime_str_now

_mock_sensor_cal_data = {
    "datetime": get_datetime_str_now(),
    "gain": 0,
    "enbw": None,
    "noise_figure": None,
    "1db_compression_point": None,
    "temperature": 26.85,
}

_mock_differential_cal_data = {"loss": 10.0}

_mock_capabilities = {"sensor": {}}

_mock_location = {"x": -999, "y": -999, "z": -999, "description": "Testing"}

logger = logging.getLogger(__name__)


class MockSensor(Sensor):
    def __init__(
        self,
        signal_analyzer=MockSignalAnalyzer(),
        gps=MockGPS(),
        preselector=None,
        switches={},
        location=_mock_location,
        capabilities=_mock_capabilities,
        sensor_cal=None,
        differential_cal=None,
    ):
        if (sensor_cal is not None) or (differential_cal is not None):
            logger.warning(
                "Calibration object provided to mock sensor will not be used to query calibration data."
            )
        super().__init__(
            signal_analyzer=signal_analyzer,
            gps=gps,
            preselector=preselector,
            switches=switches,
            location=location,
            capabilities=capabilities,
            sensor_cal=sensor_cal,
            differential_cal=differential_cal,
        )

    @property
    def sensor_calibration_data(self):
        return _mock_sensor_cal_data

    @property
    def differential_calibration_data(self):
        return _mock_differential_cal_data
