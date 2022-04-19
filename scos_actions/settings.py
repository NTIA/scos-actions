from os import path
from django.conf import settings
from scos_actions import calibration
from scos_actions.tests.resources.utils import create_dummy_calibration
import logging

logger = logging.getLogger(__name__)

def get_sigan_calibration(sigan_cal_file):
    if not RUNNING_TESTS and not MOCK_SIGAN:
        try:
            sigan_cal = calibration.load_from_json(sigan_cal_file)
        except Exception as err:
            logger.error("Unable to load sigan calibration data, reverting to none")
            logger.exception(err)
            sigan_cal = None

    else:  # If in testing, create our own test files
        dummy_calibration = create_dummy_calibration()
        sigan_cal = dummy_calibration

    return sigan_cal


def get_sensor_calibration(sensor_cal_file):
    """Get calibration data from sensor_cal_file and sigan_cal_file."""
    # Try and load sensor/sigan calibration data
    if not RUNNING_TESTS and not MOCK_SIGAN:
        try:
            sensor_cal = calibration.load_from_json(sensor_cal_file)
        except Exception as err:
            logger.error(
                "Unable to load sensor calibration data, reverting to none"
            )
            logger.exception(err)
            sensor_cal = None

    else:  # If in testing, create our own test files
        dummy_calibration = create_dummy_calibration()
        sensor_cal = dummy_calibration
    return sensor_cal


CONFIG_DIR = path.join(
    path.dirname(path.abspath(__file__)), "configs"
)

ACTION_DEFINITIONS_DIR = path.join(
    path.dirname(path.abspath(__file__)), "configs/actions"
)

#set sigan_calibration file and sensor_calibration_file
if not settings.configured or not hasattr(settings, "SIGAN_CALIBRATION_FILE"):
    SIGAN_CALIBRATION_FILE = path.join(CONFIG_DIR, "sigan_calibration.json.example")
else:
    SIGAN_CALIBRATION_FILE = settings.SIGAN_CALIBRATION_FILE
if not settings.configured or not hasattr(settings, "SENSOR_CALIBRATION_FILE"):
    SENSOR_CALIBRATION_FILE = path.join(CONFIG_DIR, "sensor_calibration.json.example")
else:
    SENSOR_CALIBRATION_FILE = settings.SENSOR_CALIBRATION_FILE

if not settings.configured:
    PRESELECTOR_CONFIG_FILE = None
    SENSOR_DEFINITION_FILE = None
    FQDN = None
    PRESELECTOR_MODULE = 'its_preselector.web_relay_preselector'
    PRESELECTOR_CLASS = 'WebRelayPreselector'
    sensor_calibration = create_dummy_calibration()
    sigan_calibration = create_dummy_calibration()
else:
    MOCK_SIGAN = settings.MOCK_SIGAN
    RUNNING_TESTS = settings.RUNNING_TESTS
    SENSOR_DEFINITION_FILE = settings.SENSOR_DEFINITION_FILE
    FQDN = settings.FQDN
    if settings.PRESELECTOR_CONFIG:
        PRESELECTOR_CONFIG_FILE = settings.PRESELECTOR_CONFIG
    else:
        PRESELECTOR_CONFIG_FILE = None
        
    if settings.PRESELECTOR_MODULE and settings.PRESELECTOR_CLASS:
        PRESELECTOR_MODULE = settings.PRESELECTOR_MODULE
        PRESELECTOR_CLASS = settings.PRESELECTOR_CLASS
    else:
        PRESELECTOR_MODULE = 'its_preselector.web_relay_preselector'
        PRESELECTOR_CLASS = 'WebRelayPreselector'
    sensor_calibration = get_sensor_calibration(SENSOR_CALIBRATION_FILE)
    sigan_calibration = get_sigan_calibration(SIGAN_CALIBRATION_FILE)