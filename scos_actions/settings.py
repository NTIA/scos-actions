from os import path
from django.conf import settings
from scos_actions import calibration

import logging

logger = logging.getLogger(__name__)


def get_sigan_calibration(sigan_cal_file):
    try:
        sigan_cal = calibration.load_from_json(sigan_cal_file)
    except Exception as err:
        logger.error("Unable to load sigan calibration data, reverting to none")
        logger.exception(err)
        sigan_cal = None

    return sigan_cal


def get_sensor_calibration(sensor_cal_file):
    """Get calibration data from sensor_cal_file and sigan_cal_file."""
    try:
        sensor_cal = calibration.load_from_json(sensor_cal_file)
    except Exception as err:
        logger.error(
            "Unable to load sensor calibration data, reverting to none"
        )
        logger.exception(err)
        sensor_cal = None
    return sensor_cal


logger.info('Initializing scos-actions settings')
CONFIG_DIR = path.join(
    path.dirname(path.abspath(__file__)), "configs"
)

ACTION_DEFINITIONS_DIR = path.join(
    path.dirname(path.abspath(__file__)), "configs/actions"
)

# set sigan_calibration file and sensor_calibration_file
if not settings.configured or not hasattr(settings, "SIGAN_CALIBRATION_FILE"):
    logger.warning("Using default sigan cal file.")
    SIGAN_CALIBRATION_FILE = path.join(CONFIG_DIR, 'sigan_calibration_example.json')
    sigan_calibration = None
else:
    SIGAN_CALIBRATION_FILE = settings.SIGAN_CALIBRATION_FILE
if not settings.configured or not hasattr(settings, "SENSOR_CALIBRATION_FILE"):
    logger.warning('Using default sensor cal file.')
    SENSOR_CALIBRATION_FILE = path.join(CONFIG_DIR, 'sensor_calibration_example.json')
    sensor_calibration = None
else:
    SENSOR_CALIBRATION_FILE = settings.SENSOR_CALIBRATION_FILE
    logger.debug('SCOS_ACTIONS: SENSOR_CALIBRATION_FILE: ' + SENSOR_CALIBRATION_FILE)

SWITCH_CONFIGS_DIR = path.join(CONFIG_DIR, 'switches')
if not settings.configured:
    PRESELECTOR_CONFIG_FILE = None
    SENSOR_DEFINITION_FILE = None
    FQDN = None
    PRESELECTOR_MODULE = 'its_preselector.web_relay_preselector'
    PRESELECTOR_CLASS = 'WebRelayPreselector'
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
    if settings.SWITCH_CONFIGS_DIR:
        SWITCH_CONFIGS_DIR = settings.SWITCH_CONFIGS_DIR

logger.info('Loading sensor cal file: ' + SENSOR_CALIBRATION_FILE)
sensor_calibration = get_sensor_calibration(SENSOR_CALIBRATION_FILE)
logger.info('Loading sigan cal file: ' + SIGAN_CALIBRATION_FILE)
sigan_calibration = get_sigan_calibration(SIGAN_CALIBRATION_FILE)
if sensor_calibration:
    logger.info("last sensor cal: " + sensor_calibration.calibration_datetime)
