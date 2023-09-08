import logging
from pathlib import Path

from django.conf import settings
from environs import Env

logger = logging.getLogger(__name__)
env = Env()

logger.info("Initializing scos-actions settings")
CONFIG_DIR = Path(__file__).parent.resolve() / "configs"
ACTION_DEFINITIONS_DIR = CONFIG_DIR / "actions"

# set sigan_calibration file and sensor_calibration_file
if not settings.configured or not hasattr(settings, "SIGAN_CALIBRATION_FILE"):
    logger.warning("Using default sigan cal file.")
    SIGAN_CALIBRATION_FILE = CONFIG_DIR / "sigan_calibration_example.json"
    sigan_calibration = None
else:
    SIGAN_CALIBRATION_FILE = Path(settings.SIGAN_CALIBRATION_FILE)
    logger.debug(f"SCOS_ACTIONS: SIGAN_CALIBRATION_FILE: {SIGAN_CALIBRATION_FILE}")

if not settings.configured or not hasattr(settings, "SENSOR_CALIBRATION_FILE"):
    logger.warning("Using default sensor cal file.")
    SENSOR_CALIBRATION_FILE = CONFIG_DIR / "sensor_calibration_example.json"
    sensor_calibration = None
else:
    SENSOR_CALIBRATION_FILE = Path(settings.SENSOR_CALIBRATION_FILE)
    logger.debug(f"SCOS_ACTIONS: SENSOR_CALIBRATION_FILE: {SENSOR_CALIBRATION_FILE}")

SWITCH_CONFIGS_DIR = env("SWITCH_CONFIGS_DIR", default=None)
if not settings.configured:
    PRESELECTOR_CONFIG_FILE = None
    SENSOR_DEFINITION_FILE = None
    FQDN = None
    PRESELECTOR_MODULE = env("PRESELECTOR_MODULE", default=None)
    PRESELECTOR_CLASS = env("PRESELECTOR_CLASS", default=None)
    SIGAN_POWER_CYCLE_STATES = env("SIGAN_POWER_CYCLE_STATES", default=None)
    SIGAN_POWER_SWITCH = env("SIGAN_POWER_SWITCH", default=None)
    MOCK_SIGAN = env("MOCK_SIGAN", default=None)
    SCOS_SENSOR_GIT_TAG = env("SCOS_SENSOR_GIT_TAG", default="unknown")
else:
    MOCK_SIGAN = settings.MOCK_SIGAN
    RUNNING_TESTS = settings.RUNNING_TESTS
    SENSOR_DEFINITION_FILE = Path(settings.SENSOR_DEFINITION_FILE)
    FQDN = settings.FQDN
    SCOS_SENSOR_GIT_TAG = settings.SCOS_SENSOR_GIT_TAG
    if settings.PRESELECTOR_CONFIG:
        PRESELECTOR_CONFIG_FILE = settings.PRESELECTOR_CONFIG
    else:
        PRESELECTOR_CONFIG_FILE = None

    if settings.PRESELECTOR_MODULE and settings.PRESELECTOR_CLASS:
        PRESELECTOR_MODULE = settings.PRESELECTOR_MODULE
        PRESELECTOR_CLASS = settings.PRESELECTOR_CLASS
    else:
        PRESELECTOR_MODULE = "its_preselector.web_relay_preselector"
        PRESELECTOR_CLASS = "WebRelayPreselector"
    if hasattr(settings, "SWITCH_CONFIGS_DIR"):
        SWITCH_CONFIGS_DIR = Path(settings.SWITCH_CONFIGS_DIR)

    SIGAN_POWER_SWITCH = None
    SIGAN_POWER_CYCLE_STATES = None
    if hasattr(settings, "SIGAN_POWER_SWITCH"):
        SIGAN_POWER_SWITCH = settings.SIGAN_POWER_SWITCH
    if hasattr(settings, "SIGAN_POWER_CYCLE_STATES"):
        SIGAN_POWER_CYCLE_STATES = settings.SIGAN_POWER_CYCLE_STATES
