import logging
from os import path
from pathlib import Path

from django.conf import settings
from environs import Env

logger = logging.getLogger(__name__)
env = Env()

logger.debug("Initializing scos-actions settings")
CONFIG_DIR = Path(__file__).parent.resolve() / "configs"
ACTION_DEFINITIONS_DIR = CONFIG_DIR / "actions"

if not settings.configured or not hasattr(settings, "DEFAULT_CALIBRATION_FILE"):
    DEFAULT_CALIBRATION_FILE = path.join(CONFIG_DIR, "default_calibration.json")
else:
    DEFAULT_CALIBRATION_FILE = settings.DEFAULT_CALIBRATION_FILE

# set sigan_calibration file and sensor_calibration_file
if not settings.configured or not hasattr(settings, "SIGAN_CALIBRATION_FILE"):
    logger.warning("Sigan calibration file is not defined.")
    SIGAN_CALIBRATION_FILE = ""
    sigan_calibration = None
else:
    SIGAN_CALIBRATION_FILE = settings.SIGAN_CALIBRATION_FILE
    logger.debug(f"SCOS_ACTIONS: SIGAN_CALIBRATION_FILE: {SIGAN_CALIBRATION_FILE}")

if not settings.configured or not hasattr(settings, "SENSOR_CALIBRATION_FILE"):
    logger.warning(
        f"Sensor calibration file is not defined. Settings configured: {settings.configured}"
    )
    SENSOR_CALIBRATION_FILE = ""
    sensor_calibration = None
else:
    SENSOR_CALIBRATION_FILE = settings.SENSOR_CALIBRATION_FILE
    logger.debug(f"SCOS_ACTIONS: SENSOR_CALIBRATION_FILE: {SENSOR_CALIBRATION_FILE}")

SWITCH_CONFIGS_DIR = env("SWITCH_CONFIGS_DIR", default=None)
SCOS_SENSOR_GIT_TAG = env("SCOS_SENSOR_GIT_TAG", default="unknown")
if not settings.configured:
    PRESELECTOR_CONFIG_FILE = None
    SENSOR_DEFINITION_FILE = None
    FQDN = None
    PRESELECTOR_MODULE = env("PRESELECTOR_MODULE", default=None)
    PRESELECTOR_CLASS = env("PRESELECTOR_CLASS", default=None)
    SIGAN_POWER_CYCLE_STATES = env("SIGAN_POWER_CYCLE_STATES", default=None)
    SIGAN_POWER_SWITCH = env("SIGAN_POWER_SWITCH", default=None)
    MOCK_SIGAN = env("MOCK_SIGAN", default=None)

else:
    MOCK_SIGAN = settings.MOCK_SIGAN
    RUNNING_TESTS = settings.RUNNING_TESTS
    SENSOR_DEFINITION_FILE = Path(settings.SENSOR_DEFINITION_FILE)
    FQDN = settings.FQDN
    SIGAN_POWER_SWITCH = None
    SIGAN_POWER_CYCLE_STATES = None
    if hasattr(settings, "SIGAN_POWER_SWITCH"):
        SIGAN_POWER_SWITCH = settings.SIGAN_POWER_SWITCH
    if hasattr(settings, "SIGAN_POWER_CYCLE_STATES"):
        SIGAN_POWER_CYCLE_STATES = settings.SIGAN_POWER_CYCLE_STATES
