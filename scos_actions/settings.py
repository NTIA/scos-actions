from os import path
from django.conf import settings

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