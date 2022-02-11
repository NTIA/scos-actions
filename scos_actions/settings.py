from os import path
from django.conf import settings

ACTION_DEFINITIONS_DIR = path.join(
    path.dirname(path.abspath(__file__)), "configs/actions"
)
PRESELECTOR_CONFIG_FILE =  path.join(
    path.dirname(path.abspath(__file__)), "configs/preselector_config.json"
)
SENSOR_DEFINITION_FILE = settings.SIGAN_CALIBRATION_FILE
FQDN = settings.FQDN
