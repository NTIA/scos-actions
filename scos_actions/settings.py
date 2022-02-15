from os import path
from django.conf import settings

ACTION_DEFINITIONS_DIR = path.join(
    path.dirname(path.abspath(__file__)), "configs/actions"
)
if not settings.configured:
    PRESELECTOR_CONFIG_FILE = None
    SENSOR_DEFINITION_FILE = None
    FQDN = None
    PRESELECTOR_MODULE = 'its_preselector.web_relay_preselector'
    PRESELECTOR_CLASS = 'WebRelayPreselector'
else:
    PRESELECTOR_CONFIG_FILE = settings.PRESELECTOR_CONFIG
    SENSOR_DEFINITION_FILE = settings.SENSOR_DEFINITION_FILE
    FQDN = settings.FQDN
    PRESELECTOR_MODULE = settings.PRESELECTOR_MODULE
    PRESELECTOR_CLASS = settings.PRESELECTOR_CLASS
