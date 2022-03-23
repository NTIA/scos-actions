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