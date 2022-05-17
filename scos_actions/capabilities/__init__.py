import logging

from scos_actions.settings import SENSOR_DEFINITION_FILE
from scos_actions.settings import FQDN
from scos_actions import utils

capabilities = {}
if SENSOR_DEFINITION_FILE:
    logging.info('Loading ' + SENSOR_DEFINITION_FILE)
    capabilities["sensor"] = utils.load_from_json(SENSOR_DEFINITION_FILE)
else:
    capabilities["sensor"] = {}
if FQDN:
    capabilities["sensor"]["id"] = FQDN
else:
    capabilities["sensor"]["id"] = "unknown"