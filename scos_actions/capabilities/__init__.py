import logging

from scos_actions import utils
from scos_actions.settings import FQDN, SENSOR_DEFINITION_FILE

logger = logging.getLogger(__name__)
capabilities = {}

if SENSOR_DEFINITION_FILE:
    logger.info(f"Loading {SENSOR_DEFINITION_FILE}")
    capabilities["sensor"] = utils.load_from_json(SENSOR_DEFINITION_FILE)
else:
    capabilities["sensor"] = {}
if FQDN:
    capabilities["sensor"]["id"] = FQDN
else:
    capabilities["sensor"]["id"] = "unknown"
