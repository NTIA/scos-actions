from acos_actions.settings import SENSOR_DEFINITION_FILE
from scos_actions.settings import FQDN
from scos_actions import utils

capabilities = {}
capabilities["sensor"] = utils.load_from_json(SENSOR_DEFINITION_FILE)
capabilities["sensor"]["id"] = FQDN