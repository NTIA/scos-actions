import hashlib
import logging

from scos_actions import utils
from scos_actions.settings import FQDN, SENSOR_DEFINITION_FILE

logger = logging.getLogger(__name__)
capabilities = {}
SENSOR_DEFINITION_HASH = None

logger.info(f"Loading {SENSOR_DEFINITION_FILE}")
try:
    capabilities["sensor"] = utils.load_from_json(SENSOR_DEFINITION_FILE)
    # Generate sensor definition file hash (SHA 512)
    with open(SENSOR_DEFINITION_FILE, "rb") as f:
        digest = hashlib.file_digest(f, "sha512")
    SENSOR_DEFINITION_HASH = digest.hexdigest()
except Exception as e:
    logger.warning(
        f"Failed to load sensor definition file: {SENSOR_DEFINITION_FILE}"
        + "\nAn empty sensor definition will be used"
    )
    capabilities["sensor"] = {}

if FQDN:
    capabilities["sensor"]["id"] = FQDN
else:
    capabilities["sensor"]["id"] = "unknown"
