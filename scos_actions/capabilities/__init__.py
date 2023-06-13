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
except Exception as e:
    logger.warning(
        f"Failed to load sensor definition file: {SENSOR_DEFINITION_FILE}"
        + "\nAn empty sensor definition will be used"
    )
    capabilities["sensor"] = {"sensor_spec": {"id": "unknown"}}

if capabilities["sensor"]:
    # Generate sensor definition file hash (SHA 512)
    try:
        _hash = hashlib.sha512()
        with open(SENSOR_DEFINITION_FILE, "rb") as f:
            for b in iter(lambda: f.read(4096), b""):
                _hash.update(b)
        SENSOR_DEFINITION_HASH = _hash.hexdigest()
        logger.debug("Generated sensor definition hash")
    except Exception as e:
        logger.error(f"Unable to generate sensor definition hash")
        # SENSOR_DEFINITION_HASH is None, do not raise Exception
        logger.debug(e)

if FQDN:
    capabilities["sensor"]["sensor_spec"]["id"] = FQDN
