import hashlib
import logging

from scos_actions import utils
from scos_actions.metadata.utils import construct_geojson_point
from scos_actions.settings import FQDN, SENSOR_DEFINITION_FILE

logger = logging.getLogger(__name__)
capabilities = {}
SENSOR_DEFINITION_HASH = None
SENSOR_LOCATION = None

logger.info(f"Loading {SENSOR_DEFINITION_FILE}")
try:
    capabilities["sensor"] = utils.load_from_json(SENSOR_DEFINITION_FILE)
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
except Exception as e:
    logger.warning(
        f"Failed to load sensor definition file: {SENSOR_DEFINITION_FILE}"
        + "\nAn empty sensor definition will be used"
    )
    capabilities["sensor"] = {"sensor_spec": {"id": "unknown"}}

# Extract location from sensor definition file, if present
if "location" in capabilities["sensor"]:
    try:
        sensor_loc = capabilities["sensor"].pop("location")
        SENSOR_LOCATION = construct_geojson_point(
            sensor_loc["x"],
            sensor_loc["y"],
            sensor_loc["z"] if "z" in sensor_loc else None,
        )
    except:
        logger.warning("Failed to get sensor location from sensor definition.")

if FQDN:
    capabilities["sensor"]["sensor_spec"]["id"] = FQDN
