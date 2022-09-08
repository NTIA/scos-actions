import logging

from scos_actions.calibration.calibration import load_from_json
from scos_actions.settings import SENSOR_CALIBRATION_FILE, SIGAN_CALIBRATION_FILE

logger = logging.getLogger(__name__)


def get_sigan_calibration(sigan_cal_file):
    try:
        sigan_cal = load_from_json(sigan_cal_file)
    except Exception as err:
        logger.error("Unable to load sigan calibration data, reverting to none")
        logger.exception(err)
        sigan_cal = None

    return sigan_cal


def get_sensor_calibration(sensor_cal_file):
    """Get calibration data from sensor_cal_file and sigan_cal_file."""
    try:
        sensor_cal = load_from_json(sensor_cal_file)
    except Exception as err:
        logger.error("Unable to load sensor calibration data, reverting to none")
        logger.exception(err)
        sensor_cal = None
    return sensor_cal


logger.info(f"Loading sensor cal file: {SENSOR_CALIBRATION_FILE}")
sensor_calibration = get_sensor_calibration(SENSOR_CALIBRATION_FILE)
logger.info(f"Loading sigan cal file: {SIGAN_CALIBRATION_FILE}")
sigan_calibration = get_sigan_calibration(SIGAN_CALIBRATION_FILE)
if sensor_calibration:
    logger.info(f"last sensor cal: {sensor_calibration.calibration_datetime}")
