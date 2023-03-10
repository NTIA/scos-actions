import logging
from pathlib import Path

from scos_actions.calibration.calibration import Calibration, load_from_json
from scos_actions.settings import SENSOR_CALIBRATION_FILE, SIGAN_CALIBRATION_FILE

logger = logging.getLogger(__name__)


def get_sigan_calibration(sigan_cal_file: Path) -> Calibration:
    """
    Load signal analyzer calibration data from file.

    :param sigan_cal_file: Path to JSON file containing signal
        analyzer calibration data.
    :return: The signal analyzer ``Calibration`` object.
    """
    try:
        sigan_cal = load_from_json(sigan_cal_file)
    except Exception as err:
        sigan_cal = None
        logger.error("Unable to load sigan calibration data, reverting to none")
        logger.exception(err)
    return sigan_cal


def get_sensor_calibration(sensor_cal_file: Path) -> Calibration:
    """
    Load sensor calibration data from file.

    :param sensor_cal_file: Path to JSON file containing sensor
        calibration data.
    :return: The sensor ``Calibration`` object.
    """
    try:
        sensor_cal = load_from_json(sensor_cal_file)
    except Exception as err:
        sensor_cal = None
        logger.error("Unable to load sensor calibration data, reverting to none")
        logger.exception(err)
    return sensor_cal


logger.info(f"Loading sensor cal file: {SENSOR_CALIBRATION_FILE}")
sensor_calibration = get_sensor_calibration(SENSOR_CALIBRATION_FILE)
logger.info(f"Loading sigan cal file: {SIGAN_CALIBRATION_FILE}")
sigan_calibration = get_sigan_calibration(SIGAN_CALIBRATION_FILE)
if sensor_calibration:
    logger.info(f"Last sensor cal: {sensor_calibration.last_calibration_datetime}")
