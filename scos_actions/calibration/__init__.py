import logging
from os import path

from scos_actions.calibration.calibration import Calibration, load_from_json
from scos_actions.settings import (
    DEFAULT_CALIBRATION_FILE,
    SENSOR_CALIBRATION_FILE,
    SIGAN_CALIBRATION_FILE,
)

logger = logging.getLogger(__name__)


def get_sigan_calibration(sigan_cal_file: str) -> Calibration:
    """
    Load signal analyzer calibration data from file.

    :param sigan_cal_file: Path to JSON file containing signal
        analyzer calibration data.
    :return: The signal analyzer ``Calibration`` object.
    """
    try:
        sigan_cal = load_from_json(sigan_cal_file)
    except Exception:
        sigan_cal = None
        logger.exception("Unable to load sigan calibration data, reverting to none")
    return sigan_cal


def get_sensor_calibration(sensor_cal_file: str) -> Calibration:
    """
    Load sensor calibration data from file.

    :param sensor_cal_file: Path to JSON file containing sensor
        calibration data.
    :return: The sensor ``Calibration`` object.
    """
    try:
        sensor_cal = load_from_json(sensor_cal_file)
    except Exception:
        sensor_cal = None
        logger.exception("Unable to load sensor calibration data, reverting to none")
    return sensor_cal


def check_for_default_calibration(cal_file_path: str, cal_type: str) -> bool:
    default_cal = False
    if cal_file_path == DEFAULT_CALIBRATION_FILE:
        default_cal = True
        logger.warning(
            f"***************LOADING DEFAULT {cal_type} CALIBRATION***************"
        )
    return default_cal


sensor_calibration = None
if SENSOR_CALIBRATION_FILE is None:
    logger.warning("Sensor calibration file is None. Not loading calibration file.")
elif not path.exists(SENSOR_CALIBRATION_FILE):
    logger.warning(
        SENSOR_CALIBRATION_FILE
        + " does not exist. Not loading sensor calibration file."
    )
else:
    logger.debug(f"Loading sensor cal file: {SENSOR_CALIBRATION_FILE}")
    default_sensor_calibration = check_for_default_calibration(
        SENSOR_CALIBRATION_FILE, "Sensor"
    )
    sensor_calibration = get_sensor_calibration(SENSOR_CALIBRATION_FILE)

sigan_calibration = None
if SIGAN_CALIBRATION_FILE is None:
    logger.warning("Sigan calibration  file is None. Not loading calibration file.")
elif not path.exists(SIGAN_CALIBRATION_FILE):
    logger.warning(
        SIGAN_CALIBRATION_FILE + " does not exist. Not loading sigan calibration file."
    )
else:
    logger.debug(f"Loading sigan cal file: {SIGAN_CALIBRATION_FILE}")
    default_sigan_calibration = check_for_default_calibration(
        SIGAN_CALIBRATION_FILE, "Sigan"
    )
    sigan_calibration = get_sigan_calibration(SIGAN_CALIBRATION_FILE)

if sensor_calibration:
    logger.debug(f"Last sensor cal: {sensor_calibration.last_calibration_datetime}")
