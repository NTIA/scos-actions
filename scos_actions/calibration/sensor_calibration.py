import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Union

from environs import Env

from scos_actions.calibration.interfaces.calibration import Calibration
from scos_actions.calibration.utils import CalibrationEntryMissingException
from scos_actions.utils import get_datetime_str_now, parse_datetime_iso_format_str

logger = logging.getLogger(__name__)


@dataclass
class SensorCalibration(Calibration):
    """
    Extends the ``Calibration`` class to represent calibration
    data that may be updated. Within SCOS Sensor,``SensorCalibration``
    instances are used to handle calibration files generated prior
    to deployment through a lab-based calibration as well as the result
    of calibrations that are performed by the sensor in the field. This
    class provides an implementation for the update method to allow calibration
    data to be updated with new values.
    """

    last_calibration_datetime: str
    sensor_uid: str

    def update(
        self,
        params: dict,
        calibration_datetime_str: str,
        gain_dB: float,
        noise_figure_dB: float,
        temp_degC: float,
    ) -> None:
        """
        Update the calibration data by overwriting or adding an entry.

        This updates the instance variables of the ``SensorCalibration``
        object and additionally writes these changes to file specified
        by the instance's file_path property.

        :param params: Parameters used for calibration. This must include
            entries for all of the ``Calibration.calibration_parameters``
            Example: ``{"sample_rate": 14000000.0, "attenuation": 10.0}``
        :param calibration_datetime_str: Calibration datetime string,
            as returned by ``scos_actions.utils.get_datetime_str_now()``
        :param gain_dB: Gain value from calibration, in dB.
        :param noise_figure_dB: Noise figure value for calibration, in dB.
        :param temp_degC: Temperature at calibration time, in degrees Celsius.
        """
        logger.debug(f"Updating calibration file for params {params}")
        try:
            # Get existing calibration data entry which will be updated
            data_entry = self.get_calibration_dict(params)
        except CalibrationEntryMissingException:
            # Existing entry does not exist for these parameters. Make one.
            data_entry = self.calibration_data
            for p_name in self.calibration_parameters:
                p_val = str(params[p_name]).lower()
                try:
                    data_entry = data_entry[p_val]
                except KeyError:
                    logger.debug(
                        f"Creating calibration data field for {p_name}={p_val}"
                    )
                    data_entry[p_val] = {}
                    data_entry = data_entry[p_val]
        except Exception as e:
            logger.exception("Failed to update calibration data.")
            raise e

        # Update last calibration datetime
        self.last_calibration_datetime = calibration_datetime_str

        # Update calibration data entry (updates entry in self.calibration_data)
        data_entry.update(
            {
                "datetime": calibration_datetime_str,
                "gain": gain_dB,
                "noise_figure": noise_figure_dB,
                "temperature": temp_degC,
            }
        )

        # Write updated calibration data to file
        self.to_json()

    def expired(self) -> bool:
        env = Env()
        time_limit = env.int("CALIBRATION_EXPIRATION_LIMIT", default=None)
        logger.debug("Checking if calibration has expired.")
        now_string = get_datetime_str_now()
        now = parse_datetime_iso_format_str(now_string)
        if time_limit is None:
            return False
        elif self.calibration_data is None:
            return True
        elif len(self.calibration_data) == 0:
            return True
        elif date_expired(self.last_calibration_datetime, now, time_limit):
            return True
        else:
            cal_data = self.calibration_data
            return has_expired_cal_data(cal_data, now, time_limit)


def has_expired_cal_data(cal_data: dict, now: datetime, time_limit: int) -> bool:
    expired = False
    if "datetime" in cal_data:
        expired = expired or date_expired(cal_data["datetime"], now, time_limit)

    for key, value in cal_data.items():
        if isinstance(value, dict):
            expired = expired or has_expired_cal_data(value, now, time_limit)
    return expired


def date_expired(cal_date: str, now: datetime, time_limit: int):
    cal_datetime = parse_datetime_iso_format_str(cal_date)
    elapsed = now - cal_datetime
    logger.debug(f"{cal_datetime} is {elapsed} seconds old")
    if elapsed.total_seconds() > time_limit:
        logger.debug(
            f"Calibration at {cal_date} has expired at {elapsed.total_seconds()} seconds old."
        )
        return True
    return False
