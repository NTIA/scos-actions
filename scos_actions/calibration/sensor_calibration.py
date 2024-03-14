"""
TODO
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Union

from environs import Env

from scos_actions.calibration.interfaces.calibration import Calibration
from scos_actions.calibration.utils import CalibrationEntryMissingException
from scos_actions.utils import parse_datetime_iso_format_str

logger = logging.getLogger(__name__)


@dataclass
class SensorCalibration(Calibration):
    last_calibration_datetime: str
    clock_rate_lookup_by_sample_rate: List[Dict[str, float]]
    sensor_uid: str

    def get_clock_rate(self, sample_rate: Union[float, int]) -> Union[float, int]:
        """Find the clock rate (Hz) using the given sample_rate (samples per second)"""
        for mapping in self.clock_rate_lookup_by_sample_rate:
            if mapping["sample_rate"] == sample_rate:
                return mapping["clock_frequency"]
        return sample_rate

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
        object and additionally writes these changes to the specified
        output file.

        :param params: Parameters used for calibration. This must include
            entries for all of the ``Calibration.calibration_parameters``
            Example: ``{"sample_rate": 14000000.0, "attenuation": 10.0}``
        :param calibration_datetime_str: Calibration datetime string,
            as returned by ``scos_actions.utils.get_datetime_str_now()``
        :param gain_dB: Gain value from calibration, in dB.
        :param noise_figure_dB: Noise figure value for calibration, in dB.
        :param temp_degC: Temperature at calibration time, in degrees Celsius.
        :param file_path: File path for saving the updated calibration data.
        """
        try:
            # Get existing calibration data entry which will be updated
            data_entry = self.get_calibration_dict(params)
        except CalibrationEntryMissingException:
            # Existing entry does not exist for these parameters. Make one.
            data_entry = self.calibration_data
            for p_name in self.calibration_parameters:
                p_val = params[p_name]
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
        time_limit = env("CALIBRATION_EXPIRATION_LIMIT", default=None)
        if time_limit is None:
            return False
        elif self.calibration_data is None:
            return True
        elif len(self.calibration_data) == 0:
            return True
        else:
            now_string = datetime.utcnow().isoformat(timespec="milliseconds") + "Z"
            now = parse_datetime_iso_format_str(now_string)
            cal_data = self.calibration_data
            return has_expired_cal_data(cal_data, now, time_limit)


def has_expired_cal_data(cal_data: dict, now: datetime, time_limit: int) -> bool:
    expired = False
    if "datetime" in cal_data:
        expired = expired or date_expired(cal_data, now, time_limit)

    for key, value in cal_data.items():
        if isinstance(value, dict):
            expired = expired or has_expired_cal_data(value, now, time_limit)
    return expired


def date_expired(cal_data: dict, now: datetime, time_limit: int):
    cal_datetime = parse_datetime_iso_format_str(cal_data["datetime"])
    elapsed = now - cal_datetime
    if elapsed.total_seconds() > time_limit:
        return True
    return False
