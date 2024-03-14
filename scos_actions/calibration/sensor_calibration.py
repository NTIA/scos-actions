"""
TODO
"""
import logging
from dataclasses import dataclass
from typing import Dict, List, Union

from scos_actions.calibration.interfaces.calibration import Calibration
from scos_actions.calibration.utils import CalibrationEntryMissingException

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