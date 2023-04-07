import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

logger = logging.getLogger(__name__)


@dataclass
class Calibration:
    last_calibration_datetime: str
    calibration_parameters: List[str]
    calibration_data: dict
    clock_rate_lookup_by_sample_rate: List[Dict[str, float]]

    def get_clock_rate(self, sample_rate):
        """Find the clock rate (Hz) using the given sample_rate (samples per second)"""
        for mapping in self.clock_rate_lookup_by_sample_rate:
            mapped = get_comparable_value(mapping["sample_rate"])
            actual = get_comparable_value(sample_rate)
            if mapped == actual:
                return mapping["clock_frequency"]
        return sample_rate

    def get_calibration_dict(self, cal_params: List[Union[float, int, bool]]) -> dict:
        """
        Get calibration data closest to the specified parameter values.

        :param cal_params: List of calibration parameter values. For example,
            if ``calibration_parameters`` are ``["sample_rate", "gain"]``,
            then the input to this method could be ``["15360000.0", "40"]``.
        :return: The calibration data corresponding to the input parameter values.
        """
        # Check if the sample rate was calibrated
        cal_data = self.calibration_data
        for i, setting_value in enumerate(cal_params):
            setting = self.calibration_parameters[i]
            logger.debug(f"Looking up calibration for {setting} at {setting_value}")
            cal_data = filter_by_parameter(cal_data, setting_value)
        logger.debug(f"Got calibration data: {cal_data}")
        return cal_data

    def update(
        self,
        params: dict,
        calibration_datetime_str: str,
        gain_dB: float,
        noise_figure_dB: float,
        temp_degC: float,
        file_path: Path,
    ) -> None:
        """
        Update the calibration data by overwriting or adding an entry.

        This method updates the instance variables of the ``Calibration``
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
        :raises Exception:
        """
        cal_data = self.calibration_data
        self.last_calibration_datetime = calibration_datetime_str

        # Ensure all required calibration parameters were used
        if not set(params.keys()) >= set(self.calibration_parameters):
            raise Exception(
                "Not enough parameters specified to update calibration.\n"
                + f"Required parameters are {self.calibration_parameters}"
            )

        # Get calibration entry by parameters used
        for parameter in self.calibration_parameters:
            value = str(params[parameter]).lower()
            logger.debug(f"Updating calibration at {parameter} = {value}")
            try:
                cal_data = cal_data[value]
            except KeyError:
                logger.debug(
                    f"Creating required calibration data field for {parameter} = {value}"
                )
                cal_data[value] = {}
                cal_data = cal_data[value]

        # Update calibration data
        cal_data.update(
            {
                "datetime": calibration_datetime_str,
                "gain_sensor": gain_dB,
                "noise_figure_sensor": noise_figure_dB,
                "temperature": temp_degC,
            }
        )

        # Write updated calibration data to file
        cal_dict = {
            "last_calibration_datetime": self.last_calibration_datetime,
            "calibration_parameters": self.calibration_parameters,
            "clock_rate_lookup_by_sample_rate": self.clock_rate_lookup_by_sample_rate,
            "calibration_data": self.calibration_data,
        }
        with open(file_path, "w") as outfile:
            outfile.write(json.dumps(cal_dict))


def get_comparable_value(f: Union[float, int]) -> int:
    """Allow a frequency of type [float] to be compared with =="""
    f = int(round(f))
    return f


def load_from_json(fname: Path):
    with open(fname) as file:
        calibration = json.load(file)
    # Check that the required fields are in the dict
    required_keys = {
        "last_calibration_datetime",
        "calibration_data",
        "clock_rate_lookup_by_sample_rate",
        "calibration_parameters",
    }
    if not calibration.keys() >= required_keys:
        raise Exception(
            "Loaded calibration dictionary is missing required fields."
            + f"Existing fields: {set(calibration.keys())}\n"
            + f"Required fields: {required_keys}\n"
        )
    # Create and return the Calibration object
    return Calibration(
        calibration["last_calibration_datetime"],
        calibration["calibration_parameters"],
        convert_keys(calibration["calibration_data"]),
        calibration["clock_rate_lookup_by_sample_rate"],
    )


def convert_keys(dictionary):
    if isinstance(dictionary, dict):
        keys = list(dictionary.keys())
        for key in keys:
            vals = dictionary[key]
            try:
                new_key = float(key)
                vals = convert_keys(vals)
                dictionary.pop(key)
                dictionary[new_key] = vals
            except Exception:
                vals = convert_keys(vals)
                dictionary.pop(key)
                dictionary[key] = vals
        return dictionary
    else:
        return dictionary


def filter_by_parameter(calibrations: dict, value: Union[float, int, bool]) -> dict:
    """
    Select a certain element by the value of a top-level key in a dictionary.

    This method should be recursively called to select calibration
    data matching a set of calibration parameters. The ordering of
    nested dictionaries should match the ordering of the required
    calibration parameters in the calibration file.

    If ``value`` is a float or bool, ``str(value).lower()`` is used
    as the dictionary key. If ``value`` is an int, and the previous
    approach does not work, ``str(float(value))`` is attempted. This
    allows for value ``1`` to match a key ``"1.0"``.

    :param calibrations: Calibration data dictionary.
    :param value: The parameter value for filtering. This value should
        exist as a top-level key in ``calibrations``.
    :raises KeyError: If ``value`` cannot be matched to a top-level key
        in ``calibrations``.
    :return: The value of ``calibrations[value]``, which should be a dict.
    """
    try:
        filtered_data = calibrations.get(str(value).lower(), None)
        if filtered_data is None and isinstance(value, int):
            # Try equivalent float for ints, i.e., match "1.0" to 1
            filtered_data = calibrations[str(float(value))]
        return filtered_data
    except KeyError as e:
        logger.error(
            f"Could not location calibration data with at {value}"
            + f"\nAttempted lookup using key '{str(value).lower()}'"
            + f"{f'and {float(value)}' if isinstance(value, int) else ''}"
            + f"\nUsing calibration data: {calibrations}"
        )
        raise e
