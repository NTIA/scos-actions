import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Calibration:
    calibration_datetime: str
    calibration_parameters: list
    calibration_data: dict
    clock_rate_lookup_by_sample_rate: list  # list of dict

    def get_clock_rate(self, sample_rate):
        """Find the clock rate (Hz) using the given sample_rate (samples per second)"""
        for mapping in self.clock_rate_lookup_by_sample_rate:
            mapped = get_comparable_value(mapping["sample_rate"])
            actual = get_comparable_value(sample_rate)
            if mapped == actual:
                return mapping["clock_frequency"]
        return sample_rate

    def get_calibration_dict(self, args):
        """Find the calibration points closest to the specified args (gain, attenuation,ref_level...)."""

        # Check if the sample rate was calibrated
        cal_data = self.calibration_data
        for i in range(len(args)):
            setting_value = args[i]
            setting = self.calibration_parameters[i]
            logger.debug(
                "looking up calibration for {} at {}".format(setting, setting_value)
            )
            cal_data = filter_by_parameter(cal_data, setting, setting_value)
            if "calibration_datetime" not in cal_data:
                cal_data["calibration_datetime"] = self.calibration_datetime
        logger.debug(f"Cal Data: {cal_data}")
        return cal_data

    def update(
        self,
        params: dict,
        calibration_datetime: str,
        gain: float,
        noise_figure: float,
        temp: float,
        file_path: Path,
    ) -> None:
        cal_data = self.calibration_data
        self.calibration_datetime = calibration_datetime

        # Ensure all required calibration parameters were used
        if not set(params.keys()) >= set(self.calibration_parameters):
            raise Exception(
                "Not enough parameters specified to update calibration.\n"
                + f"Required parameters are {self.calibration_parameters}"
            )

        # Get calibration entry by parameters used
        for parameter in self.calibration_parameters:
            value = params[parameter]
            logger.debug(f"Updating calibration at {parameter} = {value}")
            try:
                cal_data = cal_data[value]
            except KeyError:
                logger.debug(
                    f"Creating required calibration data field for {parameter} = {value}"
                )
                cal_data[value] = {}
                cal_data = cal_data[value]

        # Update calibration entry
        cal_data.update(
            {
                "calibration_datetime": self.calibration_datetime,
                "gain_sensor": gain,
                "noise_figure_sensor": noise_figure,
                "temperature": temp,
            }
        )

        # Write updated calibration data to file
        dict = {
            "calibration_datetime": str(self.calibration_datetime),
            "calibration_parameters": self.calibration_parameters,
            "clock_rate_lookup_by_sample_rate": self.clock_rate_lookup_by_sample_rate,
            "calibration_data": self.calibration_data,
        }
        with open(file_path, "w") as outfile:
            outfile.write(json.dumps(dict))


def get_comparable_value(f):
    """Allow a frequency of type [float] to be compared with =="""
    f = int(round(f))
    return f


def load_from_json(fname: Path):
    with open(fname) as file:
        calibration = json.load(file)
    # Check that the required fields are in the dict
    if not calibration.keys() >= {
        "calibration_datetime",
        "calibration_data",
        "clock_rate_lookup_by_sample_rate",
    }:
        raise Exception("Loaded calibration dictionary is missing required fields.")
    calibration_data = convert_keys(calibration["calibration_data"])
    # Create and return the Calibration object
    return Calibration(
        calibration["calibration_datetime"],
        calibration["calibration_parameters"],
        calibration_data,
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


def filter_by_parameter(calibrations, parameter, value):
    filtered_data = None
    if value not in calibrations:
        filtered_data = check_floor_of_parameter(calibrations, parameter, value)
        if filtered_data is None:
            filtered_data = check_ceiling_of_parameter(calibrations, parameter, value)
            if filtered_data is None:
                logger.debug(calibrations)
                raise Exception(
                    "No calibration was performed with {} at {}".format(
                        parameter, value
                    )
                )
    else:
        filtered_data = calibrations[value]

    return filtered_data


def check_floor_of_parameter(calibrations, parameter, value):
    value = math.floor(value)
    logger.debug(f"Checking floor value of: {value}")
    if value in calibrations:
        return calibrations[value]
    else:
        return None


def check_ceiling_of_parameter(calibrations, parameter, value):
    value = math.ceil(value)
    logger.debug(f"Checking ceiling at: {value}")
    if value in calibrations:
        return calibrations[value]
    else:
        return None
