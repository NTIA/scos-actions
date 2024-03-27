import dataclasses
import json
import logging
from abc import abstractmethod
from pathlib import Path
from typing import List, get_origin

from scos_actions.calibration.utils import (
    CalibrationParametersMissingException,
    filter_by_parameter,
)

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Calibration:
    """
    Base class to handle calibrated gains, noise figures, compression points, and losses.
    The calibration_parameters defined the settings used to perform calibrations and the
    order in which calibrations may be accessed in the calibration_data dictionary.
    For example, if calibration_parameters where [frequency, sample_rate] then the
    calibration for a particular frequency and sample rate would be accessed in
    the calibration_data dictionary by the string value of the frequency and
    sample rate, like calibration_data["3555000000.0"]["14000000.0"]. The
    calibration_reference indicates the reference point for the calibration, e.d.,
    antenna terminal, or noise source output. The file_path determines where
    updates (if allowed) will be saved.
    """

    calibration_parameters: List[str]
    calibration_data: dict
    calibration_reference: str
    file_path: Path

    def __post_init__(self):
        self._validate_fields()
        # Convert key names in data to strings
        # This means that formatting will always match between
        # native types provided in Python and data loaded from JSON
        self.calibration_data = json.loads(json.dumps(self.calibration_data))

    def _validate_fields(self) -> None:
        """Loosely check that the input types are as expected."""
        for f_name, f_def in self.__dataclass_fields__.items():
            # Note that nested types are not checked: i.e., "List[str]"
            # will surely be a list, but may not be filled with strings.
            f_type = get_origin(f_def.type) or f_def.type
            actual_value = getattr(self, f_name)
            if not isinstance(actual_value, f_type):
                c_name = self.__class__.__name__
                actual_type = type(actual_value)
                raise TypeError(
                    f"{c_name} field {f_name} must be {f_type}, not {actual_type}"
                )

    def get_calibration_dict(self, params: dict) -> dict:
        """
        Get calibration data entry at the specified parameter values.

        :param params: Parameters used for calibration. This must include
            entries for all of the ``Calibration.calibration_parameters``
            Example: ``{"sample_rate": 14000000.0, "attenuation": 10.0}``
        :return: The calibration data corresponding to the input parameter values.
        """
        # Check that input includes all required calibration parameters
        if not set(params.keys()) >= set(self.calibration_parameters):
            raise CalibrationParametersMissingException(
                params, self.calibration_parameters
            )
        cal_data = self.calibration_data
        for p_name in self.calibration_parameters:
            p_value = params[p_name]
            logger.debug(f"Looking up calibration data at {p_name}={p_value}")
            cal_data = filter_by_parameter(cal_data, p_value)

        logger.debug(f"Got calibration data: {cal_data}")

        return cal_data

    @abstractmethod
    def update(self):
        """Update the calibration data"""
        raise NotImplementedError

    @classmethod
    def from_json(cls, fname: Path):
        """
        Load a calibration from a JSON file.

        The JSON file must contain top-level fields
        with names identical to the dataclass fields for
        the class being constructed.

        :param fname: The ``Path`` to the JSON calibration file.
        :raises Exception: If the provided file does not include
            the required keys.
        :return: The ``Calibration`` object generated from the file.
        """
        with open(fname) as file:
            calibration = json.load(file)
        cal_file_keys = set(calibration.keys())

        # Check that only the required fields are in the dict
        required_keys = {f.name for f in dataclasses.fields(cls)}
        required_keys -= {"file_path"}  # not required in JSON
        if cal_file_keys == required_keys:
            pass
        elif cal_file_keys >= required_keys:
            extra_keys = cal_file_keys - required_keys
            logger.warning(
                f"Loaded calibration file contains fields which will be ignored: {extra_keys}"
            )
            for k in extra_keys:
                calibration.pop(k, None)
        else:
            raise Exception(
                "Loaded calibration dictionary is missing required fields.\n"
                + f"Existing fields: {cal_file_keys}\n"
                + f"Required fields: {required_keys}\n"
                + f"Missing fields: {required_keys - cal_file_keys}"
            )

        # Create and return the Calibration object
        return cls(file_path=fname, **calibration)

    def to_json(self) -> None:
        """
        Save the calibration to a JSON file.

        The JSON file will be located at ``self.file_path`` and will
        contain a copy of ``self.__dict__``, except for the ``file_path``
        key/value pair. This includes all dataclass fields, with their
        parameter names as JSON key names.
        """
        dict_to_json = self.__dict__.copy()
        # Remove keys which should not save to JSON
        dict_to_json.pop("file_path", None)
        with open(self.file_path, "w") as outfile:
            outfile.write(json.dumps(dict_to_json))
        logger.debug("Finished updating calibration file.")
