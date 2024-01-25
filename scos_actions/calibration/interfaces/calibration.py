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
    calibration_parameters: List[str]
    calibration_data: dict
    is_default: bool
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
    def from_json(cls, fname: Path, is_default: bool):
        """
        Load a calibration from a JSON file.

        The JSON file must contain top-level fields
        with names identical to the dataclass fields for
        the class being constructed.

        :param fname: The ``Path`` to the JSON calibration file.
        :param is_default: If True, the loaded calibration file
            is treated as the default calibration file.
        :raises Exception: If the provided file does not include
            the required keys.
        :return: The ``Calibration`` object generated from the file.
        """
        with open(fname) as file:
            calibration = json.load(file)
        cal_file_keys = set(calibration.keys())

        # Check that only the required fields are in the dict
        required_keys = {f.name for f in dataclasses.fields(cls)}
        required_keys -= {"is_default", "file_path"}  # are not required in JSON
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
        return cls(is_default=is_default, file_path=fname, **calibration)

    def to_json(self) -> None:
        """
        Save the calibration to a JSON file.

        The JSON file will be located at ``self.file_path`` and will
        contain a copy of ``self.__dict__``, except for the ``file_path``
        and ``is_default`` key/value pairs. This includes all dataclass
        fields, with their parameter names as JSON key names.
        """
        dict_to_json = self.__dict__.copy()
        # Remove keys which should not save to JSON
        dict_to_json.pop("file_path", None)
        dict_to_json.pop("is_default", None)
        with open(self.file_path, "w") as outfile:
            outfile.write(json.dumps(dict_to_json))
