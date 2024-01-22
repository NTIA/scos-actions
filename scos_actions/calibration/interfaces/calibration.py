import dataclasses
import json
import logging
from abc import abstractmethod
from pathlib import Path
from typing import Any, List, get_origin

from scos_actions.calibration.utils import filter_by_parameter

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
            f_type = get_origin(f_def.type) or f_def.type
            actual_value = getattr(self, f_name)
            if not isinstance(actual_value, f_type):
                c_name = self.__class__.__name__
                actual_type = type(actual_value)
                raise TypeError(
                    f"{c_name} field {f_name} must be {f_type}, not {actual_type}"
                )

    def get_calibration_dict(self, cal_params: List[Any]) -> dict:
        """
        Get calibration data closest to the specified parameter values.

        :param cal_params: List of calibration parameter values. For example,
            if ``calibration_parameters`` are ``["sample_rate", "gain"]``,
            then the input to this method could be ``["15360000.0", "40"]``.
        :return: The calibration data corresponding to the input parameter values.
        """
        cal_data = self.calibration_data
        for i, setting_value in enumerate(cal_params):
            setting = self.calibration_parameters[i]
            logger.debug(f"Looking up calibration for {setting} at {setting_value}")
            cal_data = filter_by_parameter(cal_data, setting_value)
        logger.debug(f"Got calibration data: {cal_data}")

        return cal_data

    def _retrieve_data_to_update(self, params: dict) -> dict:
        """
        Locate the calibration data entry to update, based on a set
        of calibration parameters.

        :param params: Parameters used for calibration. This must include
            entries for all of the ``Calibration.calibration_parameters``
            Example: ``{"sample_rate": 14000000.0, "attenuation": 10.0}``
        :return: A dict containing the existing calibration entry at
            the specified parameter set, which may be empty if none exists.
        :raises ValueError: If not all calibration parameters exist as keys
            in ``params``.
        """
        # Use params keys as calibration_parameters if none exist
        if len(self.calibration_parameters) == 0:
            logger.warning(
                f"Setting required calibration parameters to {list(params.keys())}"
            )
            self.calibration_parameters = list(params.keys())
        elif not set(params.keys()) >= set(self.calibration_parameters):
            # Otherwise ensure all required parameters were used
            raise ValueError(
                "Not enough parameters specified to update calibration.\n"
                + f"Required parameters are {self.calibration_parameters}"
            )

        # Retrieve the existing calibration data entry based on
        # the provided parameters and their values
        data_entry = self.calibration_data
        for parameter in self.calibration_parameters:
            value = str(params[parameter]).lower()
            logger.debug(f"Updating calibration at {parameter} = {value}")
            try:
                data_entry = data_entry[value]
            except KeyError:
                logger.debug(
                    f"Creating required calibration data field for {parameter} = {value}"
                )
                data_entry[value] = {}
                data_entry = data_entry[value]
        return data_entry

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
