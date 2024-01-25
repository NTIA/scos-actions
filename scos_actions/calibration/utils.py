from typing import Union


class CalibrationException(Exception):
    """Basic exception handling for calibration functions."""

    def __init__(self, msg):
        super().__init__(msg)


class CalibrationEntryMissingException(CalibrationException):
    """Raised when filter_by_parameter cannot locate calibration data."""

    def __init__(self, msg):
        super().__init__(msg)


class CalibrationParametersMissingException(CalibrationException):
    """Raised when a dictionary does not contain all calibration parameters as keys."""

    def __init__(self, provided_dict: dict, required_keys: list):
        msg = (
            "Missing required parameters to lookup calibration data.\n"
            + f"Required parameters are {required_keys}\n"
            + f"Provided parameters are {list(provided_dict.keys())}"
        )
        super().__init__(msg)


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
    :raises CalibrationException: If ``value`` cannot be matched to a
        top-level key in ``calibrations``, or if ``calibrations`` is not
        a dict.
    :return: The value of ``calibrations[value]``, which should be a dict.
    """
    try:
        filtered_data = calibrations.get(str(value).lower(), None)
        if filtered_data is None and isinstance(value, int):
            # Try equivalent float for ints, i.e., match "1.0" to 1
            filtered_data = calibrations.get(str(float(value)), None)
        if filtered_data is None and isinstance(value, float) and value.is_integer():
            # Check for, e.g., key '25' if value is '25.0'
            filtered_data = calibrations.get(str(int(value)), None)
        if filtered_data is None:
            raise KeyError
        else:
            return filtered_data
    except AttributeError:
        # calibrations does not have ".get()"
        # Generally means that calibrations is None or not a dict
        msg = f"Provided calibration data is not a dict: {calibrations}"
        raise CalibrationException(msg)
    except KeyError:
        msg = (
            f"Could not locate calibration data at {value}"
            + f"\nAttempted lookup using key '{str(value).lower()}'"
            + f"{f'and {float(value)}' if isinstance(value, int) else ''}"
            + f"\nUsing calibration data: {calibrations}"
        )
        raise CalibrationEntryMissingException(msg)
