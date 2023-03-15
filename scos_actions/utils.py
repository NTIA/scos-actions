import json
import logging
from datetime import datetime
from pathlib import Path

from dateutil import parser

from scos_actions.status import start_time

logger = logging.getLogger(__name__)


class ParameterException(Exception):
    """Basic exception handling for parameter-related problems."""

    def __init__(self, msg):
        super().__init__(msg)


def get_datetime_str_now():
    return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"


def convert_datetime_to_millisecond_iso_format(timestamp):
    return timestamp.replace(tzinfo=None).isoformat(timespec="milliseconds") + "Z"


def parse_datetime_iso_format_str(d):
    return parser.isoparse(d)


def convert_string_to_millisecond_iso_format(timestamp):
    # convert iso formatted datetime string to millisecond iso format
    if timestamp:
        parsed_timestamp = parse_datetime_iso_format_str(timestamp)
        return convert_datetime_to_millisecond_iso_format(parsed_timestamp)
    return None


def load_from_json(fname: Path):
    logger = logging.getLogger(__name__)
    try:
        with open(fname) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Unable to load JSON file {fname}")
        raise e


def get_iterable_parameters(parameters: dict, sortby: str = "frequency"):
    """
    Convert parameter dictionary into iterable list.

    The input parameters, as read from the YAML file, will be
    converted into an iterable list, in which each element is
    an individual set of corresponding parameters. This is useful
    for multi-frequency measurements, for example.

    The 'name' key is ignored, and not included in the returned list.

    This method also allows for YAML config files to specify
    single values for any parameters which should be held constant
    in every iteration. For example, specify only a single gain value
    to use the same gain value for all measurements in a stepped-frequency
    acquisition.

    The output list is automatically sorted by the key provided as the
    ``sortby`` parameter. By default, ``sortby`` is "frequency".

    :param parameters: The parameter dictionary, as loaded by the action.
    :param sortby: The key to sort the resulting list by, in ascending order.
        Defaults to "frequency".
    :return: An iterable list of parameter dictionaries based on the input.
        If only single values are given for all parameters in the input, a
        list will still be returned, containing a single dictionary.
    :raises ParameterException: If a parameter in the input has a number of
        values which is neither 1 nor the maximum number of values specified
        for any parameter.
    """
    # Create copy of parameters with all values as lists
    params = {k: (v if isinstance(v, list) else [v]) for k, v in parameters.items()}
    del params["name"]
    # Find longest set of parameters
    max_param_length = max(len(p) for p in params.values())
    if max_param_length > 1:
        for p_key, p_val in params.items():
            if len(p_val) == 1:
                # Repeat parameter to max length
                msg = f"Parameter {p_key} has only one value specified.\n"
                msg += "It will be used for all iterations in the action."
                logger.debug(msg)
                params[p_key] = p_val * max_param_length
            elif len(p_val) < max_param_length:
                # Don't make assumptions otherwise. Raise an error.
                msg = f"Parameter {p_key} has {len(p_val)} specified values.\n"
                msg += "YAML parameters must have either 1 value or a number of values equal to "
                msg += f"that of the parameter with the most values provided ({max_param_length})."
                raise ParameterException(msg)
    # Construct iterable parameter mapping
    result = [dict(zip(params, v)) for v in zip(*params.values())]
    result.sort(key=lambda param: param[sortby])
    return result


def list_to_string(a_list):
    string_list = [str(i) for i in a_list]
    return ",".join(string_list)


def get_parameter(p: str, params: dict):
    """
    Get a parameter by key from a parameter dictionary.

    :param p: The parameter name (key).
    :param params: The parameter dictionary.
    :return: The specified parameter (value).
    :raises ParameterException: If p is not a key in params.
    """
    if p not in params:
        raise ParameterException(
            f"{p} missing from measurement parameters."
            + f"Available parameters: {params}"
        )
    return params[p]


def get_days_up():
    elapsed = datetime.utcnow() - start_time
    days = elapsed.days
    fractional_day = elapsed.seconds / (60 * 60 * 24)
    return round(days + fractional_day, 4)
