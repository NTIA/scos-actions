import logging
from datetime import datetime
from dateutil import parser
import json
import copy

logger = logging.getLogger(__name__)


class ParameterException(Exception):
    """Basic exception handling for missing parameters."""
    def __init__(self, param):
        super().__init__(f"{param} missing from measurement parameters.")


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


def load_from_json(fname):
    logger = logging.getLogger(__name__)
    try:
        with open(fname) as f:
            return json.load(f)
    except Exception:
        logger.exception("Unable to load JSON file {}".format(fname))


def get_iteration_parameters(i, parameters):
    iteration_params = {}
    for key in parameters:
        if key != 'name':
            iteration_params[key] = parameters[key][i]
    return iteration_params


def list_to_string(a_list):
    string_list = [str(i) for i in a_list ]
    return ','.join(string_list)


def get_parameter_map(params):
    if isinstance(params, list):
        key_map = {}
        for param in params:
            for key, value in param.items():
                key_map[key] = value
        return key_map
    elif isinstance(params, dict):
        return copy.deepcopy(params)


def get_parameter(p: str, params: dict):
    """
    Get a parameter by key from a parameter dictionary.

    :param p: The parameter name (key).
    :param params: The parameter dictionary.
    :return: The specified parameter (value).
    :raises ParameterException: If p is not a key in params.
    """
    if p not in params:
        raise ParameterException(p)
    return params[p]
