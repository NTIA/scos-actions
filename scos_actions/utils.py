import logging
from datetime import datetime
from dateutil import parser


logger = logging.getLogger(__name__)


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
