from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Union

from scos_actions.metadata.interfaces.ntia_sensor import Calibration, SiganSettings
from scos_actions.metadata.interfaces.sigmf_object import SigMFObject
from scos_actions.utils import convert_datetime_to_millisecond_iso_format


@dataclass
class CaptureSegment(SigMFObject):
    """
    Interface for generating SigMF Capture Segment Objects.

    The `sample_start` parameter is required.

    Optionally supports extensions defined in `ntia-sensor`.

    :param sample_start: The sample index in the Dataset file at which this
        Segment takes effect.
    :param global_index: The index of the sample referenced by `sample_start`
        relative to an original sample stream.
    :param header_bytes: The number of bytes preceding a chunk of samples that
        are not sample data, used for NCDs.
    :param frequency: The center frequency of the signal, in Hz.
    :param datetime: Timestamp of the sample index specified by `sample_start`.
    :param duration: Duration of IQ signal capture, in ms (from `ntia-sensor`).
    :param overload: Whether signal analyzer overload occurred (from `ntia-sensor`).
    :param sensor_calibration: Sensor calibration metadata (from `ntia-sensor`).
    :param sigan_calibration: Signal analyzer calibration metadata (from `ntia-sensor`).
    :param sigan_settings: Signal analyzer settings using during capture (from `ntia-sensor`).
    """

    sample_start: Optional[int] = None
    global_index: Optional[int] = None
    header_bytes: Optional[int] = None
    frequency: Optional[float] = None
    datetime: Optional[Union[datetime, str]] = None
    duration: Optional[int] = None
    overload: Optional[bool] = None
    sensor_calibration: Optional[Calibration] = None
    sigan_calibration: Optional[Calibration] = None
    sigan_settings: Optional[SiganSettings] = None

    def __post_init__(self):
        super().__post_init__()
        # Ensure required keys have been set
        self.check_required(self.sample_start, "sample_start")
        # Convert datetime to string if needed
        if isinstance(self.datetime, datetime):
            self.datetime = convert_datetime_to_millisecond_iso_format(self.datetime)
        # Define SigMF key names
        self.obj_keys.update(
            {
                "sample_start": "core:sample_start",
                "global_index": "core:global_index",
                "header_bytes": "core:header_bytes",
                "frequency": "core:frequency",
                "datetime": "core:datetime",
                "duration": "ntia-sensor:duration",
                "overload": "ntia-sensor:overload",
                "sensor_calibration": "ntia-sensor:sensor_calibration",
                "sigan_calibration": "ntia-sensor:sigan_calibration",
                "sigan_settings": "ntia-sensor:sigan_settings",
            }
        )
        # Create metadata object
        super().create_json_object()
