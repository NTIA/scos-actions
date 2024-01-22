from typing import Optional

import msgspec
from scos_actions.metadata.structs.ntia_sensor import Calibration, SiganSettings
from scos_actions.metadata.utils import SIGMF_OBJECT_KWARGS

capture_segment_rename_map = {
    "sample_start": "core:sample_start",
    "global_index": "core:global_index",
    "header_bytes": "core:header_bytes",
    "frequency": "core:frequency",
    "datetime": "core:datetime",
    "duration": "ntia-sensor:duration",
    "overload": "ntia-sensor:overload",
    "sensor_calibration": "ntia-sensor:sensor_calibration",
    # sigan_calibration is unused by SCOS Sensor but still defined
    # in the ntia-sensor extension as of v2.0.0
    "sigan_calibration": "ntia-sensor:sigan_calibration",
    "sigan_settings": "ntia-sensor:sigan_settings",
}


class CaptureSegment(
    msgspec.Struct, rename=capture_segment_rename_map, **SIGMF_OBJECT_KWARGS
):
    """
    Interface for generating SigMF Capture Segment Objects.

    Optionally supports extensions defined in `ntia-sensor`.

    :param sample_start: The sample index in the Dataset file at which this
        Segment takes effect.
    :param global_index: The index of the sample referenced by `sample_start`
        relative to an original sample stream.
    :param header_bytes: The number of bytes preceding a chunk of samples that
        are not sample data, used for NCDs.
    :param frequency: The center frequency of the signal, in Hz.
    :param datetime: Timestamp of the sample index specified by `sample_start`. Must be
        an ISO 8601 formatted string.
    :param duration: Duration of IQ signal capture, in ms (from `ntia-sensor`).
    :param overload: Whether signal analyzer overload occurred (from `ntia-sensor`).
    :param sensor_calibration: Sensor calibration metadata (from `ntia-sensor`).
    :param sigan_calibration: Signal analyzer calibration metadata (from `ntia-sensor`).
    :param sigan_settings: Signal analyzer settings using during capture (from `ntia-sensor`).
    """

    sample_start: int
    global_index: Optional[int] = None
    header_bytes: Optional[int] = None
    frequency: Optional[float] = None
    datetime: Optional[str] = None
    duration: Optional[int] = None
    overload: Optional[bool] = None
    sensor_calibration: Optional[Calibration] = None
    sigan_calibration: Optional[Calibration] = None
    sigan_settings: Optional[SiganSettings] = None
