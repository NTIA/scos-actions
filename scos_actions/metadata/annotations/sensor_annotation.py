from dataclasses import dataclass
from typing import Optional

from scos_actions.metadata.annotation_segment import AnnotationSegment


@dataclass
class SensorAnnotation(AnnotationSegment):
    """
    Interface for generating SensorAnnotation segments.

    All parameters are optional. Attenuation, gain, and overload
    values can generally be found in the measurement_result
    dictionary.

    Refer to the documentation of the ``ntia-sensor`` extension of
    SigMF for more information.

    :param rf_path_index: Index of the RF Path.
    :param overload: Indicator of sensor overload.
    :param attenuation_setting_sigan: Attenuation setting of the signal
        analyzer.
    :param gain_setting_sigan: Gain setting of the signal analyzer.
    :param gps_nmea: NMEA message from a GPS receiver.
    """

    rf_path_index: Optional[int] = None
    overload: Optional[bool] = None
    attenuation_setting_sigan: Optional[float] = None
    gain_setting_sigan: Optional[float] = None
    gps_nmea: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        # Define SigMF key names
        self.sigmf_keys.update(
            {
                "rf_path_index": "ntia-sensor:rf_path_index",
                "overload": "ntia-sensor:overload",
                "attenuation_setting_sigan": "ntia-sensor:attenutation_setting_sigan",
                "gain_setting_sigan": "ntia-sensor:gain_setting_sigan",
                "gps_nmea": "ntia-sensor:gps_nmea",
            }
        )
        # Create annotation segment
        super().create_annotation_segment()
