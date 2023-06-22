import logging
from typing import List, Optional

import msgspec

from scos_actions.metadata.structs.ntia_core import Antenna, HardwareSpec
from scos_actions.metadata.structs.ntia_environment import Environment
from scos_actions.metadata.utils import SIGMF_OBJECT_KWARGS

logger = logging.getLogger(__name__)


class SignalAnalyzer(msgspec.Struct, **SIGMF_OBJECT_KWARGS):
    """
    Interface for generating `ntia-sensor` `SignalAnalyzer` objects.

    :param sigan_spec: Metadata to describe/specify the signal analyzer.
    :param frequency_low: Low frequency of operational range of the signal
        analyzer, in Hz.
    :param frequency_high: Low frequency of operational range of the signal
        analyzer, in Hz.
    :param noise_figure: Noise figure of the signal analyzer, in dB.
    :param max_power: Maximum input power of the signal analyzer, in dBm.
    :param a2d_bits: Number of bits in the A/D converter.
    """

    sigan_spec: Optional[HardwareSpec] = None
    frequency_low: Optional[float] = None
    frequency_high: Optional[float] = None
    noise_figure: Optional[float] = None
    max_power: Optional[float] = None
    a2d_bits: Optional[int] = None

    def __post_init__(self):
        super().__post_init__()
        # Define SigMF key names
        self.obj_keys.update(
            {
                "sigan_spec": "sigan_spec",
                "frequency_low": "frequency_low",
                "frequency_high": "frequency_high",
                "noise_figure": "noise_figure",
                "max_power": "max_power",
                "a2d_bits": "a2d_bits",
            }
        )


class CalSource(
    msgspec.Struct, rename={"cal_source_type": "type"}, **SIGMF_OBJECT_KWARGS
):
    """
    Interface for generating `ntia-sensor` `CalSource` objects.

    :param cal_source_spec: Metadata to describe/specify the calibration source.
    :param cal_source_type: Type of calibration source.
    :param enr: Excess noise ratio, in dB.
    """

    cal_source_spec: Optional[HardwareSpec] = None
    cal_source_type: Optional[str] = None
    enr: Optional[float] = None


class Amplifier(msgspec.Struct, **SIGMF_OBJECT_KWARGS):
    """
    Interface for generating `ntia-sensor` `Amplifier` objects.

    :param amplifier_spec: Metadata to describe/specify the amplifier.
    :param gain: Gain of the low noise amplifier, in dB.
    :param noise_figure: Noise figure of the low noise amplifier, in dB.
    :param max_power: Maximum power of the low noise amplifier, in dBm.
    """

    amplifier_spec: Optional[HardwareSpec] = None
    gain: Optional[float] = None
    noise_figure: Optional[float] = None
    max_power: Optional[float] = None


class Filter(msgspec.Struct, **SIGMF_OBJECT_KWARGS):
    """
    Interface for generating `ntia-sensor` `Filter` objects.

    :param filter_spec: Metadata to describe/specify the filter.
    :param frequency_low_passband: Low frequency of filter 1 dB passband, in Hz.
    :param frequency_high_passband: High frequency of filter 1 dB passband, in Hz.
    :param frequency_low_stopband: Low frequency of filter 60 dB stopband, in Hz.
    :param frequency_high_stopband: High frequency of filter 60 dB stopband, in Hz.
    """

    filter_spec: Optional[HardwareSpec] = None
    frequency_low_passband: Optional[float] = None
    frequency_high_passband: Optional[float] = None
    frequency_low_stopband: Optional[float] = None
    frequency_high_stopband: Optional[float] = None


class RFPath(msgspec.Struct, **SIGMF_OBJECT_KWARGS):
    """
    Interface for generating `ntia-sensor` `RFPath` objects.

    :param id: Unique name or ID for the RF path.
    :param cal_source_id: ID of the calibration source.
    :param filter_id: ID of the filter.
    :param amplifier_id: ID of the amplifier.
    """

    id: str
    cal_source_id: Optional[str] = None
    filter_id: Optional[str] = None
    amplifier_id: Optional[str] = None


class Preselector(msgspec.Struct, **SIGMF_OBJECT_KWARGS):
    """
    Interface for generating `ntia-sensor` `Preselector` objects.

    :param preselector_spec: Metadata to describe/specify the preselector.
    :param cal_sources: Metadata to describe/specify the preselector calibration source(s).
    :param amplifiers: Metadata to describe/specify the preselector low noise amplifier(s).
    :param filters: Metadata to describe the preselector RF filter(s).
    :param rf_paths: Metadata that describes preselector RF path(s).
    """

    preselector_spec: Optional[HardwareSpec] = None
    cal_sources: Optional[List[CalSource]] = None
    amplifiers: Optional[List[Amplifier]] = None
    filters: Optional[List[Filter]] = None
    rf_paths: Optional[List[RFPath]] = None


class Calibration(
    msgspec.Struct,
    rename={"compression_point": "1db_compression_point"},
    **SIGMF_OBJECT_KWARGS,
):
    """
    Interface for generating `ntia-sensor` `Calibration` objects.

    :param datetime: Timestamp for the calibration data in this object.
    :param gain: Calibrated gain of signal analyzer or sensor, in dB.
    :param noise_figure: Calibrated gain of signal analyzer or sensor, in dB.
    :param compression_point: Signal analyzer or sensor input power level at which the received power level
        is compressed by 1 dB, in dBm.
    :param enbw: Equivalent noise bandwidth of the signal analyzer or sensor, in Hz.
    :param mean_noise_power: Mean signal analyzer or sensor noise floor power, in
        `mean_noise_power_units`.
    :param mean_noise_power_units: The units of the `mean_noise_power`.
    :param reference: Reference point for the calibration data, e.g., `"signal analyzer input"`,
        `"preselector input"`.
    :param temperature: Temperature during calibration, in degrees Celsius.
    """

    datetime: Optional[str] = None
    gain: Optional[float] = None
    noise_figure: Optional[float] = None
    compression_point: Optional[float] = None
    enbw: Optional[float] = None
    mean_noise_power: Optional[float] = None
    mean_noise_power_units: Optional[str] = None
    reference: Optional[str] = None
    temperature: Optional[float] = None


class SiganSettings(msgspec.Struct, **SIGMF_OBJECT_KWARGS):
    """
    Interface for generating `ntia-sensor` `SiganSettings` objects.

    :param gain: Gain setting of the signal analyzer, in dB.
    :param reference_level: Reference level setting of the signal analyzer, in dBm.
    :param attenuation: Attenuation setting of the signal analyzer, in dB.
    :param preamp_enable: True if signal analyzer preamplifier is enabled.
    """

    gain: Optional[float] = None
    reference_level: Optional[float] = None
    attenuation: Optional[float] = None
    preamp_enable: Optional[float] = None


class Sensor(msgspec.Struct, **SIGMF_OBJECT_KWARGS):
    """
    Interface for generating `ntia-sensor` `Sensor` objects.

    :param sensor_spec: A `HardwareSpec` object specifying the sensor.
    :param antenna: An `Antenna` object specifying the antenna.
    :param preselector: A `Preselector` object specifying the preselector.
    :param signal_analyzer: A `SignalAnalyzer` object specifying the signal analyzer.
    :param computer_spec: A `HardwareSpec` object specifying the onbaord computer.
    :param mobile: Whether the sensor is mobile.
    :param environment: An `Environment` object specifying the sensor's environment.
    :param sensor_sha512: SHA-512 hash of the sensor definition.
    """

    sensor_spec: HardwareSpec
    antenna: Optional[Antenna] = None
    preselector: Optional[Preselector] = None
    signal_analyzer: Optional[SignalAnalyzer] = None
    computer_spec: Optional[HardwareSpec] = None
    mobile: Optional[bool] = None
    environment: Optional[Environment] = None
    sensor_sha512: Optional[str] = None
