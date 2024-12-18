from enum import Enum
from typing import Optional, Union

import msgspec

from scos_actions.metadata.utils import SIGMF_OBJECT_KWARGS


class FilterType(str, Enum):
    IIR = "IIR"
    FIR = "FIR"


class DigitalFilter(msgspec.Struct, tag=True, **SIGMF_OBJECT_KWARGS):
    """
    Interface for generating `ntia-algorithm` `DigitalFilter` objects.

    :param id: Unique ID of the filter.
    :param filter_type: Type of the digital filter, given by the `FilterType`
        enum.
    :param feedforward_coefficients: Coefficients that define the feedforward
        filter stage.
    :param feedback_coefficients: Coefficients that define the feedback filter
        stage. SHOULD ONLY be present when `filter_type` is `"IIR"`.
    :param attenuation_cutoff: Attenuation that specifies the `frequency_cutoff`
        (typically 3 dB), in dB.
    :param frequency_cutoff: Frequency that characterizes the boundary between
        passband and stopband. Beyond this frequency, the signal is attenuated
        by at least `attenuation_cutoff`.
    :param description: Supplemental description of the filter.
    """

    id: str
    filter_type: FilterType
    feedforward_coefficients: Optional[list[float]] = None
    feedback_coefficients: Optional[list[float]] = None
    attenuation_cutoff: Optional[float] = None
    frequency_cutoff: Optional[float] = None
    description: Optional[str] = None


class Graph(msgspec.Struct, **SIGMF_OBJECT_KWARGS):
    """
    Interface for generating `ntia-algorithm` `Graph` objects.

    :param name: The name of the graph, e.g. `"Power Spectral Density"`,
        `"Signal Power vs. Time"`, `"M4S Detection"`
    :param series:
    """

    name: str
    series: Optional[list[str]] = None
    length: Optional[int] = None
    x_units: Optional[str] = None
    x_axis: Optional[list[Union[int, float, str]]] = None
    x_start: Optional[list[float]] = None
    x_stop: Optional[list[float]] = None
    x_step: Optional[list[float]] = None
    y_units: Optional[str] = None
    y_axis: Optional[list[Union[int, float, str]]] = None
    y_start: Optional[list[float]] = None
    y_stop: Optional[list[float]] = None
    y_step: Optional[list[float]] = None
    processing: Optional[list[str]] = None
    reference: Optional[str] = None
    description: Optional[str] = None


class DFT(msgspec.Struct, tag=True, **SIGMF_OBJECT_KWARGS):
    """
    Interface for generating `ntia-algorithm` `DFT` objects.

    :param id: A unique ID that may be referenced to associate a `Graph` with
        these parameters.
    :param equivalent_noise_bandwidth: Bandwidth of a brickwall filter that has
        the same integrated noise power as that of a sample of the DFT, in Hz.
    :param samples: Length of the DFT.
    :param dfts: Number of DFTs (each of length `samples`) integrated over by
        detectors, e.g. when using the Bartlett method or M4S detection.
    :param window: E.g., `"blackman-harris"`, `"flattop"`, `"gaussian_a3.5"`,
        `"gauss top"`, `"hamming"`, `"hanning"`, `"rectangular"`.
    :param baseband: Indicates whether or not the frequency axis described in
        the corresponding `Graph` object should be interpreted as baseband frequencies.
    :param description: Supplemental description of the processing.
    """

    id: str
    equivalent_noise_bandwidth: float
    samples: int
    dfts: int
    window: str
    baseband: bool
    description: Optional[str] = None
