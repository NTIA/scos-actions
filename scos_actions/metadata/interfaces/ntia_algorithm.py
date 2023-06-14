from enum import Enum
from typing import List, Optional, Union

import msgspec

from scos_actions.metadata.utils import SIGMF_OBJECT_KWARGS


class FilterType(str, Enum):
    IIR = "IIR"
    FIR = "FIR"


class DigitalFilter(msgspec.Struct, **SIGMF_OBJECT_KWARGS):
    """
    Interface for generating `ntia-algorithm` `DigitalFilter` objects.

    :param id: Unique ID of the filter.
    :param filter_type: Type of the digital fitler, given by the `FilterType`
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

    id: Optional[str]
    filter_type: Optional[FilterType]
    feedforward_coefficients: Optional[List[float]] = None
    feedback_coefficients: Optional[List[float]] = None
    attenuation_cutoff: Optional[float] = None
    frequency_cutoff: Optional[float] = None
    description: Optional[float] = None


class Graph(msgspec.Struct, **SIGMF_OBJECT_KWARGS):
    """
    Interface for generating `ntia-algorithm` `Graph` objects.

    :param name: The name of the graph, e.g. `"Power Spectral Density"`,
        `"Signal Power vs. Time"`, `"M4S Detection"`
    :param series:
    """

    name: Optional[str]
    series: Optional[List[str]] = None
    length: Optional[int] = None
    x_units: Optional[str] = None
    x_axis: Optional[List[Union[int, float, str]]] = None
    x_start: Optional[List[float]] = None
    x_stop: Optional[List[float]] = None
    x_step: Optional[List[float]] = None
    y_units: Optional[str] = None
    y_axis: Optional[List[Union[int, float, str]]] = None
    y_start: Optional[List[float]] = None
    y_stop: Optional[List[float]] = None
    y_step: Optional[List[float]] = None
    processing: Optional[List[str]] = None
    reference: Optional[str] = None
    description: Optional[str] = None


class DFT(msgspec.Struct, **SIGMF_OBJECT_KWARGS):
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

    id: Optional[str]
    equivalent_noise_bandwidth: Optional[float]
    samples: Optional[int]
    dfts: Optional[int]
    window: Optional[str]
    baseband: Optional[bool]
    description: Optional[str] = None
