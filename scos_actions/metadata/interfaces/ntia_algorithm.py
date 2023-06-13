from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union

from scos_actions.metadata.interfaces.sigmf_object import SigMFObject


class FilterType(str, Enum):
    IIR = "IIR"
    FIR = "FIR"


@dataclass
class DigitalFilter(SigMFObject):
    """
    Interface for generating `ntia-algorithm` `DigitalFilter` objects.

    The `id` and `filter_type` parameters are required.

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

    id: Optional[str] = None
    filter_type: Optional[FilterType] = None
    feedforward_coefficients: Optional[List[float]] = None
    feedback_coefficients: Optional[List[float]] = None
    attenuation_cutoff: Optional[float] = None
    frequency_cutoff: Optional[float] = None
    description: Optional[float] = None

    def __post_init__(self):
        super().__post_init__()
        # Ensure required keys have been set
        self.check_required(self.id, "id")
        self.check_required(self.filter_type, "filter_type")
        # Define SigMF key names
        self.obj_keys.update(
            {
                "id": "id",
                "filter_type": "filter_type",
                "feedforward_coefficients": "feedforward_coefficients",
                "feedback_coefficients": "feedback_coefficients",
                "attenuation_cutoff": "attenuation_cutoff",
                "frequency_cutoff": "frequency_cutoff",
                "description": "description",
            }
        )
        # Create metadata object
        super().create_json_object()


@dataclass
class Graph(SigMFObject):
    """
    Interface for generating `ntia-algorithm` `Graph` objects.

    The `name` parameter is required.

    :param name: The name of the graph, e.g. `"Power Spectral Density"`,
        `"Signal Power vs. Time"`, `"M4S Detection"`
    :param series:
    """

    name: Optional[str] = None
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

    def __post_init__(self):
        super().__post_init__()
        # Ensure required keys have been set
        self.check_required(self.name, "name")
        # Define SigMF key names
        self.obj_keys.update(
            {
                "name": "name",
                "series": "series",
                "length": "length",
                "x_units": "x_units",
                "x_axis": "x_axis",
                "x_start": "x_start",
                "x_stop": "x_stop",
                "x_step": "x_step",
                "y_units": "y_units",
                "y_axis": "y_axis",
                "y_start": "y_start",
                "y_stop": "y_stop",
                "y_step": "y_step",
                "processing": "processing",
                "reference": "processing",
                "description": "description",
            }
        )
        # Create metadata object
        super().create_json_object()


@dataclass
class DFT(SigMFObject):
    """
    Interface for generating `ntia-algorithm` `DFT` objects.

    The `id`, `equivalent_noise_bandwidth`, `samples`, `dfts`, `window`,
    and `baseband` parameters are required.

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

    id: Optional[str] = None
    equivalent_noise_bandwidth: Optional[float] = None
    samples: Optional[int] = None
    dfts: Optional[int] = None
    window: Optional[str] = None
    baseband: Optional[bool] = None
    description: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        # Ensure required keys have been set
        self.check_required(self.id, "id")
        self.check_required(
            self.equivalent_noise_bandwidth, "equivalent_noise_bandwidth"
        )
        self.check_required(self.samples, "samples")
        self.check_required(self.dfts, "dfts")
        self.check_required(self.window, "window")
        self.check_required(self.baseband, "baseband")
        # Define SigMF key names
        self.obj_keys.update(
            {
                "id": "id",
                "equivalent_noise_bandwidth": "equivalent_noise_bandwidth",
                "samples": "samples",
                "dfts": "dfts",
                "window": "window",
                "baseband": "baseband",
                "description": "description",
            }
        )
        # Create metadata object
        super().create_json_object()
