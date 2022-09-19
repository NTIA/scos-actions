from dataclasses import dataclass
from typing import List, Optional

from scos_actions.metadata.annotation_segment import AnnotationSegment


@dataclass
class FrequencyDomainDetection(AnnotationSegment):
    """
    Interface for generating FrequencyDomainDetection annotation segments.

    Refer to the documentation of the ``ntia-algorithm`` extension of SigMF
    for more information.

    The parameters ``detector``, ``number_of_ffts``, ``number_of_samples_in_fft``,
    ``window``, and ``units`` are required.

    :param detector: Detector type, e.g. ``"fft_sample_iq"``, ``"fft_sample_power"``,
        ``"fft_mean_power"``, etc. If the detector string does not start with "fft_"
        already, "fft_" will be prepended to the input detector string.
    :param number_of_ffts: Number of FFTs to be integrated over by detector.
    :param number_of_samples_in_fft: Number of samples in FFT to calculate
        ``delta_f = samplerate / number_of_samples_in_fft``.
    :param window: Window type used in FFT, e.g. ``"blackman-harris"``, ``"flattop"``,
        ``"hanning"``, ``"rectangular"``, etc.
    :param equivalent_noise_bandwidth: Bandwidth of brickwall filter that has the
        same integrated noise power as that of the actual filter.
    :param units: Data units, e.g. ``"dBm"``, ``"watts"``, ``"volts"``.
    :param reference: Data reference point, e.g. ``"signal analyzer input"``,
        ``"preselector input"``, ``"antenna terminal"``.
    :param frequency_start: Frequency (Hz) of first data point.
    :param frequency_stop: Frequency (Hz) of last data point.
    :param frequency_step: Frequency step size (Hz) between data points.
    :param frequencies: A list of the frequencies (Hz) of the data points.
    """

    detector: Optional[str] = None
    number_of_ffts: Optional[int] = None
    number_of_samples_in_fft: Optional[int] = None
    window: Optional[str] = None
    equivalent_noise_bandwidth: Optional[float] = None
    units: Optional[str] = None
    reference: Optional[str] = None
    frequency_start: Optional[float] = None
    frequency_stop: Optional[float] = None
    frequency_step: Optional[float] = None
    frequencies: Optional[List[float]] = None

    def __post_init__(self):
        super().__post_init__()
        # Ensure required keys have been set
        self.check_required(self.detector, "detector")
        self.check_required(self.number_of_ffts, "number_of_ffts")
        self.check_required(self.number_of_samples_in_fft, "number_of_samples_in_fft")
        self.check_required(self.window, "window")
        self.check_required(self.units, "units")
        # Prepend "fft" to detector name if needed
        if self.detector[:4] != "fft_":
            self.detector = "fft_" + self.detector
        # Define SigMF key names
        self.sigmf_keys.update(
            {
                "detector": "ntia-algorithm:detector",
                "number_of_ffts": "ntia-algorithm:number_of_ffts",
                "number_of_samples_in_fft": "ntia-algorithm:number_of_samples_in_fft",
                "window": "ntia-algorithm:window",
                "equivalent_noise_bandwidth": "ntia-algorithm:equivalent_noise_bandwidth",
                "units": "ntia-algorithm:units",
                "reference": "ntia-algorithm:reference",
                "frequency_start": "ntia-algorithm:frequency_start",
                "frequency_stop": "ntia-algorithm:frequency_stop",
                "frequency_step": "ntia-algorithm:frequency_step",
                "frequencies": "ntia-algorithm:frequencies",
            }
        )
        # Create annotation segment
        super().create_annotation_segment()
