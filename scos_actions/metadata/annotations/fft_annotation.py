from scos_actions.metadata.metadata import Metadata
from scos_actions.metadata.sigmf_builder import SigMFBuilder


class FrequencyDomainDetectionAnnotation(Metadata):
    def __init__(
        self,
        start: int,
        count: int,
        fft_size: int,
        window: str,
        enbw: float,
        detector: str,
        nffts: int,
        units: str,
        reference: str,
        frequency_start: float,
        frequency_stop: float,
        frequency_step: float,
    ):
        super().__init__(start, count)
        self.fft_size = fft_size
        self.window = window
        self.enbw = enbw
        self.detector = detector
        self.nffts = nffts
        self.units = units
        self.reference = reference
        self.frequency_start = frequency_start
        self.frequency_stop = frequency_stop
        self.frequency_step = frequency_step

    def create_metadata(self, sigmf_builder: SigMFBuilder, measurement_result: dict):
        metadata = {
            "ntia-core:annotation_type": "FrequencyDomainDetection",
            "ntia-algorithm:number_of_samples_in_fft": self.fft_size,
            "ntia-algorithm:window": self.window,
            "ntia-algorithm:equivalent_noise_bandwidth": self.enbw,
            "ntia-algorithm:detector": "fft_" + self.detector,
            "ntia-algorithm:number_of_ffts": self.nffts,
            "ntia-algorithm:units": self.units,
            "ntia-algorithm:reference": self.reference,
            "ntia-algorithm:frequency_start": self.frequency_start,
            "ntia-algorithm:frequency_stop": self.frequency_stop,
            "ntia-algorithm:frequency_step": self.frequency_step,
        }
        sigmf_builder.add_annotation(self.start, self.fft_size, metadata)
