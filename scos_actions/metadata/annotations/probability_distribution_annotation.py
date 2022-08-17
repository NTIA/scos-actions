from scos_actions.metadata.metadata import Metadata
from scos_actions.metadata.sigmf_builder import SigMFBuilder


class ProbabilityDistributionAnnotation(Metadata):
    def __init__(
        self,
        start: int,
        count: int,
        function: str,
        units: str,
        probability_units: str,
    ):
        super().__init__(start, count)
        self.detector = detector
        self.num_samps = num_samps
        self.units = units
        self.reference = reference

    def create_metadata(self, sigmf_builder: SigMFBuilder, measurement_result: dict):
        metadata = {
            "ntia-core:annotation_type": "ProbabilityDistributionAnnotation",
            "ntia-algorithm:function": self.function,
            "ntia-algorithm:units": self.units,
            "ntia-algorithm:probability_units": self.probability_units,
            "ntia-algorithm:number_of_samples": self.num_samps,
            "ntia-algorithm:reference": self.reference,
            "ntia-algorithm:probability_start": self.probability_start,
            "ntia-algorithm:probability_stop": self.probability_stop,
            "ntia-algorithm:probabilities": self.probabilities,
            "ntia-algorithm:downsampled": self.downsampled,
            "ntia-algorithm:downsampling_method": self.downsampling_method,
        }
        sigmf_builder.add_annotation(self.start, self.count, metadata)
