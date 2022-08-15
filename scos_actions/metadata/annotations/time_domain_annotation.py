from scos_actions.metadata.metadata import Metadata
from scos_actions.metadata.sigmf_builder import SigMFBuilder


class TimeDomainAnnotation(Metadata):
    def __init__(
        self,
        start: int,
        count: int,
        detector: str = "sample_iq",
        units: str = "volts",
        reference: str = "preselector input",
    ):
        super().__init__(start, count)
        self.detector = detector
        self.units = units
        self.reference = reference

    def create_metadata(self, sigmf_builder: SigMFBuilder, measurement_result: dict):
        metadata = {
            "ntia-core:annotation_type": "TimeDomainDetection",
            "ntia-algorithm:detector": self.detector,
            "ntia-algorithm:number_of_samples": self.count,
            "ntia-algorithm:units": self.units,
            "ntia-algorithm:reference": self.reference,
        }
        sigmf_builder.add_annotation(self.start, self.count, metadata)
