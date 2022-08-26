from scos_actions.metadata.metadata import Metadata
from scos_actions.metadata.sigmf_builder import SigMFBuilder


class TimeDomainAnnotation(Metadata):
    def __init__(self, start, count):
        super().__init__(start, count)

    def create_metadata(self, sigmf_builder: SigMFBuilder, measurement_result):
        time_domain_detection_md = {
            "ntia-core:annotation_type": "TimeDomainDetection",
            "ntia-algorithm:detector": "sample_iq",
            "ntia-algorithm:number_of_samples": self.count,
            "ntia-algorithm:units": "volts",
            "ntia-algorithm:reference": "preselector input",
        }
        sigmf_builder.add_annotation(self.start, self.count, time_domain_detection_md)
