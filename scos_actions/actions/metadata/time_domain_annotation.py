from scos_actions.actions.metadata.metadata import Metadata
from scos_actions.actions.sigmf_builder import SigMFBuilder

class TimeDomainAnnotation(Metadata):

    def __init__(self, sigmf_builder:SigMFBuilder, start, length):
        super().__init__(sigmf_builder, start, length)

    def create_metadata(self, sigan_cal, sensor_cal, measurement_result):
        time_domain_detection_md = {
            "ntia-core:annotation_type": "TimeDomainDetection",
            "ntia-algorithm:detector": "sample_iq",
            "ntia-algorithm:number_of_samples": self.length,
            "ntia-algorithm:units": "volts",
            "ntia-algorithm:reference": "preslector input",
        }
        self.sigmf_builder.add_annotation(self.start, self.length, time_domain_detection_md)
