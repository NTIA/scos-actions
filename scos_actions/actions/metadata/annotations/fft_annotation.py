from scos_actions.actions.metadata.metadata import Metadata
from scos_actions.actions.metadata.sigmf_builder import SigMFBuilder


class FrequencyDomainDetectionAnnotation(Metadata):

    def __init__(self, detector, start, count):
        super().__init__(start, count)
        self.detector = detector

    def create_metadata(self, sigmf_builder: SigMFBuilder, measurement_result):
        metadata = {
            "ntia-core:annotation_type": "FrequencyDomainDetection",
            "ntia-algorithm:number_of_samples_in_fft": measurement_result['fft_size'],
            "ntia-algorithm:window": measurement_result['window'],
            "ntia-algorithm:equivalent_noise_bandwidth": measurement_result['enbw'],
            "ntia-algorithm:detector": 'fft_' + self.detector,
            "ntia-algorithm:number_of_ffts": measurement_result['nffts'],
            "ntia-algorithm:units": 'dBm',
            "ntia-algorithm:reference": '"preselector input"',
            "ntia-algorithm:frequency_start": measurement_result['frequency_start'],
            "ntia-algorithm:frequency_stop": measurement_result['frequency_stop'],
            "ntia-algorithm:frequency_step": measurement_result['frequency_step'],
        }
        sigmf_builder.add_annotation(self.start, measurement_result['fft_size'], metadata)
