from scos_actions.actions.metadata.metadata_decorator import Metadata
from scos_actions.actions.sigmf_builder import SigMFBuilder

class FftAnnotation(Metadata):

    def __init__(self,detector, sigmf_builder: SigMFBuilder, start, length):
        super().__init__(sigmf_builder, start, length)
        self.detector = detector

    def create_metadata(self, sigan_cal, sensor_cal, measurement_result):
        metadata = {
            "ntia-core:annotation_type": "FrequencyDomainDetection",
            "ntia-algorithm:number_of_samples_in_fft": measurement_result['fft_size'],
            "ntia-algorithm:window": measurement_result['window'],
            "ntia-algorithm:equivalent_noise_bandwidth": measurement_result['enbw'],
            "ntia-algorithm:detector": self.detector,
            "ntia-algorithm:number_of_ffts": measurement_result['nffts'],
            "ntia-algorithm:units": 'dBm',
            "ntia-algorithm:reference": '"preselector input"',
            "ntia-algorithm:frequency_start": measurement_result['frequency_start'],
            "ntia-algorithm:frequency_stop": measurement_result['frequency_stop'],
            "ntia-algorithm:frequency_step": measurement_result['frequency_step'],
        }
        self.sigmf_builder.add_annotation(self.start, measurement_result['fft_size'], metadata)