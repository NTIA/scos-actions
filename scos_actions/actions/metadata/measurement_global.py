from scos_actions.actions.metadata.metadata import Metadata
from scos_actions.actions.sigmf_builder import SigMFBuilder


class MeasurementMetadata(Metadata):

    def __init__(self):
        super().__init__()

    def create_metadata(self, sigmf_builder: SigMFBuilder, measurement_result: dict):
        freq_low = None
        freq_high = None

        if 'frequency_low' in measurement_result:
            freq_low = measurement_result['frequency_low']
        elif 'frequency' in measurement_result:
            freq_low = measurement_result['frequency']
            freq_high = measurement_result['frequency']
        if 'frequency_high' in measurement_result:
            freq_high = measurement_result['frequency_high']

        if freq_high is None:
            raise Exception('frequency_high is a required measurement metadata value.')
        if freq_low is None:
            raise Exception('frequency_low is a required measurement metadata value.')

        sigmf_builder.add_to_global(
            "ntia-core:measurement",
            {
                "time_start": measurement_result['start_time'],
                "time_stop": measurement_result['end_time'],
                "domain": measurement_result['domain'],
                "measurement_type": measurement_result['measurement_type'],
                "frequency_tuned_low": freq_low,
                "frequency_tuned_high": freq_high
            },
        )
