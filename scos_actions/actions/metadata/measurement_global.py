from scos_actions.actions.metadata.metadata import Metadata
from scos_actions.actions.sigmf_builder import SigMFBuilder


class MeasurementMetadata(Metadata):

    def __init__(self, sigmf_builder: SigMFBuilder):
        super().__init__(sigmf_builder)

    def create_metadata(self, sigan_cal, sensor_cal, measurement_result):

        if 'frequency_low' in measurement_result:
            freq_low = measurement_result['frequency_low']
        elif 'frequency' in measurement_result:
            freq_low = measurement_result['frequency']
            freq_high = measurement_result['frequency']

        if 'frequency_high' in measurement_result:
            freq_high = measurement_result['frequency_high']

        self.sigmf_builder.add_to_global(
            "ntia-core:measurement",
            {
                "time_start": measurement_result['start_time'],
                "time_stop": measurement_result['end_time'],
                "domain": measurement_result['domain'],
                "measurement_type":measurement_result['measurement_type'],
                "frequency_tuned_low": freq_low,
                "frequency_tuned_high": freq_high
            },
        )