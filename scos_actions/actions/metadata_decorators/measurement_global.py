from scos_actions.actions.metadata_decorators.metadata_decorator import Metadata
from scos_actions.actions.sigmf_builder import SigMFBuilder


class MeasurementMetadata(Metadata):

    def __init__(self, sigmf_builder: SigMFBuilder):
        super().__init__(sigmf_builder)

    def create_metadata(self, sigan_cal, sensor_cal, measurement_result):
        self.sigmf_builder.add_to_global(
            "ntia-core:measurement",
            {
                "time_start": measurement_result['start_time'],
                "time_stop": measurement_result['end_time'],
                "domain": measurement_result['domain'],
                "measurement_type":measurement_result['measurement_type'],
                "frequency_tuned_low": measurement_result['frequency_low'],
                "frequency_tuned_high": measurement_result['frequency_hight']
            },
        )
