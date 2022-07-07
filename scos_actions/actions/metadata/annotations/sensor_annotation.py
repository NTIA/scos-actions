from scos_actions.actions.metadata.metadata import Metadata
from scos_actions.actions.sigmf_builder import SigMFBuilder


class SensorAnnotation(Metadata):

    def __init__(self, start, count):
        super().__init__(start, count)

    def create_metadata(self, sigmf_builder: SigMFBuilder,
                        measurement_result: dict):
        metadata = {"ntia-core:annotation_type": "SensorAnnotation"}
        if 'overload' in measurement_result:
            metadata["ntia-sensor:overload"] = measurement_result['overload']
        if 'gain' in measurement_result:
            metadata["ntia-sensor:gain_setting_sigan"] = measurement_result['gain']
        if 'attenuation' in measurement_result:
            metadata["ntia-sensor:attenuation_setting_sigan"] = measurement_result['attenuation']
        sigmf_builder.add_annotation(self.start, self.count, metadata)
