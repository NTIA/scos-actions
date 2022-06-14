from scos_actions.actions.metadata.metadata import Metadata
from scos_actions.actions.sigmf_builder import SigMFBuilder

class SensorAnnotation(Metadata):

    def __init__(self, sigmf_builder: SigMFBuilder, start, length):
        super().__init__(sigmf_builder, start, length)

    def create_metadata(self, sigan_cal, sensor_cal, measurement_result):
        metadata = {"ntia-core:annotation_type": "SensorAnnotation"}
        if 'overload' in measurement_result:
            metadata["ntia-sensor:overload"] = measurement_result['overload']
        if 'gain' in measurement_result:
            metadata["ntia-sensor:gain_setting_sigan"] = measurement_result['gain']
        if 'attenuation' in measurement_result:
            metadata["ntia-sensor:attenuation_setting_sigan"] = measurement_result['attenuation']
        self.sigmf_builder.add_annotation(self.start, self.length, metadata)


