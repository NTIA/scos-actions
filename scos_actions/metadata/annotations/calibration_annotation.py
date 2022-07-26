from scos_actions.metadata.metadata import Metadata
from scos_actions.metadata.sigmf_builder import SigMFBuilder


class CalibrationAnnotation(Metadata):

    def __init__(self, start, count):
        super().__init__(start, count)

    def create_metadata(self, sigmf_builder:SigMFBuilder, measurement_result: dict):
        sigan_cal = measurement_result['sigan_cal']
        sensor_cal = measurement_result['sensor_cal']
        annotation = self.create_calibration_annotation(sigan_cal, sensor_cal)
        sigmf_builder.add_annotation(self.start, self.count, annotation)

    def create_calibration_annotation(self, sigan_cal, sensor_cal):
        """Create the SigMF calibration annotation."""
        annotation_md = {
            "ntia-core:annotation_type": "CalibrationAnnotation",
            "ntia-sensor:gain_sigan": sigan_cal["gain_sigan"],
            "ntia-sensor:gain_sensor": sensor_cal['gain_sensor'],
            "ntia-sensor:noise_figure_sigan": sigan_cal[
                "noise_figure_sigan"
            ],
            "ntia-sensor:1db_compression_point_sigan": sigan_cal[
                "1db_compression_sigan"
            ],
            "ntia-sensor:enbw_sigan": sigan_cal["enbw_sigan"],
            "ntia-sensor:gain_preselector": sensor_cal[
                "gain_preselector"
            ],
            "ntia-sensor:noise_figure_sensor": sensor_cal[
                "noise_figure_sensor"
            ],
            "ntia-sensor:1db_compression_point_sensor": sensor_cal[
                "1db_compression_sensor"
            ],
            "ntia-sensor:enbw_sensor": sensor_cal["enbw_sensor"]

        }
        if 'temperature' in sensor_cal:
            annotation_md["ntia-sensor:temperature"] =  sensor_cal["temperature"]

        return annotation_md
