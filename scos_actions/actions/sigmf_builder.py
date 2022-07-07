from enum import Enum
from sigmf import SigMFFile

GLOBAL_INFO = {
    "core:version": "0.0.2",
    "core:extensions": {
        "ntia-algorithm": "v1.0.0",
        "ntia-core": "v1.0.0",
        "ntia-environment": "v1.0.0",
        "ntia-location": "v1.0.0",
        "ntia-scos": "v1.0.0",
        "ntia-sensor": "v1.0.0",
    },
}


def get_coordinate_system_sigmf():
    return {
        "id": "WGS 1984",
        "coordinate_system_type": "GeographicCoordinateSystem",
        "distance_unit": "decimal degrees",
        "time_unit": "seconds",
    }


class MeasurementType(Enum):
    SINGLE_FREQUENCY = "single-frequency"
    SCAN = "scan"


class Domain(Enum):
    FREQUENCY = "frequency"
    TIME = "time"


class SigMFBuilder:
    def __init__(self):
        self.sigmf_md = SigMFFile()
        self.sigmf_md.set_global_info(GLOBAL_INFO.copy())
        self.metadata_generators = {}

    def reset(self):
        self.sigmf_md = SigMFFile()
        self.sigmf_md.set_global_info(GLOBAL_INFO.copy())

    @property
    def metadata(self):
        return self.sigmf_md._metadata

    def set_data_type(self, is_complex):
        if is_complex:
            self.sigmf_md.set_global_field(
                "core:datatype", "cf32_le"
            )  # 2x 32-bit float, Little Endian
        else:
            self.sigmf_md.set_global_field(
                "core:datatype", "rf32_le"
            )  # 32-bit float, Little Endian

    def set_sample_rate(self, sample_rate):
        self.sigmf_md.set_global_field("core:sample_rate", sample_rate)

    def set_recording(self, recording_id):
        self.sigmf_md.set_global_field("ntia-scos:recording", recording_id)

    def set_measurement(
            self, start_time, end_time, domain, measurement_type, frequency
    ):
        self.sigmf_md.set_global_field(
            "ntia-core:measurement",
            {
                "time_start": start_time,
                "time_stop": end_time,
                "domain": domain.value,
                "measurement_type": measurement_type.value,
                "frequency_tuned_low": frequency,
                "frequency_tuned_high": frequency,
            },
        )

    def set_sensor(self, sensor):
        self.sigmf_md.set_global_field("ntia-sensor:sensor", sensor)

    def set_task(self, task_id):
        self.sigmf_md.set_global_field("ntia-scos:task", task_id)

    def set_action(self, name, description, summary):
        self.sigmf_md.set_global_field(
            "ntia-scos:action",
            {
                "name": name,
                "description": description,
                "summary": summary,
            },
        )

    def set_schedule(self, schedule_entry_json):
        self.sigmf_md.set_global_field("ntia-scos:schedule", schedule_entry_json)

    def set_coordinate_system(self, coordinate_system=get_coordinate_system_sigmf()):
        self.sigmf_md.set_global_field(
            "ntia-location:coordinate_system", coordinate_system
        )

    def set_capture(self, frequency, capture_time):
        capture_md = {
            "core:frequency": frequency,
            "core:datetime": capture_time,
        }

        self.sigmf_md.add_capture(start_index=0, metadata=capture_md)

    def add_annotation(self, start_index, length, annotation_md):
        self.sigmf_md.add_annotation(
            start_index=start_index, length=length, metadata=annotation_md
        )

    def set_last_calibration_time(self, last_cal_time):
        self.sigmf_md.set_global_field(
            "ntia-sensor:calibration_datetime", last_cal_time
        )

    def add_to_global(self, key, value):
        self.sigmf_md.set_global_field(key, value)

    def set_base_sigmf_global(
            self,
            schedule_entry_json,
            sensor_def,
            measurement_result,
            recording_id=None,
            is_complex=True,
    ):
        sample_rate = measurement_result["sample_rate"]
        if 'calibration_datetime' in measurement_result:
            self.set_last_calibration_time(measurement_result['calibration_datetime'])

        description = measurement_result['description']
        self.set_action(
            measurement_result["name"], description, description.splitlines()[0]
        )
        self.set_coordinate_system()
        self.set_data_type(is_complex=is_complex)
        self.set_sample_rate(sample_rate)
        self.set_schedule(schedule_entry_json)
        self.set_sensor(sensor_def)
        self.set_task(measurement_result['task_id'])
        if recording_id:
            self.set_recording(recording_id)

    def add_sigmf_capture(self, sigmf_builder, measurement_result):
        sigmf_builder.set_capture(
            measurement_result["frequency"], measurement_result["capture_time"]
        )

    def add_metadata_generator(self, key, generator):
        self.metadata_generators[key] = generator

    def remove_metadata_generator(self, key):
        self.metadata_generators.pop(key, '')

    def build(self, measurement_result):
        for metadata_creator in self.metadata_generators.values():
            metadata_creator.create_metadata(self, measurement_result)
