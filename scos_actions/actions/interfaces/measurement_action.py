import logging

from scos_actions.actions.interfaces.action import Action
from scos_actions.hardware import gps as mock_gps
from scos_actions.hardware import sigan as mock_sigan
from scos_actions.actions.interfaces.signals import measurement_action_completed
from scos_actions.actions.metadata.annotations.calibration_annotation import CalibrationAnnotation
from scos_actions.actions.metadata.measurement_global import MeasurementMetadata
from scos_actions.actions.metadata.annotations.sensor_annotation import SensorAnnotation


logger = logging.getLogger(__name__)


class MeasurementAction(Action):
    """The MeasurementAction base class.

    To create an action, create a subclass of `Action` with a descriptive
    docstring and override the `__call__` method.

    """

    def __init__(self, parameters, sigan=mock_sigan, gps=mock_gps):
        super().__init__(parameters, sigan, gps)
        self.received_samples = 0

    def __call__(self, schedule_entry, task_id):
        self.test_required_components()
        self.configure(self.parameters)
        measurement_result = self.execute(schedule_entry, task_id)
        self.add_metadata_generators(measurement_result)
        self.create_metadata(schedule_entry, measurement_result)
        data = self.transform_data(measurement_result)
        self.send_signals(task_id, data)

    def add_metadata_generators(self, measurement_result):
        self.received_samples = len(measurement_result["data"].flatten())
        calibration_annotation = CalibrationAnnotation(self.sigmf_builder, 0, self.received_samples)
        self.metadata_generators[type(calibration_annotation).__name__] = calibration_annotation
        measurement_metadata = MeasurementMetadata(self.sigmf_builder)
        self.metadata_generators[type(measurement_metadata).__name__] = measurement_metadata
        sensor_annotation = SensorAnnotation(self.sigmf_builder, 0, self.received_samples)
        self.metadata_generators[type(sensor_annotation).__name__] = sensor_annotation

    def create_metadata(self, schedule_entry, measurement_result, recording=None):
        self.sigmf_builder.set_base_sigmf_global(
            schedule_entry,
            self.sensor_definition,
            measurement_result, recording, self.is_complex
        )
        self.sigmf_builder.add_sigmf_capture(self.sigmf_builder, measurement_result)
        for metadata_creator in self.metadata_generators.values():
            metadata_creator.create_metadata(self.sigan.sigan_calibration_data, self.sigan.sensor_calibration_data,
                                             measurement_result)

    def test_required_components(self):
        """Fail acquisition if a required component is not available."""
        if not self.sigan.is_available:
            msg = "acquisition failed: signal analyzer required but not available"
            raise RuntimeError(msg)



    def send_signals(self, task_id, measurement_data):
        measurement_action_completed.send(
            sender=self.__class__,
            task_id=task_id,
            data=measurement_data,
            metadata=self.sigmf_builder.metadata,
        )

    def acquire_data(self, num_samples, nskip):

        logger.debug(
            f"acquiring {num_samples} samples and skipping the first {nskip if nskip else 0} samples"
        )
        measurement_result = self.sigan.acquire_time_domain_samples(
            num_samples, num_samples_skip=nskip
        )

        return measurement_result

    def transform_data(self, measurement_result):
        return measurement_result["data"]
