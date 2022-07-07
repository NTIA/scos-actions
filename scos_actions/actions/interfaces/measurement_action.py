import logging

from scos_actions.actions.interfaces.action import Action
from scos_actions.hardware import gps as mock_gps
from scos_actions.hardware import sigan as mock_sigan
from scos_actions.actions.interfaces.signals import measurement_action_completed
from scos_actions.actions.metadata.annotations.calibration_annotation import CalibrationAnnotation
from scos_actions.actions.metadata.measurement_global import MeasurementMetadata
from scos_actions.actions.metadata.annotations.sensor_annotation import SensorAnnotation
from scos_actions.actions.sigmf_builder import SigMFBuilder

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
        sigmf_builder = self.get_sigmf_builder(measurement_result)
        self.create_metadata(sigmf_builder, schedule_entry, measurement_result)
        data = self.transform_data(measurement_result)
        self.send_signals(task_id, sigmf_builder.metadata, data)

    def get_sigmf_builder(self, measurement_result) -> SigMFBuilder:
        sigmf_builder = SigMFBuilder()
        self.received_samples = len(measurement_result["data"].flatten())
        calibration_annotation = CalibrationAnnotation( 0, self.received_samples)
        sigmf_builder.add_metadata_generator(type(calibration_annotation).__name__, calibration_annotation)
        measurement_metadata = MeasurementMetadata()
        sigmf_builder.add_metadata_generator(type(measurement_metadata).__name__, measurement_metadata)
        sensor_annotation = SensorAnnotation( 0, self.received_samples)
        sigmf_builder.add_metadata_generator(type(sensor_annotation).__name__, sensor_annotation)
        return sigmf_builder

    def create_metadata(self, sigmf_builder, schedule_entry, measurement_result, recording=None):

        sigmf_builder.set_base_sigmf_global(
            schedule_entry,
            self.sensor_definition,
            measurement_result, recording, self.is_complex
        )
        sigmf_builder.add_sigmf_capture(sigmf_builder, measurement_result)
        sigmf_builder.build( measurement_result)

    def test_required_components(self):
        """Fail acquisition if a required component is not available."""
        if not self.sigan.is_available:
            msg = "acquisition failed: signal analyzer required but not available"
            raise RuntimeError(msg)

    def send_signals(self, task_id, metadata, measurement_data):
        measurement_action_completed.send(
            sender=self.__class__,
            task_id=task_id,
            data=measurement_data,
            metadata=metadata,
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