import logging
from abc import abstractmethod

from scos_actions.actions.interfaces.action import Action
from scos_actions.hardware.mocks.mock_gps import MockGPS
from scos_actions.metadata.sigmf_builder import SigMFBuilder
from scos_actions.signals import measurement_action_completed

logger = logging.getLogger(__name__)


class MeasurementAction(Action):
    """The MeasurementAction base class.

    To create an action, create a subclass of `Action` with a descriptive
    docstring and override the `__call__` method.

    """

    def __init__(self, parameters, sigan, gps=None):
        if gps is None:
            gps = MockGPS()
        super().__init__(parameters, sigan, gps)
        self.received_samples = 0

    def __call__(self, schedule_entry, task_id):
        self.test_required_components()
        self.configure(self.parameters)
        measurement_result = self.execute(schedule_entry, task_id)
        sigmf_builder = self.get_sigmf_builder(measurement_result, schedule_entry)
        self.create_metadata(schedule_entry, measurement_result)
        data = self.transform_data(measurement_result)
        self.send_signals(task_id, sigmf_builder.metadata, data)

    def get_sigmf_builder(
        self, measurement_result: dict, schedule_entry: dict
    ) -> SigMFBuilder:
        sigmf_builder = super().get_sigmf_builder(schedule_entry)
        sigmf_builder.set_sample_rate(measurement_result["sample_rate"])
        self.received_samples = len(measurement_result["data"].flatten())
        sigmf_builder.set_data_type(is_complex=self.is_complex())
        sigmf_builder.set_task(measurement_result["task_id"])
        sigmf_builder.set_classification(measurement_result["classification"])
        return sigmf_builder

    def create_metadata(
        self,
        schedule_entry: dict,
        measurement_result: dict,
        recording: int = None,
    ) -> SigMFBuilder:
        sigmf_builder = self.get_sigmf_builder(measurement_result, schedule_entry)

        if "calibration_datetime" in measurement_result:
            sigmf_builder.set_last_calibration_time(
                measurement_result["calibration_datetime"]
            )

        sigmf_builder.set_data_type(is_complex=self.is_complex())
        sigmf_builder.set_sample_rate(measurement_result["sample_rate"])
        sigmf_builder.set_task(measurement_result["task_id"])
        if recording is not None:
            sigmf_builder.set_recording(recording)

        sigmf_builder.build()
        return sigmf_builder

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

    def acquire_data(
        self, num_samples: int, nskip: int = 0, cal_adjust: bool = True
    ) -> dict:
        logger.debug(
            f"Acquiring {num_samples} IQ samples, skipping the first {nskip} samples"
            + f" and {'' if cal_adjust else 'not '}applying gain adjustment based"
            + " on calibration data"
        )
        measurement_result = self.sigan.acquire_time_domain_samples(
            num_samples,
            num_samples_skip=nskip,
            cal_adjust=cal_adjust,
        )

        return measurement_result

    def transform_data(self, measurement_result: dict):
        return measurement_result["data"]

    @abstractmethod
    def is_complex(self) -> bool:
        pass

    @abstractmethod
    def execute(self, schedule_entry, task_id):
        pass
