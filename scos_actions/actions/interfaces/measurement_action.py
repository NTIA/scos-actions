import logging
from abc import abstractmethod

from scos_actions.actions.interfaces.action import Action
from scos_actions.hardware.mocks.mock_gps import MockGPS
from scos_actions.metadata.interfaces import ntia_scos
from scos_actions.metadata.sigmf_builder import SigMFBuilder
from scos_actions.signals import measurement_action_completed
from scos_actions.utils import get_value_if_exists

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
        sigmf_builder = self.get_sigmf_builder(measurement_result)
        self.create_metadata(sigmf_builder, schedule_entry, measurement_result)
        data = self.transform_data(measurement_result)
        self.send_signals(task_id, sigmf_builder.metadata, data)

    def get_sigmf_builder(self, measurement_result: dict) -> SigMFBuilder:
        sigmf_builder = SigMFBuilder()
        self._action_metadata_obj = ntia_scos.Action(
            name=self.name,
            description=self.description,
            summary=self.summary,
        )
        self.received_samples = len(measurement_result["data"].flatten())
        sigmf_builder.set_classification(measurement_result["classification"])
        return sigmf_builder

    def create_metadata(
        self,
        sigmf_builder: SigMFBuilder,
        schedule_entry: dict,
        measurement_result: dict,
        recording: int = None,
    ):
        schedule_entry_obj = ntia_scos.ScheduleEntry(
            schedule_entry["name"],  # name should be unique
            schedule_entry["name"],
            start=get_value_if_exists("start", schedule_entry),
            stop=get_value_if_exists("stop", schedule_entry),
            interval=get_value_if_exists("interval", schedule_entry),
            priority=get_value_if_exists("priority", schedule_entry),
            roles=get_value_if_exists("roles", schedule_entry),
        )
        action_obj = ntia_scos.Action(
            name=self.name,
            description=self.description,
            summary=self.summary,
        )

        sigmf_builder.set_base_sigmf_global(
            schedule_entry_obj,
            action_obj,
            self.sensor_definition,
            measurement_result,
            recording,
            self.is_complex(),
        )
        sigmf_builder.set_action(self._action_metadata_obj)
        sigmf_builder.build()

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
