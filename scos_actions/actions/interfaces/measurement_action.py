import logging
from abc import abstractmethod

from scos_actions.actions.interfaces.action import Action
from scos_actions.hardware.mocks.mock_gps import MockGPS
from scos_actions.metadata.annotations import CalibrationAnnotation, SensorAnnotation
from scos_actions.metadata.measurement_global import MeasurementMetadata
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
        sigmf_builder = self.get_sigmf_builder(measurement_result)
        self.create_metadata(sigmf_builder, schedule_entry, measurement_result)
        data = self.transform_data(measurement_result)
        self.send_signals(task_id, sigmf_builder.metadata, data)

    def get_sigmf_builder(self, measurement_result) -> SigMFBuilder:
        sigmf_builder = SigMFBuilder()
        self.received_samples = len(measurement_result["data"].flatten())
        calibration_annotation = CalibrationAnnotation(
            sample_start=0,
            sample_count=self.received_samples,
            sigan_cal=measurement_result["sigan_cal"],
            sensor_cal=measurement_result["sensor_cal"],
        )
        sigmf_builder.add_metadata_generator(
            type(calibration_annotation).__name__, calibration_annotation
        )
        f_low, f_high = None, None
        if "frequency_low" in measurement_result:
            f_low = measurement_result["frequency_low"]
        elif "frequency" in measurement_result:
            f_low = measurement_result["frequency"]
            f_high = measurement_result["frequency"]
        if "frequency_high" in measurement_result:
            f_high = measurement_result["frequency_high"]

        measurement_metadata = MeasurementMetadata(
            domain=measurement_result["domain"],
            measurement_type=measurement_result["measurement_type"],
            time_start=measurement_result["start_time"],
            time_stop=measurement_result["end_time"],
            frequency_tuned_low=f_low,
            frequency_tuned_high=f_high,
            classification=measurement_result["classification"],
        )
        sigmf_builder.add_metadata_generator(
            type(measurement_metadata).__name__, measurement_metadata
        )

        sensor_annotation = SensorAnnotation(
            sample_start=0,
            sample_count=self.received_samples,
            overload=measurement_result["overload"]
            if "overload" in measurement_result
            else None,
            attenuation_setting_sigan=measurement_result["attenuation"]
            if "attenuation" in measurement_result
            else None,
            gain_setting_sigan=measurement_result["gain"]
            if "gain" in measurement_result
            else None,
        )
        sigmf_builder.add_metadata_generator(
            type(sensor_annotation).__name__, sensor_annotation
        )
        return sigmf_builder

    def create_metadata(
        self, sigmf_builder, schedule_entry, measurement_result, recording=None
    ):
        sigmf_builder.set_base_sigmf_global(
            schedule_entry,
            self.sensor_definition,
            measurement_result,
            recording,
            self.is_complex(),
        )
        sigmf_builder.add_sigmf_capture(sigmf_builder, measurement_result)
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
        self, num_samples: int, nskip: int = 0, gain_adjust: bool = True
    ) -> dict:
        logger.debug(
            f"Acquiring {num_samples} IQ samples, skipping the first {nskip} samples"
            + f" and {'' if gain_adjust else 'not '}applying gain adjustment based"
            + " on calibration data"
        )
        measurement_result = self.sigan.acquire_time_domain_samples(
            num_samples,
            num_samples_skip=nskip,
            gain_adjust=gain_adjust,
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
