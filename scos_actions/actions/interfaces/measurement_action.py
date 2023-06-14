import logging
from abc import abstractmethod

import numpy as np

from scos_actions.actions.interfaces.action import Action
from scos_actions.hardware.mocks.mock_gps import MockGPS
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

    def __call__(self, schedule_entry: dict, task_id: int):
        self.test_required_components()
        self.configure(self.parameters)
        measurement_result = self.execute(schedule_entry, task_id)
        self.get_sigmf_builder(schedule_entry)  # Initializes SigMFBuilder
        self.create_metadata(measurement_result)  # Fill metadata
        data = self.transform_data(measurement_result)
        self.send_signals(task_id, self.sigmf_builder.metadata, data)

    def create_metadata(
        self,
        measurement_result: dict,
        recording: int = None,
    ) -> None:
        """Add SigMF metadata to the `sigmf_builder` from the `measurement_result`."""
        # Set the received_samples instance variable
        if "data" in measurement_result:
            if isinstance(measurement_result["data"], np.ndarray):
                self.received_samples = len(measurement_result["data"].flatten())
            else:
                try:
                    self.received_samples = len(measurement_result["data"])
                except TypeError:
                    logger.warning(
                        "Failed to get received sample count from measurement result."
                    )
        else:
            logger.warning(
                "Failed to get received sample count from measurement result."
            )

        # Fill metadata fields using the measurement result
        warning_str = "Measurement result is missing a '{}' value"
        try:
            self.sigmf_builder.set_sample_rate(measurement_result["sample_rate"])
        except KeyError:
            logger.warning(warning_str.format("sample_rate"))
        try:
            self.sigmf_builder.set_task(measurement_result["task_id"])
        except KeyError:
            logger.warning(warning_str.format("task_id"))
        try:
            self.sigmf_builder.set_classification(measurement_result["classification"])
        except KeyError:
            logger.warning(warning_str.format("classification"))
        try:
            self.sigmf_builder.set_last_calibration_time(
                measurement_result["calibration_datetime"]
            )
        except KeyError:
            logger.warning(warning_str.format("calibration_datetime"))

        # Set data type metadata using is_complex method
        # This assumes data is 32-bit little endian floating point
        self.sigmf_builder.set_data_type(is_complex=self.is_complex())

        # Set the recording, if provided
        if recording is not None:
            self.sigmf_builder.set_recording(recording)

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
