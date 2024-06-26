import logging
from abc import abstractmethod
from typing import Optional

import numpy as np

from scos_actions.actions.interfaces.action import Action
from scos_actions.hardware.sensor import Sensor
from scos_actions.metadata.structs import ntia_sensor
from scos_actions.metadata.structs.capture import CaptureSegment
from scos_actions.signals import measurement_action_completed

logger = logging.getLogger(__name__)


class MeasurementAction(Action):
    """The MeasurementAction base class.

    To create an action, create a subclass of `Action` with a descriptive
    docstring and override the `__call__` method.

    """

    def __init__(self, parameters: dict):
        super().__init__(parameters)
        self.received_samples = 0

    def __call__(self, sensor: Sensor, schedule_entry: dict, task_id: int):
        self._sensor = sensor
        self.get_sigmf_builder(schedule_entry)
        self.test_required_components()
        self.configure(self.parameters)
        measurement_result = self.execute(schedule_entry, task_id)
        self.create_metadata(measurement_result)  # Fill metadata
        data = self.transform_data(measurement_result)
        self.send_signals(task_id, self.sigmf_builder.metadata, data)

    def create_capture_segment(
        self,
        sample_start: int,
        sigan_settings: Optional[ntia_sensor.SiganSettings],
        measurement_result: dict,
    ) -> CaptureSegment:
        capture_segment = CaptureSegment(
            sample_start=sample_start,
            frequency=measurement_result["frequency"],
            datetime=measurement_result["capture_time"],
            duration=measurement_result["duration_ms"],
            overload=measurement_result["overload"],
            sigan_settings=sigan_settings,
        )
        # Set calibration metadata if it exists
        cal_meta = self.get_calibration(measurement_result)
        if cal_meta is not None:
            capture_segment.sensor_calibration = cal_meta
        return capture_segment

    def get_calibration(self, measurement_result: dict) -> ntia_sensor.Calibration:
        cal_meta = None
        if (
            self.sensor.sensor_calibration_data is not None
            and measurement_result["applied_calibration"] is not None
        ):
            cal_meta = ntia_sensor.Calibration(
                datetime=self.sensor.sensor_calibration_data["datetime"],
                gain=round(measurement_result["applied_calibration"]["gain"], 3),
                noise_figure=round(
                    measurement_result["applied_calibration"]["noise_figure"], 3
                ),
                reference=measurement_result["reference"],
            )
            if "compression_point" in measurement_result["applied_calibration"]:
                cal_meta.compression_point = measurement_result["applied_calibration"][
                    "compression_point"
                ]
            if "temperature" in self.sensor.sensor_calibration_data:
                cal_meta.temperature = round(
                    self.sensor.sensor_calibration_data["temperature"], 1
                )
        return cal_meta

    def create_metadata(
        self,
        measurement_result: dict,
        recording: Optional[int] = None,
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
            cap = measurement_result["capture_segment"]
            logger.debug(f"Adding capture:{cap}")
            self.sigmf_builder.add_capture(measurement_result["capture_segment"])
        except KeyError:
            logger.warning(warning_str.format("capture_segment"))

        # Set data type metadata using is_complex method
        # This assumes data is 32-bit little endian floating point
        self.sigmf_builder.set_data_type(is_complex=self.is_complex())

        # Set the recording, if provided
        if recording is not None:
            self.sigmf_builder.set_recording(recording)

    def get_sigan_settings(
        self, measurement_result: dict
    ) -> Optional[ntia_sensor.SiganSettings]:
        """
        Retrieve any sigan settings from the measurement result dict, and return
        a `ntia-sensor` `SiganSettings` object. Values are pulled from the
        `measurement_result` dict if their keys match the names of fields in
        the `SiganSettings` object. If no matches are made, `None` is returned.
        """
        sigan_settings = {
            k: v
            for k, v in measurement_result.items()
            if k in ntia_sensor.SiganSettings.__struct_fields__
        }
        if sigan_settings == {}:
            sigan_settings = None
        else:
            sigan_settings = ntia_sensor.SiganSettings(**sigan_settings)
        return sigan_settings

    def test_required_components(self):
        """Fail acquisition if a required component is not available."""
        if not self.sensor.signal_analyzer.is_available:
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
        self,
        num_samples: int,
        nskip: int = 0,
        cal_adjust: bool = True,
        cal_params: Optional[dict] = None,
    ) -> dict:
        logger.debug(
            f"Acquiring {num_samples} IQ samples, skipping the first {nskip} samples"
            + f" and {'' if cal_adjust else 'not '}applying gain adjustment based"
            + " on calibration data"
        )
        measurement_result = self.sensor.acquire_time_domain_samples(
            num_samples,
            num_samples_skip=nskip,
            cal_adjust=cal_adjust,
            cal_params=cal_params,
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
