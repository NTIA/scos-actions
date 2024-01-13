import logging
from abc import abstractmethod
from typing import Union

import numpy as np

from scos_actions.actions.interfaces.action import Action
from scos_actions.metadata.structs import ntia_sensor
from scos_actions.metadata.structs.capture import CaptureSegment
from scos_actions.signals import measurement_action_completed

logger = logging.getLogger(__name__)


class MeasurementAction(Action):
    """The MeasurementAction base class.

    To create an action, create a subclass of `Action` with a descriptive
    docstring and override the `__call__` method.

    """

    def __init__(self, parameters):
        super().__init__(parameters)
        self.received_samples = 0

    def __call__(self, sensor, schedule_entry: dict, task_id: int):
        self._sensor = sensor
        self.get_sigmf_builder(sensor, schedule_entry)
        self.test_required_components()
        self.configure(self.parameters)
        measurement_result = self.execute(schedule_entry, task_id)
        self.create_metadata(measurement_result)  # Fill metadata
        data = self.transform_data(measurement_result)
        self.send_signals(task_id, self.sigmf_builder.metadata, data)

    def create_capture_segment(
        self,
        sample_start: int,
        start_time: str,
        center_frequency_Hz: float,
        duration_ms: int,
        overload: bool,
        sigan_settings: Union[ntia_sensor.SiganSettings, None],
    ) -> CaptureSegment:
        capture_segment = CaptureSegment(
            sample_start=sample_start,
            frequency=center_frequency_Hz,
            datetime=start_time,
            duration=duration_ms,
            overload=overload,
            sigan_settings=sigan_settings,
        )
        sigan_cal = self.sensor.signal_analyzer.sigan_calibration_data
        sensor_cal = self.sensor.signal_analyzer.sensor_calibration_data
        # Rename compression point keys if they exist
        # then set calibration metadata if it exists
        if sensor_cal is not None:
            if "1db_compression_point" in sensor_cal:
                sensor_cal["compression_point"] = sensor_cal.pop(
                    "1db_compression_point"
                )
            capture_segment.sensor_calibration = ntia_sensor.Calibration(**sensor_cal)
        if sigan_cal is not None:
            if "1db_compression_point" in sigan_cal:
                sigan_cal["compression_point"] = sigan_cal.pop("1db_compression_point")
            capture_segment.sigan_calibration = ntia_sensor.Calibration(**sigan_cal)
        return capture_segment

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
    ) -> Union[ntia_sensor.SiganSettings, None]:
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
        self, num_samples: int, nskip: int = 0, cal_adjust: bool = True
    ) -> dict:
        logger.debug(
            f"Acquiring {num_samples} IQ samples, skipping the first {nskip} samples"
            + f" and {'' if cal_adjust else 'not '}applying gain adjustment based"
            + " on calibration data"
        )
        measurement_result = self.sensor.signal_analyzer.acquire_time_domain_samples(
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
