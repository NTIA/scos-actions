# What follows is a parameterizable description of the algorithm used by this
# action. The first line is the summary and should be written in plain text.
# Everything following that is the extended description, which can be written
# in Markdown and MathJax. Each name in curly brackets '{}' will be replaced
# with the value specified in the `description` method which can be found at
# the very bottom of this file. Since this parameterization step affects
# everything in curly brackets, math notation such as {m \over n} must be
# escaped to {{m \over n}}.
#
# To print out this docstring after parameterization, see
# scos-sensor/scripts/print_action_docstring.py. You can then paste that into the
# SCOS Markdown Editor (link below) to see the final rendering.
#
# Resources:
# - MathJax reference: https://math.meta.stackexchange.com/q/5020
# - Markdown reference: https://commonmark.org/help/
# - SCOS Markdown Editor: https://ntia.github.io/scos-md-editor/
#
r"""Capture time-domain IQ samples at {center_frequency:.2f} MHz.

# {name}

## Signal analyzer setup and sample acquisition

Each time this task runs, the following process is followed:
{acquisition_plan}

## Time-domain processing

If specified, a voltage scaling factor is applied to the complex time-domain
signals.
"""

import logging

import numpy as np

from scos_actions import utils
from scos_actions.actions.interfaces.action import Action
from scos_actions.actions.interfaces.signals import measurement_action_completed
from scos_actions.actions.sigmf_builder import Domain, MeasurementType, SigMFBuilder
from scos_actions.hardware import gps as mock_gps
from scos_actions.actions.metadata_decorators.time_domain_annotation_decorator import TimeDomainAnnotationDecorator
from scos_actions.actions.metadata_decorators.calibration_annotation_decorator import CalibrationAnnotationDecorator
from scos_actions.actions.metadata_decorators.measurement_global_deocorator import MeasurementDecorator
from scos_actions.actions.metadata_decorators.sensor_annotation_decorator import SensorAnnotationDecorator

logger = logging.getLogger(__name__)


class SingleFrequencyTimeDomainIqAcquisition(Action):
    """Acquire IQ data at each of the requested frequencies.

    :param parameters: The dictionary of parameters needed for the action and the signal analyzer.

    The action will set any matching attributes found in the signal analyzer object. The following
    parameters are required by the action:

        name: name of the action
        frequency: center frequency in Hz
        duration_ms: duration to acquire in ms

    or the parameters required by the signal analyzer, see the documentation from the Python
    package for the signal analyzer being used.

    :param sigan: instance of SignalAnalyzerInterface
    """

    def __init__(self, parameters, sigan, gps=mock_gps):
        super().__init__(parameters=parameters, sigan=sigan, gps=gps)
        self.is_complex = True

    def __call__(self, schedule_entry_json, task_id):
        """This is the entrypoint function called by the scheduler."""
        self.test_required_components()
        start_time = utils.get_datetime_str_now()
        measurement_result = self.acquire_data(self.parameters)
        measurement_result['start_time'] = start_time
        end_time = utils.get_datetime_str_now()
        measurement_result['end_time'] = end_time
        measurement_result['domain'] = Domain.TIME.value
        measurement_result['measurement_type'] = MeasurementType.SINGLE_FREQUENCY.value
        measurement_result['task_id'] = task_id
        measurement_result['frequency_low'] = self.parameter_map['frequency']
        measurement_result['frequency_high'] = self.parameter_map['frequency']
        measurement_result['calibration_datetime'] = self.sigan.sensor_calibration_data['calibration_datetime']
        measurement_result['name'] = self.parameter_map['name']
        self.add_metadata_decorators(measurement_result)
        self.create_metadata(schedule_entry_json, measurement_result)
        measurement_action_completed.send(
            sender=self.__class__,
            task_id=task_id,
            data=measurement_result["data"].astype(np.complex64),
            metadata=self.sigmf_builder.metadata,
        )

    def add_metadata_decorators(self, measurement_result):
        received_samples = len(measurement_result["data"].flatten())
        time_domain_decorator = TimeDomainAnnotationDecorator(self.sigmf_builder, 0, received_samples)
        self.decorators[type(time_domain_decorator).__name__] = time_domain_decorator
        calibration_annotation_decorator = CalibrationAnnotationDecorator(self.sigmf_builder, 0, received_samples)
        self.decorators[type(calibration_annotation_decorator).__name__] = calibration_annotation_decorator
        measurement_decorator = MeasurementDecorator(self.sigmf_builder)
        self.decorators[type(measurement_decorator).__name__] = measurement_decorator
        sensor_annotation = SensorAnnotationDecorator(self.sigmf_builder, 0, received_samples)
        self.decorators[type(sensor_annotation).__name__] = sensor_annotation

    def create_metadata(self, schedule_entry, measurement_result):
        self.sigmf_builder.set_base_sigmf_global(
            self.sigmf_builder,
            schedule_entry,
            self.sensor_definition,
            measurement_result, self.is_complex
        )
        self.sigmf_builder.add_sigmf_capture(self.sigmf_builder, measurement_result)
        self.sigmf_builder.add_base_sigmf_annotations(self.sigmf_builder, measurement_result)
        for decorator in self.decorators.values():
            decorator.decorate(self.sigan.sigan_calibration_data, self.sigan.sensor_calibration_data,
                               measurement_result)

    def test_required_components(self):
        """Fail acquisition if a required component is not available."""
        if not self.sigan.is_available:
            msg = "acquisition failed: signal analyzer required but not available"
            raise RuntimeError(msg)

    def acquire_data(self, measurement_params):
        self.configure(measurement_params)

        # Use the signal analyzer's actual reported sample rate instead of requested rate
        sample_rate = self.sigan.sample_rate

        num_samples = int(sample_rate * measurement_params["duration_ms"] * 1e-3)

        nskip = None
        if "nskip" in measurement_params:
            nskip = measurement_params["nskip"]
        logger.debug(
            f"acquiring {num_samples} samples and skipping the first {nskip if nskip else 0} samples"
        )
        measurement_result = self.sigan.acquire_time_domain_samples(
            num_samples, num_samples_skip=nskip
        )

        return measurement_result


    @property
    def description(self):
        """Parameterize and return the module-level docstring."""
        center_frequency = self.parameter_map["frequency"] / 1e6
        duration_ms = self.parameter_map["duration_ms"]
        used_keys = ["frequency", "duration_ms", "name"]
        acq_plan = f"The signal analyzer is tuned to {center_frequency:.2f} MHz and the following parameters are set:\n"
        for name, value in self.parameter_map.items():
            if name not in used_keys:
                acq_plan += f"{name} = {value}\n"
        acq_plan += f"\nThen, acquire samples for {duration_ms} ms\n."

        defs = {
            "name": self.name,
            "center_frequency": center_frequency,
            "acquisition_plan": acq_plan,
        }

        # __doc__ refers to the module docstring at the top of the file
        return __doc__.format(**defs)
