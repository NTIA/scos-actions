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

from scos_actions import utils
from scos_actions.actions.action_utils import get_num_skip
from scos_actions.actions.interfaces.measurement_action import MeasurementAction
from scos_actions.actions.sigmf_builder import Domain, MeasurementType
from scos_actions.hardware import gps as mock_gps
from scos_actions.actions.metadata.annotations.time_domain_annotation import TimeDomainAnnotation
from numpy import complex64


logger = logging.getLogger(__name__)


class SingleFrequencyTimeDomainIqAcquisition(MeasurementAction):
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

    def execute(self, schedule_entry, task_id):
        start_time = utils.get_datetime_str_now()
        nskip = get_num_skip(self.parameter_map)
        # Use the signal analyzer's actual reported sample rate instead of requested rate
        sample_rate = self.sigan.sample_rate
        num_samples = int(sample_rate * self.parameter_map["duration_ms"] * 1e-3)
        measurement_result = self.acquire_data(num_samples, nskip)
        measurement_result['start_time'] = start_time
        end_time = utils.get_datetime_str_now()
        measurement_result.update(self.parameter_map)
        measurement_result['end_time'] = end_time
        measurement_result['domain'] = Domain.TIME.value
        measurement_result['measurement_type'] = MeasurementType.SINGLE_FREQUENCY.value
        measurement_result['task_id'] = task_id
        measurement_result['calibration_datetime'] = self.sigan.sensor_calibration_data['calibration_datetime']
        measurement_result['description'] = self.description
        return measurement_result

    def add_metadata_generators(self, measurement_result):
        super().add_metadata_generators(measurement_result)
        time_domain_annotation = TimeDomainAnnotation(self.sigmf_builder, 0, self.received_samples)
        self.metadata_generators[type(time_domain_annotation).__name__] = time_domain_annotation


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

    def transform_data(self, measurement_result):
        return measurement_result['data'].astype(complex64)
