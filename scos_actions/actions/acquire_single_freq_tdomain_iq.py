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

from numpy import complex64

from scos_actions import utils
from scos_actions.actions.interfaces.measurement_action import MeasurementAction
from scos_actions.utils import get_parameter

logger = logging.getLogger(__name__)

# Define parameter keys
FREQUENCY = "frequency"
SAMPLE_RATE = "sample_rate"
DURATION_MS = "duration_ms"
NUM_SKIP = "nskip"
CLASSIFICATION = "classification"
CAL_ADJUST = "calibration_adjust"


class SingleFrequencyTimeDomainIqAcquisition(MeasurementAction):
    """Acquire IQ data at each of the requested frequencies.

    The action will set any matching attributes found in the
    signal analyzer object. The following parameters are
    required by the action:

        name: name of the action
        frequency: center frequency in Hz
        duration_ms: duration to acquire in ms

    For the parameters required by the signal analyzer, see the
    documentation from the Python package for the signal analyzer
    being used.

    :param parameters: The dictionary of parameters needed for
    the action and the signal analyzer.
    :param sigan: instance of SignalAnalyzerInterface.
    """

    def __init__(self, parameters: dict):
        super().__init__(parameters=parameters)
        # Pull parameters from action config
        self.nskip = get_parameter(NUM_SKIP, self.parameters)
        self.duration_ms = get_parameter(DURATION_MS, self.parameters)
        self.frequency_Hz = get_parameter(FREQUENCY, self.parameters)
        self.classification = get_parameter(CLASSIFICATION, self.parameters)
        self.cal_adjust = get_parameter(CAL_ADJUST, self.parameters)

    def execute(self, schedule_entry: dict, task_id: int) -> dict:
        # Use the sigan's actual reported instead of requested sample rate
        sample_rate = self.sensor.signal_analyzer.sample_rate
        num_samples = int(sample_rate * self.duration_ms * 1e-3)
        measurement_result = self.acquire_data(
            num_samples, self.nskip, self.cal_adjust, cal_params=self.parameters
        )
        end_time = utils.get_datetime_str_now()
        measurement_result.update(self.parameters)
        measurement_result["end_time"] = end_time
        measurement_result["task_id"] = task_id
        # measurement_result["calibration_datetime"] = (
        #     self.sensor.sensor_calibration_data["datetime"]
        # )
        measurement_result["classification"] = self.classification
        sigan_settings = self.get_sigan_settings(measurement_result)
        logger.debug(f"sigan settings:{sigan_settings}")
        measurement_result["capture_segment"] = self.create_capture_segment(
            sample_start=0,
            start_time=measurement_result["capture_time"],
            center_frequency_Hz=self.frequency_Hz,
            duration_ms=self.duration_ms,
            overload=measurement_result["overload"],
            sigan_settings=sigan_settings,
        )
        return measurement_result

    @property
    def description(self):
        """Parameterize and return the module-level docstring."""
        frequency_MHz = self.frequency_Hz / 1e6
        used_keys = [FREQUENCY, DURATION_MS, "name"]
        acq_plan = (
            f"The signal analyzer is tuned to {frequency_MHz:.2f} "
            + "MHz and the following parameters are set:\n"
        )
        for name, value in self.parameters.items():
            if name not in used_keys:
                acq_plan += f"{name} = {value}\n"
        acq_plan += f"\nThen, acquire samples for {self.duration_ms} ms\n."

        defs = {
            "name": self.name,
            "center_frequency": frequency_MHz,
            "acquisition_plan": acq_plan,
        }

        # __doc__ refers to the module docstring at the top of the file
        return __doc__.format(**defs)

    def transform_data(self, measurement_result):
        return measurement_result["data"].astype(complex64)

    def is_complex(self) -> bool:
        return True
