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
# REPO_ROOT/scripts/print_action_docstring.py. You can then paste that into the
# SCOS Markdown Editor (link below) to see the final rendering.
#
# Resources:
# - MathJax reference: https://math.meta.stackexchange.com/q/5020
# - Markdown reference: https://commonmark.org/help/
# - SCOS Markdown Editor: https://ntia.github.io/scos-md-editor/
#
r"""Capture time-domain IQ samples at the following {num_center_frequencies} frequencies: {center_frequencies}.

# {name}

## Radio setup and sample acquisition

Each time this task runs, the following process is followed:

{acquisition_plan}

This will take a minimum of {min_duration_ms:.2f} ms, not including radio
tuning, dropping samples after retunes, and data storage.

## Time-domain processing

If specified, a voltage scaling factor is applied to the complex time-domain
signals.
"""

import logging
from itertools import zip_longest

import numpy as np

from scos_actions import utils
from scos_actions.actions import sigmf_builder as scos_actions_sigmf
from scos_actions.actions.acquire_single_freq_tdomain_iq import (
    SingleFrequencyTimeDomainIqAcquisition,
)
from scos_actions.actions.interfaces.action import Action
from scos_actions.actions.interfaces.signals import measurement_action_completed
from scos_actions.actions.sigmf_builder import Domain, MeasurementType, SigMFBuilder

logger = logging.getLogger(__name__)


class SteppedFrequencyTimeDomainIqAcquisition(SingleFrequencyTimeDomainIqAcquisition):
    """Acquire IQ data at each of the requested frequecies.

    :param parameters: The dictionary of parameters needed for the action and the radio.

    The action will set any matching attributes found in the radio object. The following
    parameters are required by the action:

        name: name of the action
        fcs: an iterable of center frequencies in Hz
        durations_ms: duration to acquire in ms, per center_frequency

    For the parameters required by the radio, see the documentation for the radio being used.

    :param radio: instance of RadioInterface
    """

    def __init__(self, parameters, radio):
        super(SteppedFrequencyTimeDomainIqAcquisition, self).__init__(parameters, radio)
        self.sorted_measurement_parameters = []
        num_center_frequencies = len(self.parameters["frequency"])

        # convert dictionary of lists from yaml file to list of dictionaries
        longest_length = 0
        for key, value in parameters.items():
            if key == "name":
                continue
            if len(value) > longest_length:
                longest_length = len(value)
        for i in range(longest_length):
            sorted_params = {}
            for key in parameters.keys():
                if key == "name":
                    continue
                sorted_params[key] = parameters[key][i]
            self.sorted_measurement_parameters.append(sorted_params)
        self.sorted_measurement_parameters.sort(key=lambda params: params["frequency"])

        self.radio = radio  # make instance variable to allow mocking
        self.num_center_frequencies = num_center_frequencies

    def __call__(self, schedule_entry_json, task_id, sensor_definition):
        """This is the entrypoint function called by the scheduler."""
        self.test_required_components()

        for recording_id, measurement_params in enumerate(
            self.sorted_measurement_parameters, start=1
        ):
            start_time = utils.get_datetime_str_now()
            measurement_result = super().acquire_data(measurement_params)
            end_time = utils.get_datetime_str_now()
            received_samples = len(measurement_result["data"])
            sigmf_builder = SigMFBuilder()
            self.set_base_sigmf_global(
                sigmf_builder,
                schedule_entry_json,
                sensor_definition,
                measurement_result,
                task_id,
                recording_id,
            )
            sigmf_builder.set_measurement(
                start_time,
                end_time,
                domain=Domain.TIME,
                measurement_type=MeasurementType.SURVEY,
                frequency=measurement_result["frequency"],
            )
            self.add_sigmf_capture(sigmf_builder, measurement_result)
            self.add_base_sigmf_annotations(sigmf_builder, measurement_result)
            sigmf_builder.add_time_domain_detection(
                start_index=0,
                num_samples=received_samples,
                detector="sample_iq",
                units="volts",
                reference="preselector input",
            )
            measurement_action_completed.send(
                sender=self.__class__,
                task_id=task_id,
                data=measurement_result["data"],
                metadata=sigmf_builder.metadata,
            )

    @property
    def description(self):
        """Parameterize and return the module-level docstring."""

        acquisition_plan = ""
        used_keys = ["frequency", "duration_ms", "name"]
        acq_plan_template = "The radio is tuned to {center_frequency:.2f} MHz and the following parameters are set:\n"
        acq_plan_template += "{parameters}"
        acq_plan_template += "Then, acquire samples for {duration_ms} ms\n."

        for measurement_params in self.sorted_measurement_parameters:
            parameters = ""
            for name, value in measurement_params.items():
                if name not in used_keys:
                    parameters += f"{name} = {value}\n"
            acquisition_plan += acq_plan_template.format(
                **{
                    "center_frequency": measurement_params["frequency"] / 1e6,
                    "parameters": parameters,
                    "duration_ms": measurement_params["duration_ms"],
                }
            )

        durations = [v["duration_ms"] for v in self.sorted_measurement_parameters]
        min_duration_ms = np.sum(durations)

        defs = {
            "name": self.name,
            "num_center_frequencies": self.num_center_frequencies,
            "center_frequencies": ", ".join(
                [
                    "{:.2f} MHz".format(param["frequency"] / 1e6)
                    for param in self.sorted_measurement_parameters
                ]
            ),
            "acquisition_plan": acquisition_plan,
            "min_duration_ms": min_duration_ms,
        }

        # __doc__ refers to the module docstring at the top of the file
        return __doc__.format(**defs)
