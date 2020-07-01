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

## Data Archive

The total size for all captures will be ${total_samples}\; \text{{samples}} \times 8\;
\text{{bytes per sample}} = {filesize_mb:.2f}\; \text{{MB}}$ plus metadata.

"""

import logging
from itertools import zip_longest

import numpy as np

from scos_actions import utils
from scos_actions.actions import sigmf_builder as scos_actions_sigmf
from scos_actions.actions.interfaces.action import Action
from scos_actions.actions.interfaces.signals import measurement_action_completed
from scos_actions.actions.sigmf_builder import Domain, MeasurementType, SigMFBuilder
from scos_actions.actions.acquire_single_freq_tdomain_iq import SingleFrequencyTimeDomainIqAcquisition

logger = logging.getLogger(__name__)


class SteppedFrequencyTimeDomainIqAcquisition(SingleFrequencyTimeDomainIqAcquisition):
    """Acquire IQ data at each of the requested frequecies.

    :param name: the name of the action
    :param fcs: an iterable of center frequencies in Hz
    :param gains: requested gain in dB, per center_frequency
    :param sample_rates: iterable of sample_rates in Hz, per center_frequency
    :param durations_ms: duration to acquire in ms, per center_frequency
    :param radio: instance of RadioInterface

    """

    def __init__(self, parameters, radio):
        super(SteppedFrequencyTimeDomainIqAcquisition, self).__init__(parameters, radio)
        self.sorted_measurement_parameters = []
        num_center_frequencies = len(self.parameters["fcs"])

        parameter_names =  {
            "fcs": "frequency",
            "gains": "gain",
            "attenuations": "attenuation",
            "sample_rates": "sample_rate",
            "durations_ms": "duration_ms",
        }

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
                if key in parameter_names: # convert plural name to singular name to map to radio interface property
                    sorted_params[parameter_names[key]] = parameters[key][i]
                else:
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
            self.set_base_sigmf_global(sigmf_builder, schedule_entry_json, sensor_definition, measurement_result, task_id, recording_id)
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
        acq_plan_template = "1. Tune to {fc_MHz:.2f} MHz, "
        acq_plan_template += "set gain to {gain} dB, "
        acq_plan_template += "and acquire at {sample_rate_Msps:.2f} Msps "
        acq_plan_template += "for {duration_ms} ms\n"

        total_samples = 0
        for measurement_params in self.sorted_measurement_parameters:
            acquisition_plan += acq_plan_template.format(
                **{
                    "fc_MHz": measurement_params["frequency"] / 1e6,
                    "gain": measurement_params["gain"],
                    "sample_rate_Msps": measurement_params["sample_rate"] / 1e6,
                    "duration_ms": measurement_params["duration_ms"],
                }
            )
            total_samples += int(
                measurement_params["duration_ms"] / 1000 * measurement_params["sample_rate"]
            )

        f_low = self.sorted_measurement_parameters[0]["frequency"]
        f_low_srate = self.sorted_measurement_parameters[0]["sample_rate"]
        f_low_edge = (f_low - f_low_srate / 2.0) / 1e6

        f_high = self.sorted_measurement_parameters[-1]["frequency"]
        f_high_srate = self.sorted_measurement_parameters[-1]["sample_rate"]
        f_high_edge = (f_high - f_high_srate / 2.0) / 1e6

        durations = [v["duration_ms"] for v in self.sorted_measurement_parameters]
        min_duration_ms = np.sum(durations)

        filesize_mb = total_samples * 8 / 1e6  # 8 bytes per complex64 sample

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
            "total_samples": total_samples,
            "filesize_mb": filesize_mb,
        }

        # __doc__ refers to the module docstring at the top of the file
        return __doc__.format(**defs)
