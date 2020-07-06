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
r"""Capture time-domain IQ samples at {center_frequency}.

# {name}

## Radio setup and sample acquisition

Each time this task runs, the following process is followed:

{acquisition_plan}

## Time-domain processing

If specified, a voltage scaling factor is applied to the complex time-domain
signals.

## Data Archive

Each capture will be ${total_samples}\; \text{{samples}} \times 8\;
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

logger = logging.getLogger(__name__)


class SingleFrequencyTimeDomainIqAcquisition(Action):
    """Acquire IQ data at each of the requested frequecies.

    :param name: the name of the action
    :param fcs: an iterable of center frequencies in Hz
    :param gains: requested gain in dB, per center_frequency
    :param sample_rates: iterable of sample_rates in Hz, per center_frequency
    :param durations_ms: duration to acquire in ms, per center_frequency
    :param radio: instance of RadioInterface

    """

    def __init__(self, parameters, radio):
        super(SingleFrequencyTimeDomainIqAcquisition, self).__init__()

        self.parameters = parameters
        self.radio = radio  # make instance variable to allow mocking

    @property
    def name(self):
        return self.parameters["name"]

    def __call__(self, schedule_entry_json, task_id, sensor_definition):
        """This is the entrypoint function called by the scheduler."""
        self.test_required_components()
        start_time = utils.get_datetime_str_now()
        measurement_result = self.acquire_data(self.parameters)
        end_time = utils.get_datetime_str_now()
        received_samples = len(measurement_result["data"])
        # sigmf_builder = self.build_sigmf_md(
        #     start_time,
        #     end_time,
        #     measurement_result,
        #     schedule_entry_json,
        #     sensor_definition,
        #     task_id)
        sigmf_builder = SigMFBuilder()
        self.set_base_sigmf_global(sigmf_builder, schedule_entry_json, sensor_definition, measurement_result, task_id)
        sigmf_builder.set_measurement(
            start_time,
            end_time,
            domain=Domain.TIME,
            measurement_type=MeasurementType.SINGLE_FREQUENCY,
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
            data=measurement_result["data"].astype(np.complex64),
            metadata=sigmf_builder.metadata,
        )

    def test_required_components(self):
        """Fail acquisition if a required component is not available."""
        if not self.radio.is_available:
            msg = "acquisition failed: signal analyzer required but not available"
            raise RuntimeError(msg)

    def acquire_data(self, measurement_params):
        self.configure_sdr(measurement_params)

        # Use the radio's actual reported sample rate instead of requested rate
        sample_rate = self.radio.sample_rate

        num_samples = int(sample_rate * measurement_params["duration_ms"] * 1e-3)

        nskip = None
        if "nskip" in self.parameters:
            nskip = self.parameters["nskip"]
        logger.debug(f"acquiring {num_samples} samples and skipping the first {nskip if nskip else 0} samples")
        measurement_result = self.radio.acquire_time_domain_samples(
            num_samples, num_samples_skip=nskip
        )

        return measurement_result

    def set_base_sigmf_global(self, sigmf_builder, schedule_entry_json, sensor_def, measurement_result, task_id, recording_id=None, is_complex=True):
        sample_rate = measurement_result["sample_rate"] 
        sigmf_builder.set_last_calibration_time(self.radio.last_calibration_time)
        sigmf_builder.set_action(
            self.parameters["name"], self.description, self.description.splitlines()[0]
        )
        sigmf_builder.set_coordinate_system()
        sigmf_builder.set_data_type(is_complex=is_complex)
        sigmf_builder.set_sample_rate(sample_rate)
        sigmf_builder.set_schedule(schedule_entry_json)
        sigmf_builder.set_sensor(sensor_def)
        sigmf_builder.set_task(task_id)
        if recording_id:
            sigmf_builder.set_recording(recording_id)

    def add_sigmf_capture(self, sigmf_builder, measurement_result):
        sigmf_builder.set_capture(measurement_result["frequency"], measurement_result["capture_time"])


    def add_base_sigmf_annotations(
        self,
        sigmf_builder,
        measurement_result,
        ):
        received_samples = len(measurement_result["data"].flatten())

        sigmf_builder.add_annotation(
            start_index=0, length=received_samples, annotation_md=measurement_result["calibration_annotation"],
        )

        sigmf_builder.add_sensor_annotation(
            start_index=0,
            length=received_samples,
            overload=measurement_result["overload"],
            gain=measurement_result["gain"],
        )

    def configure_sdr(self, measurement_params):
        for key, value in measurement_params.items():
            if hasattr(self.radio, key):
                setattr(self.radio, key, value)

    @property
    def description(self):
        """Parameterize and return the module-level docstring."""

        acquisition_plan = ""
        acq_plan_template = "Tune to {fc_MHz:.2f} MHz, "
        acq_plan_template += "set gain to {gain} dB, "
        acq_plan_template += "and acquire at {sample_rate_Msps:.2f} Msps "
        acq_plan_template += "for {duration_ms} ms\n"

        total_samples = 0
        acquisition_plan += acq_plan_template.format(
            **{
                "fc_MHz": self.parameters["frequency"] / 1e6,
                "gain": self.parameters["gain"],
                "sample_rate_Msps": self.parameters["sample_rate"] / 1e6,
                "duration_ms": self.parameters["duration_ms"],
            }
        )
        total_samples += int(
            self.parameters["duration_ms"] / 1000 * self.parameters["sample_rate"]
        )

        filesize_mb = total_samples * 8 / 1e6  # 8 bytes per complex64 sample

        defs = {
            "name": self.name,
            "center_frequency": "{:.2f} MHz".format(self.parameters["frequency"] / 1e6),
            "acquisition_plan": acquisition_plan,
            "total_samples": total_samples,
            "filesize_mb": filesize_mb,
        }

        # __doc__ refers to the module docstring at the top of the file
        return __doc__.format(**defs)
