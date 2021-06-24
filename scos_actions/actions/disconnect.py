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
from scos_actions.actions.measurement_params import MeasurementParams

logger = logging.getLogger(__name__)


class Disconnect(Action):
    """Acquire IQ data at each of the requested frequecies.

    :param name: the name of the action
    :param fcs: an iterable of center frequencies in Hz
    :param gains: requested gain in dB, per center_frequency
    :param sample_rates: iterable of sample_rates in Hz, per center_frequency
    :param durations_ms: duration to acquire in ms, per center_frequency
    :param radio: instance of RadioInterface

    """

    def __init__(self, name, fcs, gains, sample_rates, durations_ms, subdev, radio):
        super(Disconnect, self).__init__()

        num_center_frequencies = len(fcs)

        parameter_names = ("center_frequency", "gain", "sample_rate", "duration_ms", "subdev")
        measurement_params_list = []

        # Sort combined parameter list by frequency
        def sortFrequency(zipped_params):
            return zipped_params[0]

        sorted_params = list(zip_longest(fcs, gains, sample_rates, durations_ms, subdev))
        sorted_params.sort(key=sortFrequency)

        for params in sorted_params:
            if None in params:
                param_name = parameter_names[params.index(None)]
                err = "Wrong number of {}s, expected {}"
                raise TypeError(err.format(param_name, num_center_frequencies))

            measurement_params_list.append(
                MeasurementParams(**dict(zip(parameter_names, params)))
            )

        self.name = name
        self.radio = radio  # make instance variable to allow mocking
        self.num_center_frequencies = num_center_frequencies
        self.measurement_params_list = measurement_params_list

    def __call__(self, schedule_entry_json, task_id, sensor_definition):
        """This is the entrypoint function called by the scheduler."""
        self.test_required_components()

        for recording_id, measurement_params in enumerate(
            self.measurement_params_list, start=1
        ):
            start_time = utils.get_datetime_str_now()
            data = self.acquire_data(measurement_params)
            end_time = utils.get_datetime_str_now()

            sigmf_builder = self.build_sigmf_md(
                measurement_params,
                start_time,
                end_time,
                self.radio.capture_time,
                schedule_entry_json,
                sensor_definition,
                task_id,
                recording_id,
            )
            measurement_action_completed.send(
                sender=self.__class__,
                task_id=task_id,
                data=data,
                metadata=sigmf_builder.metadata,
            )

    @property
    def is_multirecording(self):
        return len(self.measurement_params_list) > 1

    def test_required_components(self):
        """Fail acquisition if a required component is not available."""
        if not self.radio.is_available:
            msg = "acquisition failed: SDR required but not available"
            raise RuntimeError(msg)

    def acquire_data(self, measurement_params):
        self.configure_sdr(measurement_params)

        # Use the radio's actual reported sample rate instead of requested rate
        sample_rate = self.radio.sample_rate

        # Acquire data and build per-capture metadata
        data = np.array([], dtype=np.complex64)

        num_samples = measurement_params.get_num_samples()

        subdev = measurement_params.subdev
        #if measurement_params.subdev == None:
        #    subdev == "A:A"

        # Drop ~10 ms of samples
        # nskip = int(0.01 * sample_rate)
        # acq = self.radio.acquire_time_domain_samples(
        #     num_samples, num_samples_skip=nskip, subdev=subdev
        # ).astype(np.complex64)

        self.radio.disconnect()

        data = np.zeros(10)

        return data

    def build_sigmf_md(
        self,
        measurement_params,
        start_time,
        end_time,
        capture_time,
        schedule_entry_json,
        sensor,
        task_id,
        recording_id,
    ):
        frequency = self.radio.frequency
        sample_rate = self.radio.sample_rate
        sigmf_builder = scos_actions_sigmf.SigMFBuilder()

        sigmf_builder.set_last_calibration_time(self.radio.last_calibration_time)

        sigmf_builder.set_action(
            self.name, self.description, self.description.splitlines()[0]
        )
        sigmf_builder.set_capture(frequency, capture_time)
        sigmf_builder.set_coordinate_system()
        sigmf_builder.set_data_type(is_complex=True)
        if self.is_multirecording:
            measurement_type = scos_actions_sigmf.MeasurementType.SURVEY
        else:
            measurement_type = scos_actions_sigmf.MeasurementType.SINGLE_FREQUENCY
        sigmf_builder.set_measurement(
            start_time,
            end_time,
            domain=scos_actions_sigmf.Domain.TIME,
            measurement_type=measurement_type,
            frequency=frequency,
        )

        sigmf_builder.set_sample_rate(sample_rate)
        sigmf_builder.set_schedule(schedule_entry_json)
        sigmf_builder.set_sensor(sensor)
        sigmf_builder.set_task(task_id)
        if self.is_multirecording:
            sigmf_builder.set_recording(recording_id)

        num_samples = measurement_params.get_num_samples()

        sigmf_builder.add_time_domain_detection(
            start_index=0,
            num_samples=num_samples,
            detector="sample_iq",
            units="volts",
            reference="preselector input",
        )

        calibration_annotation_md = self.radio.create_calibration_annotation()
        sigmf_builder.add_annotation(
            start_index=0, length=num_samples, annotation_md=calibration_annotation_md,
        )

        overload = self.radio.overload

        sigmf_builder.add_sensor_annotation(
            start_index=0,
            length=num_samples,
            overload=overload,
            gain=measurement_params.gain,
        )
        return sigmf_builder

    def configure_sdr(self, measurement_params):
        self.radio.sample_rate = measurement_params.sample_rate
        self.radio.frequency = measurement_params.center_frequency
        self.radio.gain = measurement_params.gain
        self.radio.configure(self.name)

    @property
    def description(self):
        """Parameterize and return the module-level docstring."""

        acquisition_plan = ""
        acq_plan_template = "1. Tune to {fc_MHz:.2f} MHz, "
        acq_plan_template += "set gain to {gain} dB, "
        acq_plan_template += "and acquire at {sample_rate_Msps:.2f} Msps "
        acq_plan_template += "for {duration_ms} ms\n"

        total_samples = 0
        for measurement_params in self.measurement_params_list:
            acq_plan_template.format(
                **{
                    "fc_MHz": measurement_params.center_frequency / 1e6,
                    "gain": measurement_params.gain,
                    "sample_rate_Msps": measurement_params.sample_rate / 1e6,
                    "duration_ms": measurement_params.duration_ms,
                }
            )
            total_samples += int(
                measurement_params.duration_ms / 1e6 * measurement_params.sample_rate
            )

        f_low = self.measurement_params_list[0].center_frequency
        f_low_srate = self.measurement_params_list[0].sample_rate
        f_low_edge = (f_low - f_low_srate / 2.0) / 1e6

        f_high = self.measurement_params_list[-1].center_frequency
        f_high_srate = self.measurement_params_list[-1].sample_rate
        f_high_edge = (f_high - f_high_srate / 2.0) / 1e6

        durations = [v.duration_ms for v in self.measurement_params_list]
        min_duration_ms = np.sum(durations)

        filesize_mb = total_samples * 8 / 1e6  # 8 bytes per complex64 sample

        defs = {
            "name": self.name,
            "num_center_frequencies": self.num_center_frequencies,
            "center_frequencies": ", ".join(
                [
                    "{:.2f} MHz".format(param.center_frequency / 1e6)
                    for param in self.measurement_params_list
                ]
            ),
            "acquisition_plan": acquisition_plan,
            "min_duration_ms": min_duration_ms,
            "total_samples": total_samples,
            "filesize_mb": filesize_mb,
        }

        # __doc__ refers to the module docstring at the top of the file
        return __doc__.format(**defs)
