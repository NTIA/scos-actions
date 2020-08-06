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
r"""Record iq for {duration_ms}ms at time gps-sync-time at {center_frequency:.2f}MHz.

# {name}

## Radio setup and sample acquisition

This action first tunes the radio to {center_frequency:.2f} MHz and requests a sample
rate of {sample_rate:.2f} Msps and {gain} dB of gain.

"""

import logging

from scos_actions import utils
from scos_actions.actions.interfaces.action import Action
from scos_actions.actions.interfaces.signals import measurement_action_completed
from scos_actions.actions.measurement_params import MeasurementParams
from scos_actions.actions import sigmf_builder as scos_actions_sigmf
import numpy as np


logger = logging.getLogger(__name__)


class TransmitPN(Action):
    """Transmit PN sequence with BPSK modulation, defined by Hz, dB, samp_rate, and length of time. GPS clock used for start time.

    :param name: the name of the action
    :param frequency: center frequency in Hz
    :param gain: requested gain in dB
    :param sample_rate: requested sample_rate in Hz
    :param duration_ms: duration of the recording in ms

    """

    ## seed, sampspersymbol, spacing, duration
    ##  n, arg_start_time_goal, retries=5
    def __init__(self, name, frequency, gain, sample_rate, duration_ms, seed, sampspersymbol, spacing, radio):
        super(TransmitPN, self).__init__()

        self.name = name
        self.measurement_params = MeasurementParams(
            center_frequency=frequency,
            gain=gain,
            sample_rate=sample_rate,
            duration_ms=duration_ms,
            seed=seed,
            sampspersymbol=sampspersymbol,
            spacing=spacing
        )
        #self.sdr = sdr  # make instance variable to allow mocking
        self.radio = radio
        logger.debug("radio struct: ".format(radio))
        self.enbw = None

    def __call__(self, schedule_entry_json, task_id, sensor_definition):
        """This is the entrypoint function called by the scheduler."""
        # new_epoch_time = schedule_entry_json['GPS_sync_start']

        self.test_required_components()
        self.configure_sdr()
        start_time = utils.get_datetime_str_now()
        data = self.call_transmit_pn()
        end_time = utils.get_datetime_str_now()
        #sigmf_md = self.build_sigmf_md(
        #    task_id, data, task_result.schedule_entry, start_time, end_time
        #)
        #self.archive(task_result, data, sigmf_md)
        sigmf_builder = self.build_sigmf_md(start_time, end_time, self.radio.capture_time, schedule_entry_json,
                                            sensor_definition, task_id, data)
        measurement_action_completed.send(sender=self.__class__, task_id=task_id, data=data, metadata=sigmf_builder.metadata)

    def test_required_components(self):
        """Fail acquisition if a required component is not available."""
        #self.sdr.connect()
        if not self.radio.is_available:
            msg = "acquisition failed: SDR required but not available"
            raise RuntimeError(msg)

    def configure_sdr(self):
        self.radio.sample_rate = self.measurement_params.sample_rate
        self.radio.frequency = self.measurement_params.center_frequency
        self.radio.gain = self.measurement_params.gain
        self.radio.configure(self.name)

    def call_transmit_pn(self):
        msg = "Transmitting... samp_rate {} at center freq. {} MHz for {} ms"
        frequency = self.measurement_params.center_frequency
        sample_rate = self.measurement_params.sample_rate
        duration_ms = self.measurement_params.duration_ms
        #nsamps = sample_rate * duration_ms * 1e-3
        logger.debug(msg.format(sample_rate, frequency / 1e6, duration_ms))

        # Drop ~10 ms of samples
        #nskip = int(0.01 * sample_rate)
        logger.debug("in acquire_data; radio struct: ".format(self.radio))
        data = self.radio.transmit_pn()

        return data

    def build_sigmf_md(self, start_time, end_time, capture_time, schedule_entry_json, sensor, task_id, data):
        #self, measurement_params, start_time, end_time, capture_time, schedule_entry_json, sensor, task_id, data, recording_id
                
        logger.debug("Building SigMF metadata file")

        # Use the radio's actual reported sample rate instead of requested rate
        sample_rate = self.radio.sample_rate
        frequency = self.radio.frequency

        sigmf_builder = scos_actions_sigmf.SigMFBuilder()
        sigmf_builder.set_action(self.name, self.description, self.description.splitlines()[0])
        sigmf_builder.set_capture(frequency, capture_time)
        sigmf_builder.set_coordinate_system()
        sigmf_builder.set_data_type(is_complex=True)

        sigmf_builder.set_measurement(
            start_time, end_time,
            domain=scos_actions_sigmf.Domain.TIME,
            measurement_type=scos_actions_sigmf.MeasurementType.SINGLE_FREQUENCY,
            frequency=frequency
        )
        sigmf_builder.set_sample_rate(sample_rate)
        sigmf_builder.set_schedule(schedule_entry_json)
        sigmf_builder.set_sensor(sensor)
        sigmf_builder.set_task(task_id)

        num_samples = self.measurement_params.get_num_samples()

        sigmf_builder.add_time_domain_detection(start_index=0, num_samples=num_samples,
                                                detector="sample_iq", units="volts",
                                                reference="preselector input")

        calibration_annotation_md = self.radio.create_calibration_annotation()
        sigmf_builder.add_annotation(
            start_index=0,
            length=num_samples,
            annotation_md=calibration_annotation_md,
        )

        # Recover the sigan overload flag
        overload = self.radio.overload

        # Check time domain average power versus calibrated compression
        time_domain_avg_power = 10 * np.log10(np.mean(np.abs(data) ** 2))
        time_domain_avg_power += (
                10 * np.log10(1 / (2 * 50)) + 30
        )

        sigmf_builder.add_sensor_annotation(start_index=0,
                                            length=num_samples,
                                            overload=overload,
                                            gain=self.measurement_params.gain)
        return sigmf_builder

    @property
    def description(self):
        defs = {
            "name": self.name,
            "center_frequency": self.measurement_params.center_frequency / 1e6,
            "sample_rate": self.measurement_params.sample_rate / 1e6,
            "gain": self.measurement_params.gain,
            "duration_ms": self.measurement_params.duration_ms,
        }

        # __doc__ refers to the module docstring at the top of the file
        return __doc__.format(**defs)
