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
r"""Apply m4s detector over {nffts} {fft_size}-pt FFTs at {center_frequency:.2f} MHz.

# {name}

## Radio setup and sample acquisition

This action first tunes the radio to {center_frequency:.2f} MHz and requests a sample
rate of {sample_rate:.2f} Msps and {gain} dB of gain.

It then begins acquiring, and discards an appropriate number of samples while
the radio's IQ balance algorithm runs. Then, ${nffts} \times {fft_size}$
samples are acquired gap-free.

## Time-domain processing

First, the ${nffts} \times {fft_size}$ continuous samples are acquired from
the radio. If specified, a voltage scaling factor is applied to the complex
time-domain signals. Then, the data is reshaped into a ${nffts} \times
{fft_size}$ matrix:

$$
\begin{{pmatrix}}
a_{{1,1}}      & a_{{1,2}}     & \cdots  & a_{{1,fft\_size}}     \\\\
a_{{2,1}}      & a_{{2,2}}     & \cdots  & a_{{2,fft\_size}}     \\\\
\vdots         & \vdots        & \ddots  & \vdots                \\\\
a_{{nffts,1}}  & a_{{nfts,2}}  & \cdots  & a_{{nfts,fft\_size}}  \\\\
\end{{pmatrix}}
$$

where $a_{{i,j}}$ is a complex time-domain sample.

At that point, a Flat Top window, defined as

$$w(n) = &0.2156 - 0.4160 \cos{{(2 \pi n / M)}} + 0.2781 \cos{{(4 \pi n / M)}} -
         &0.0836 \cos{{(6 \pi n / M)}} + 0.0069 \cos{{(8 \pi n / M)}}$$

where $M = {fft_size}$ is the number of points in the window, is applied to
each row of the matrix.

## Frequency-domain processing

After windowing, the data matrix is converted into the frequency domain using
an FFT, doing the equivalent of the DFT defined as

$$A_k = \sum_{{m=0}}^{{n-1}}
a_m \exp\left\\{{-2\pi i{{mk \over n}}\right\\}} \qquad k = 0,\ldots,n-1$$

The data matrix is then converted to pseudo-power by taking the square of the
magnitude of each complex sample individually, allowing power statistics to be
taken.

## Applying detector

Next, the M4S (min, max, mean, median, and sample) detector is applied to the
data matrix. The input to the detector is a matrix of size ${nffts} \times
{fft_size}$, and the output matrix is size $5 \times {fft_size}$, with the
first row representing the min of each _column_, the second row representing
the _max_ of each column, and so "sample" detector simple chooses one of the
{nffts} FFTs at random.

## Power conversion

To finish the power conversion, the samples are divided by the characteristic
impedance (50 ohms). The power is then referenced back to the RF power by
dividing further by 2. The powers are normalized to the FFT bin width by
dividing by the length of the FFT and converted to dBm. Finally, an FFT window
correction factor is added to the powers given by

$$ C_{{win}} = 20log \left( \frac{{1}}{{ mean \left( w(n) \right) }} \right)

The resulting matrix is real-valued, 32-bit floats representing dBm.

"""

import logging

from scos_actions import utils
from scos_actions.actions.fft import (
    get_fft_frequencies,
    M4sDetector,
    get_frequency_domain_data,
    convert_volts_to_watts,
    apply_detector,
    convert_watts_to_dbm,
)
from scos_actions.actions.interfaces.action import Action
from scos_actions.actions.interfaces.signals import measurement_action_completed
from scos_actions.actions.sigmf_builder import SigMFBuilder, Domain, MeasurementType
from scos_actions.actions.acquire_single_freq_tdomain_iq import (
    SingleFrequencyTimeDomainIqAcquisition,
)


logger = logging.getLogger(__name__)


class SingleFrequencyFftAcquisition(SingleFrequencyTimeDomainIqAcquisition):
    """Perform m4s detection over requested number of single-frequency FFTs.

    :param name: the name of the action
    :param frequency: center frequency in Hz
    :param gain: requested gain in dB
    :param sample_rate: requested sample_rate in Hz
    :param fft_size: number of points in FFT (some 2^n)
    :param nffts: number of consecutive FFTs to pass to detector
    :param radio: instance of RadioInterface

    """

    def __init__(self, parameters, radio):
        super(SingleFrequencyFftAcquisition, self).__init__(parameters, radio)

        self.enbw = None

    def __call__(self, schedule_entry_json, task_id, sensor_definition):
        """This is the entrypoint function called by the scheduler."""

        self.test_required_components()
        start_time = utils.get_datetime_str_now()
        measurement_result = self.acquire_data()
        end_time = utils.get_datetime_str_now()
        self.apply_m4s(measurement_result)

        sigmf_builder = SigMFBuilder()
        self.set_base_sigmf_global(
            sigmf_builder,
            schedule_entry_json,
            sensor_definition,
            measurement_result,
            task_id,
            is_complex=False,
        )
        sigmf_builder.set_measurement(
            start_time,
            end_time,
            domain=Domain.FREQUENCY,
            measurement_type=MeasurementType.SINGLE_FREQUENCY,
            frequency=measurement_result["frequency"],
        )
        self.add_sigmf_capture(sigmf_builder, measurement_result)
        self.add_base_sigmf_annotations(sigmf_builder, measurement_result)
        self.add_fft_annotations(sigmf_builder, measurement_result)
        measurement_action_completed.send(
            sender=self.__class__,
            task_id=task_id,
            data=measurement_result["data"],
            metadata=sigmf_builder.metadata,
        )

    def add_fft_annotations(self, sigmf_builder, measurement_result):
        frequencies = get_fft_frequencies(
            self.parameters["fft_size"],
            measurement_result["sample_rate"],
            measurement_result["frequency"],
        ).tolist()

        for i, detector in enumerate(M4sDetector):
            sigmf_builder.add_frequency_domain_detection(
                start_index=(i * self.parameters["fft_size"]),
                fft_size=self.parameters["fft_size"],
                enbw=self.enbw,
                detector="fft_" + detector.name + "_power",
                num_ffts=self.parameters["nffts"],
                window="flattop",
                units="dBm",
                reference="preselector input",
                frequency_start=frequencies[0],
                frequency_stop=frequencies[-1],
                frequency_step=frequencies[1] - frequencies[0],
            )

    def acquire_data(self):
        super().configure_sdr(self.parameters)
        msg = "Acquiring {} FFTs at {} MHz"
        if not "nffts" in self.parameters:
            raise Exception("nffts missing from measurement parameters")
        num_ffts = self.parameters["nffts"]
        if not "fft_size" in self.parameters:
            raise Exception("fft_size missing from measurement parameters")
        fft_size = self.parameters["fft_size"]

        nskip = None
        if "nskip" in self.parameters:
            nskip = self.parameters["nskip"]
        logger.debug(
            f"acquiring {num_ffts * fft_size} samples and skipping the first {nskip if nskip else 0} samples"
        )
        measurement_results = self.radio.acquire_time_domain_samples(
            num_ffts * fft_size, num_samples_skip=nskip
        )
        return measurement_results

    def apply_m4s(self, measurement_result):
        fft_size = self.parameters["fft_size"]
        complex_fft, self.enbw = get_frequency_domain_data(
            measurement_result["data"], measurement_result["sample_rate"], fft_size
        )
        power_fft = convert_volts_to_watts(complex_fft)
        power_fft_m4s = apply_detector(power_fft)
        power_fft_dbm = convert_watts_to_dbm(power_fft_m4s)
        measurement_result["data"] = power_fft_dbm

    @property
    def description(self):
        definitions = {
            "name": self.name,
            "center_frequency": self.parameters["frequency"] / 1e6,
            "sample_rate": self.parameters["sample_rate"] / 1e6,
            "fft_size": self.parameters["fft_size"],
            "nffts": self.parameters["nffts"],
            "gain": self.parameters["gain"],
        }

        # __doc__ refers to the module docstring at the top of the file
        return __doc__.format(**definitions)
