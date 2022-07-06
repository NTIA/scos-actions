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
r"""Apply m4s detector over {nffts} {fft_size}-pt FFTs at {center_frequency:.2f} MHz.

# {name}

## Signal Analyzer setup and sample acquisition

Each time this task runs, the following process is followed:
{acquisition_plan}

## Time-domain processing

First, the ${nffts} \times {fft_size}$ continuous samples are acquired from
the signal analyzer. If specified, a voltage scaling factor is applied to the complex
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
from scos_actions.actions.interfaces.measurement_action import MeasurementAction
from scos_actions.actions.fft import (
    M4sDetector,
    get_fft_frequencies,
    get_m4s_dbm,
    get_fft_window,
    get_fft_window_correction_factors,
    get_enbw
)
from scos_actions.actions.sigmf_builder import Domain, MeasurementType, SigMFBuilder
from scos_actions.actions.metadata.annotations.fft_annotation import FrequencyDomainDetectionAnnotation
from scos_actions.hardware import gps as mock_gps
from scos_actions.actions.action_utils import get_num_samples_and_fft_size
from scos_actions.actions.action_utils import get_num_skip

logger = logging.getLogger(__name__)


class SingleFrequencyFftAcquisition(MeasurementAction):
    """Perform m4s detection over requested number of single-frequency FFTs.

    :param parameters: The dictionary of parameters needed for the action and the signal analyzer.

    The action will set any matching attributes found in the signal analyzer object. The following
    parameters are required by the action:

        name: name of the action
        frequency: center frequency in Hz
        fft_size: number of points in FFT (some 2^n)
        nffts: number of consecutive FFTs to pass to detector

    or the parameters required by the signal analyzer, see the documentation from the Python
    package for the signal analyzer being used.

    :param sigan: instance of SignalAnalyzerInterface
    """

    def __init__(self, parameters, sigan, gps=mock_gps):
        super().__init__(parameters, sigan, gps)
        self.is_complex = False

    def execute(self, schedule_entry, task_id):
        start_time = utils.get_datetime_str_now()
        nskip = get_num_skip(self.parameter_map)
        num_samples, fft_size = get_num_samples_and_fft_size(self.parameter_map)
        measurement_result = self.acquire_data(num_samples, nskip)
        fft_window = get_fft_window("Flat Top", fft_size)
        fft_window_acf, fft_window_ecf, fft_window_enbw = get_fft_window_correction_factors(fft_window)
        enbw = get_enbw(measurement_result["sample_rate"], fft_size,fft_window_enbw)
        measurement_result['data'] = get_m4s_dbm(fft_size, measurement_result,fft_window, fft_window_acf)
        measurement_result['start_time'] = start_time
        measurement_result['end_time'] = utils.get_datetime_str_now()
        measurement_result['enbw'] = enbw
        frequencies = get_fft_frequencies(
            self.parameter_map["fft_size"],
            measurement_result["sample_rate"],
            self.parameter_map["frequency"],
        ).tolist()
        measurement_result.update(self.parameter_map)
        measurement_result['description'] = self.description
        measurement_result['domain'] = Domain.FREQUENCY.value
        measurement_result['frequency_start'] = frequencies[0]
        measurement_result['frequency_stop'] = frequencies[-1]
        measurement_result['frequency_step'] = frequencies[1] - frequencies[0]
        measurement_result['window'] = 'flattop'
        measurement_result['calibration_datetime'] = self.sigan.sensor_calibration_data['calibration_datetime']
        measurement_result['task_id'] = task_id
        measurement_result['measurement_type'] = MeasurementType.SINGLE_FREQUENCY.value
        measurement_result['sigan_cal'] = self.sigan.sigan_calibration_data
        measurement_result['sensor_cal'] = self.sigan.sensor_calibration_data
        return measurement_result



    @property
    def description(self):
        center_frequency = self.parameters["frequency"] / 1e6
        nffts = self.parameters["nffts"]
        fft_size = self.parameters["fft_size"]
        used_keys = ["frequency", "nffts", "fft_size", "name"]
        acq_plan = f"The signal analyzer is tuned to {center_frequency:.2f} MHz and the following parameters are set:\n"
        for name, value in self.parameters.items():
            if name not in used_keys:
                acq_plan += f"{name} = {value}\n"
        acq_plan += (
            f"\nThen, ${nffts} \times {fft_size}$ samples are acquired gap-free."
        )

        definitions = {
            "name": self.name,
            "center_frequency": center_frequency,
            "acquisition_plan": acq_plan,
            "fft_size": fft_size,
            "nffts": nffts,
        }

        # __doc__ refers to the module docstring at the top of the file
        return __doc__.format(**definitions)

    def get_sigmf_builder(self, measurement_result) -> SigMFBuilder:
        sigmf_builder = super().get_sigmf_builder(measurement_result)
        for i, detector in enumerate(M4sDetector):
            fft_annotation = FrequencyDomainDetectionAnnotation("fft_" + detector.name + "_power",
                                                                i * self.parameter_map["fft_size"], self.parameter_map["fft_size"])
            sigmf_builder.add_metadata_generator(
                type(fft_annotation).__name__ + '_' + "fft_" + detector.name + "_power", fft_annotation)
        return sigmf_builder
