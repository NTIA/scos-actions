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
r"""Apply M4S detector over {nffts} {fft_size}-pt FFTs at {center_frequency:.2f} MHz.

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

from numpy import float32, ndarray

from scos_actions.actions.interfaces.measurement_action import MeasurementAction
from scos_actions.metadata.structs import ntia_algorithm
from scos_actions.signal_processing.fft import (
    get_fft,
    get_fft_enbw,
    get_fft_frequencies,
    get_fft_window,
    get_fft_window_correction,
)
from scos_actions.signal_processing.power_analysis import (
    apply_statistical_detector,
    calculate_power_watts,
    create_statistical_detector,
)
from scos_actions.signal_processing.unit_conversion import (
    convert_linear_to_dB,
    convert_watts_to_dBm,
)
from scos_actions.utils import get_parameter

logger = logging.getLogger(__name__)

# Define parameter keys
FREQUENCY = "frequency"
SAMPLE_RATE = "sample_rate"
NUM_SKIP = "nskip"
NUM_FFTS = "nffts"
FFT_SIZE = "fft_size"
CLASSIFICATION = "classification"
CAL_ADJUST = "calibration_adjust"


class SingleFrequencyFftAcquisition(MeasurementAction):
    """Perform M4S detection over requested number of single-frequency FFTs.

    The action will set any matching attributes found in the signal
    analyzer object. The following parameters are required by the action:

        name: name of the action
        frequency: center frequency in Hz
        fft_size: number of points in FFT (some 2^n)
        nffts: number of consecutive FFTs to pass to detector

    For the parameters required by the signal analyzer, see the
    documentation from the Python package for the signal analyzer being
    used.

    :param parameters: The dictionary of parameters needed for the
        action and the signal analyzer.
    :param sigan: Instance of SignalAnalyzerInterface.
    """

    def __init__(self, parameters: dict):
        super().__init__(parameters)
        # Pull parameters from action config
        self.fft_size = get_parameter(FFT_SIZE, self.parameters)
        self.nffts = get_parameter(NUM_FFTS, self.parameters)
        self.nskip = get_parameter(NUM_SKIP, self.parameters)
        self.frequency_Hz = get_parameter(FREQUENCY, self.parameters)
        self.classification = get_parameter(CLASSIFICATION, self.parameters)
        self.cal_adjust = get_parameter(CAL_ADJUST, self.parameters)
        assert isinstance(self.cal_adjust, bool)
        # FFT setup
        self.fft_detector = create_statistical_detector(
            "M4sDetector", ["min", "max", "mean", "median", "sample"]
        )
        self.fft_window_type = "flattop"
        self.num_samples = self.fft_size * self.nffts
        self.fft_window = get_fft_window(self.fft_window_type, self.fft_size)
        self.fft_window_acf = get_fft_window_correction(self.fft_window, "amplitude")

    def execute(self, schedule_entry: dict, task_id: int) -> dict:
        # Acquire IQ data and generate M4S result
        measurement_result = self.acquire_data(
            self.num_samples, self.nskip, self.cal_adjust, cal_params=self.parameters
        )
        # Actual sample rate may differ from configured value
        sample_rate_Hz = measurement_result["sample_rate"]
        m4s_result = self.apply_m4s(measurement_result)

        # Save measurement results
        measurement_result["data"] = m4s_result
        measurement_result.update(self.parameters)
        measurement_result["task_id"] = task_id
        measurement_result["classification"] = self.classification

        # Build capture metadata
        sigan_settings = self.get_sigan_settings(measurement_result)
        logger.debug(f"sigan settings:{sigan_settings}")
        measurement_result["duration_ms"] = round(
            (self.num_samples / sample_rate_Hz) * 1000
        )
        measurement_result["capture_segment"] = self.create_capture_segment(
            sample_start=0,
            sigan_settings=sigan_settings,
            measurement_result=measurement_result,
        )

        return measurement_result

    def apply_m4s(self, measurement_result: dict) -> ndarray:
        # IQ samples already scaled based on calibration
        # 'forward' normalization applies 1/fft_size normalization
        complex_fft = get_fft(
            measurement_result["data"],
            self.fft_size,
            "forward",
            self.fft_window,
            self.nffts,
        )
        power_fft = calculate_power_watts(complex_fft)
        m4s_result = apply_statistical_detector(power_fft, self.fft_detector, float32)
        m4s_result = convert_watts_to_dBm(m4s_result)
        # Scaling applied:
        #   RF/Baseband power conversion (-3 dB)
        #   FFT window amplitude correction
        m4s_result -= 3
        m4s_result += 2.0 * convert_linear_to_dB(self.fft_window_acf)
        return m4s_result

    @property
    def description(self):
        frequency_MHz = self.frequency_Hz / 1e6
        used_keys = [FREQUENCY, NUM_FFTS, FFT_SIZE, "name"]
        acq_plan = (
            f"The signal analyzer is tuned to {frequency_MHz:.2f} MHz"
            f" and the following parameters are set:\n"
        )
        for name, value in self.parameters.items():
            if name not in used_keys:
                acq_plan += f"{name} = {value}\n"
        acq_plan += (
            f"\nThen, ${self.nffts} \times {self.fft_size}$ samples "
            "are acquired gap-free."
        )

        definitions = {
            "name": self.name,
            "center_frequency": frequency_MHz,
            "acquisition_plan": acq_plan,
            "fft_size": self.fft_size,
            "nffts": self.nffts,
        }

        # __doc__ refers to the module docstring at the top of the file
        return __doc__.format(**definitions)

    def create_metadata(self, measurement_result: dict, recording: int = None) -> None:
        super().create_metadata(measurement_result, recording)
        dft_obj = ntia_algorithm.DFT(
            id="fft_1",
            equivalent_noise_bandwidth=get_fft_enbw(
                self.fft_window, measurement_result["sample_rate"]
            ),
            samples=self.fft_size,
            dfts=self.nffts,
            window=self.fft_window_type,
            baseband=False,
            description="Discrete Fourier transform computed using the FFT algorithm",
        )
        frequencies = get_fft_frequencies(
            self.fft_size, measurement_result["sample_rate"], self.frequency_Hz
        )
        m4s_graph = ntia_algorithm.Graph(
            name="M4S Detector Result",
            series=[det.value for det in self.fft_detector],
            length=self.fft_size,
            x_units="Hz",
            x_start=[frequencies[0]],
            x_stop=[frequencies[-1]],
            x_step=[frequencies[1] - frequencies[0]],
            y_units="dBm",
            reference=measurement_result["reference"],
            description=(
                "Results of min, max, mean, and median statistical detectors, "
                + f"along with a random sampling, from a set of {self.nffts} "
                + f"DFTs, each of length {self.fft_size}, computed from IQ data."
            ),
        )

        self.sigmf_builder.set_processing([dft_obj.id])
        self.sigmf_builder.set_processing_info([dft_obj])
        self.sigmf_builder.set_data_products([m4s_graph])

    def is_complex(self) -> bool:
        return False
