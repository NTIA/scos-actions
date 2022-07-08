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
# scos-sensor/scripts/print_action_docstring.py. You can then paste that into
# the SCOS Markdown Editor (link below) to see the final rendering.
#
# Resources:
# - MathJax reference: https://math.meta.stackexchange.com/q/5020
# - Markdown reference: https://commonmark.org/help/
# - SCOS Markdown Editor: https://ntia.github.io/scos-md-editor/
#
r"""Perform a Y-Factor Calibration.
Supports calibration of gain and noise figure for one or more channels.
For each center frequency, sets the preselector to the noise diode path, turns
noise diode on, performs a mean FFT measurement, turns the noise diode off and
performs another mean FFT measurement. The mean power on and mean power off
data are used to compute the noise figure and gain. For each measurement, the
mean detector is applied over {nffts} {fft_size}-pt FFTs at {frequencies} MHz.

# {name}

## Signal analyzer setup and sample acquisition

Each time this task runs, the following process is followed:
{acquisition_plan}

## Time-domain processing

First, the ${nffts} \times {fft_size}$ continuous samples are acquired from
the signal analyzer. If specified, a voltage scaling factor is applied to the
complex time-domain signals. Then, the data is reshaped into a ${nffts} \times
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

$$w(n) = &0.2156 - 0.4160 \cos{{(2 \pi n / M)}} + 0.2781 \cos{{(4 \pi n / M)}}
         - &0.0836 \cos{{(6 \pi n / M)}} + 0.0069 \cos{{(8 \pi n / M)}}$$

where $M = {fft_size}$ is the number of points in the window, is applied to
each row of the matrix.

## Frequency-domain processing

### To-do: add details of FFT processing

## Y-Factor Method

### To-do: add details of Y-Factor method
"""

import logging
import os
import time
from numpy import ndarray
from scipy.constants import Boltzmann
from typing import Tuple

from scos_actions import utils
from scos_actions.actions.action_utils import get_param
from scos_actions.actions.interfaces.action import Action
from scos_actions.hardware import gps as mock_gps, preselector
from scos_actions.settings import sensor_calibration, SENSOR_CALIBRATION_FILE
from scos_actions.signal_processing.calibration import y_factor
from scos_actions.signal_processing.fft import (
    create_fft_detector,
    get_fft,
    get_fft_enbw,
    get_fft_window,
)
from scos_actions.signal_processing.power_analysis import (
    apply_power_detector,
    calculate_power_watts,
)
from scos_actions.signal_processing.unit_conversion import convert_dB_to_linear

logger = logging.getLogger(__name__)

RF_PATH = "rf_path"
NOISE_DIODE_ON = {RF_PATH: "noise_diode_on"}
NOISE_DIODE_OFF = {RF_PATH: "noise_diode_off"}

# Define parameter keys
FREQUENCY = "frequency"
SAMPLE_RATE = "sample_rate"
FFT_SIZE = "fft_size"
NUM_FFTS = "nffts"
NUM_SKIP = "nskip"


class YFactorCalibration(Action):
    """
    Perform a single- or stepped-frequency Y-factor calibration.

    The action will set any matching attributes found in the signal
    analyzer object. The following parameters are required by the
    action:

        name: name of the action
        frequency: center frequency in Hz
        fft_size: number of points in FFT (some 2^n)
        nffts: number of consecutive FFTs to pass to detector


    For the parameters required by the signal analyzer, see the
    documentation from the Python package for the signal analyzer
    being used.

    :param parameters: The dictionary of parameters needed for the
        action and the signal analyzer.
    :param sigan: instance of SignalAnalyzerInterface.
    """

    def __init__(self, parameters, sigan, gps=mock_gps):
        logger.debug("Initializing calibration action")
        super().__init__(parameters, sigan, gps)
        # FFT setup
        self.fft_detector = create_fft_detector("FftMeanDetector", ["mean"])
        self.fft_window_type = "flattop"

    def __call__(self, schedule_entry_json, task_id):
        """This is the entrypoint function called by the scheduler."""
        self.test_required_components()
        frequencies = self.parameter_map["frequency"]
        detail = ""
        if isinstance(frequencies, list):
            for i in range(len(frequencies)):
                iteration_params = utils.get_parameters(i, self.parameter_map)
                if i == 0:
                    detail += self.calibrate(iteration_params)
                else:
                    detail += os.linesep + self.calibrate(iteration_params)
        elif isinstance(frequencies, float):
            detail = self.calibrate(self.parameters)

        return detail

    def calibrate(self, params):
        # Set noise diode on
        logger.debug("Setting noise diode on")
        super().configure_preselector(NOISE_DIODE_ON)
        time.sleep(0.25)

        # Debugging
        logger.debug(
            "Before configuring, sigan preamp enable = " + str(self.sigan.preamp_enable)
        )

        # Configure signal analyzer
        self.sigan.preamp_enable = True
        super().configure_sigan(params)
        param_map = self.get_parameter_map(params)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Preamp = " + str(self.sigan.preamp_enable))
            logger.debug("Ref_level: " + str(self.sigan.reference_level))
            logger.debug("Attenuation:" + str(self.sigan.attenuation))
        # Get parameters from action config
        fft_size = get_param(FFT_SIZE, param_map)
        nffts = get_param(NUM_FFTS, param_map)
        nskip = get_param(NUM_SKIP, param_map)
        fft_window = get_fft_window(self.fft_window_type, fft_size)
        num_samples = fft_size * nffts

        logger.debug("acquiring mean FFT")

        # Get noise diode on mean FFT result
        noise_on_measurement_result = self.sigan.acquire_time_domain_samples(
            num_samples, num_samples_skip=nskip, gain_adjust=False
        )
        sample_rate = noise_on_measurement_result["sample_rate"]
        mean_on_watts = self.apply_mean_fft(
            noise_on_measurement_result, fft_size, fft_window, nffts
        )

        # Set noise diode off
        logger.debug("Setting noise diode off")
        self.configure_preselector(NOISE_DIODE_OFF)
        time.sleep(0.25)

        # Get noise diode off mean FFT result
        logger.debug("Acquiring noise off mean FFT")
        noise_off_measurement_result = self.sigan.acquire_time_domain_samples(
            num_samples, num_samples_skip=nskip, gain_adjust=False
        )
        mean_off_watts = self.apply_mean_fft(
            noise_off_measurement_result, fft_size, fft_window, nffts
        )

        # Y-Factor
        enbw_hz = get_fft_enbw(fft_window, sample_rate)
        enr_linear = self.get_linear_enr(cal_source_idx=0)
        temp_k, temp_c, _ = self.get_temperature()
        noise_figure, gain = y_factor(
            mean_on_watts, mean_off_watts, enr_linear, enbw_hz, temp_k
        )
        sensor_calibration.update(
            param_map,
            utils.get_datetime_str_now(),
            gain,
            noise_figure,
            temp_c,
            SENSOR_CALIBRATION_FILE,
        )

        # Debugging
        noise_floor = Boltzmann * temp_k * enbw_hz
        logger.debug(f"Noise floor: {noise_floor} Watts")
        logger.debug(f"Noise Figure: {noise_figure} dB")
        logger.debug(f"Gain: {gain} dB")

        return "Noise Figure:{}, Gain:{}".format(noise_figure, gain)

    def apply_mean_fft(
        self, measurement_result: dict, fft_size: int, fft_window: ndarray, nffts: int
    ) -> ndarray:
        complex_fft = get_fft(
            measurement_result["data"], fft_size, "backward", fft_window, nffts
        )
        power_fft = calculate_power_watts(complex_fft)
        mean_result = apply_power_detector(power_fft, self.fft_detector)
        return mean_result

    @staticmethod
    def get_linear_enr(cal_source_idx: int = 0) -> float:
        """
        Get the excess noise ratio of a calibration source.

        :param cal_source_idx: The index of the desired
            calibration source in preselector.cal_sources.
        :returns: The excess noise ratio of the specified
            calibration source, in linear units.
        """
        # todo deal with multiple cal sources
        # (this is only partially implemented for now, an error
        # is still raised if multiple cal sources exist)
        if len(preselector.cal_sources) == 0:
            raise Exception("No calibration sources defined in preselector.")
        elif len(preselector.cal_sources) > 1:
            raise Exception("Preselector contains multiple calibration sources.")
        else:
            enr_dB = preselector.cal_sources[cal_source_idx].enr
            enr_linear = convert_dB_to_linear(enr_dB)
            return enr_linear

    @property
    def description(self):

        if isinstance(get_param(FREQUENCY, self.parameter_map), float):
            frequencies = get_param(FREQUENCY, self.parameter_map) / 1e6
            nffts = get_param(NUM_FFTS, self.parameter_map)
            fft_size = get_param(FFT_SIZE, self.parameter_map)
        else:
            frequencies = utils.list_to_string(
                [f / 1e6 for f in get_param(FREQUENCY, self.parameter_map)]
            )
            nffts = utils.list_to_string(get_param(NUM_FFTS, self.parameter_map))
            fft_size = utils.list_to_string(get_param(FFT_SIZE, self.parameter_map))
        acq_plan = (
            f"Performs a y-factor calibration at frequencies: "
            f"{frequencies}, nffts:{nffts}, fft_size: {fft_size}\n"
        )
        definitions = {
            "name": self.name,
            "frequencies": frequencies,
            "acquisition_plan": acq_plan,
            "fft_size": fft_size,
            "nffts": nffts,
        }
        # __doc__ refers to the module docstring at the top of the file
        return __doc__.format(**definitions)

    # todo support multiple temperature sensors
    @staticmethod
    def get_temperature() -> Tuple[float, float, float]:
        """
        Get the temperature from a preselector sensor.

        :returns:
            - temp_k - Thing
            - temp_c
            - temp_f
        """
        kelvin_temp = 290.0
        celsius_temp = kelvin_temp - 273.15
        fahrenheit = (celsius_temp * 9.0 / 5.0) + 32
        temp = preselector.get_sensor_value(1)
        logger.debug("Temp: " + str(temp))
        if temp is None:
            logger.warning("Temperature is None. Using 290 K instead.")
        else:
            fahrenheit = float(temp)
            celsius_temp = (5.0 * (fahrenheit - 32)) / 9.0
            kelvin_temp = celsius_temp + 273.15
            logger.debug("Temperature: " + str(kelvin_temp))
        return kelvin_temp, celsius_temp, fahrenheit

    def test_required_components(self):
        """Fail acquisition if a required component is not available."""
        if not self.sigan.is_available:
            msg = "Acquisition failed: signal analyzer required but not " + "available"
            raise RuntimeError(msg)
