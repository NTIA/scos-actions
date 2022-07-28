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

### To-do: add details of FFT processing

## Y-Factor Method

### To-do: add details of Y-Factor method
"""

import logging
import time

from numpy import ndarray
from scipy.constants import Boltzmann
from scipy.signal import sosfilt

from scos_actions import utils
from scos_actions.hardware import gps as mock_gps
from scos_actions.settings import sensor_calibration
from scos_actions.settings import SENSOR_CALIBRATION_FILE
from scos_actions.actions.interfaces.action import Action
from scos_actions.utils import get_parameter
from scos_actions.signal_processing.fft import get_fft, get_fft_enbw, get_fft_window, get_fft_window_correction

from scos_actions.signal_processing.calibration import (
    get_linear_enr,
    get_temperature,
    y_factor,
)
from scos_actions.signal_processing.power_analysis import (
    apply_power_detector,
    calculate_power_watts,
    create_power_detector,
)

from scos_actions.signal_processing.unit_conversion import convert_watts_to_dBm
from scos_actions.signal_processing.filtering import generate_elliptic_iir_low_pass_filter

import os

logger = logging.getLogger(__name__)

RF_PATH = 'rf_path'
NOISE_DIODE_ON = {RF_PATH: 'noise_diode_on'}
NOISE_DIODE_OFF = {RF_PATH: 'noise_diode_off'}

# Define parameter keys
FREQUENCY = "frequency"
SAMPLE_RATE = "sample_rate"
FFT_SIZE = "fft_size"
NUM_FFTS = "nffts"
NUM_SKIP = "nskip"
# TODO: Should calibration source index and temperature sensor number
# be required parameters?


class YFactorCalibration(Action):
    """Perform a single- or stepped-frequency Y-factor calibration.

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
        logger.debug('Initializing calibration action')
        super().__init__(parameters, sigan, gps)
        # Specify calibration source and temperature sensor indices
        # TODO: Should these be part of the action config?
        self.cal_source_idx = 0
        self.temp_sensor_idx = 1
        # FFT setup
        self.fft_detector = create_power_detector("MeanDetector", ["mean"])
        self.fft_window_type = "flattop"
        # IIR Filter
        # Hard-coded parameters for now
        self.iir_sos = generate_elliptic_iir_low_pass_filter(0.1, 40, 5e6, 8e3, 14e6)

    def __call__(self, schedule_entry_json, task_id):
        """This is the entrypoint function called by the scheduler."""
        self.test_required_components()
        detail = ''
        # iteration_params is iterable even if it contains only one set of parameters
        # it is also sorted by frequency from low to high
        iteration_params = utils.get_iterable_parameters(self.parameters)
        
        # Calibrate
        for i, p in enumerate(iteration_params):
            if i == 0:
                detail += self.calibrate(p)
            else:
                detail += os.linesep + self.calibrate(p)

        return detail


    def calibrate(self, params):
        # Configure signal analyzer
        super().configure_sigan(params)
        logger.debug('Preamp = ' + str(self.sigan.preamp_enable))
        logger.debug('Ref_level: ' + str(self.sigan.reference_level))
        logger.debug('Attenuation:' + str(self.sigan.attenuation))

        # Get parameters from action config
        fft_size = get_parameter(FFT_SIZE, params)
        nffts = get_parameter(NUM_FFTS, params)
        nskip = get_parameter(NUM_SKIP, params)
        fft_window = get_fft_window(self.fft_window_type, fft_size)
        fft_acf = get_fft_window_correction(fft_window, 'amplitude')
        num_samples = fft_size * nffts
        
        # Set noise diode on
        logger.debug('Setting noise diode on')
        super().configure_preselector(NOISE_DIODE_ON)
        time.sleep(.25)

        # Get noise diode on IQ
        logger.debug("Acquiring IQ samples with noise diode ON")
        noise_on_measurement_result = self.sigan.acquire_time_domain_samples(
            num_samples, num_samples_skip=nskip, gain_adjust=False
        )
        sample_rate = noise_on_measurement_result["sample_rate"]

        # Set noise diode off
        logger.debug('Setting noise diode off')
        self.configure_preselector(NOISE_DIODE_OFF)
        time.sleep(.25)

        # Get noise diode off IQ
        logger.debug('Acquiring IQ samples with noise diode OFF')
        noise_off_measurement_result = self.sigan.acquire_time_domain_samples(
            num_samples, num_samples_skip=nskip, gain_adjust=False
        )
        

        # Apply IIR filtering to both captures
        logger.debug("Applying IIR filter to IQ captures")
        noise_on_filtered = sosfilt(self.iir_sos, noise_on_measurement_result["data"])
        noise_off_filtered = sosfilt(self.iir_sos, noise_off_measurement_result["data"])

        # Get power values in time domain
        td_on_watts = calculate_power_watts(noise_on_filtered) / 2. # Divide by 2 for RF/baseband conversion
        td_off_watts = calculate_power_watts(noise_off_filtered) / 2.

        # Get mean power FFT results
        fft_on_watts = self.apply_mean_fft(
            noise_on_filtered, fft_size, fft_window, nffts, fft_acf
        )
        fft_off_watts = self.apply_mean_fft(
            noise_off_filtered, fft_size, fft_window, nffts, fft_acf
        )

        # Y-Factor
        enbw_hz = get_fft_enbw(fft_window, sample_rate)
        enbw_hz_td = 11.607e6  # TODO Parameterize this
        enr_linear = get_linear_enr(self.cal_source_idx)
        temp_k, temp_c, _ = get_temperature(self.temp_sensor_idx)
        td_noise_figure, td_gain = y_factor(
            td_on_watts, td_off_watts, enr_linear, sample_rate, temp_k
        )
        fft_noise_figure, fft_gain = y_factor(
            fft_on_watts, fft_off_watts, enr_linear, enbw_hz, temp_k
        )

        # Don't update the sensor calibration while testing
        # sensor_calibration.update(
        #     params,
        #     utils.get_datetime_str_now(),
        #     gain,
        #     noise_figure,
        #     temp_c,
        #     SENSOR_CALIBRATION_FILE,
        # )

        # Debugging
        noise_floor_dBm = convert_watts_to_dBm(Boltzmann * temp_k * enbw_hz)
        logger.debug(f'Noise floor: {noise_floor_dBm:.2f} dBm')
        logger.debug(f'Noise Figure (FFT): {fft_noise_figure:.2f} dB')
        logger.debug(f'Gain (FFT): {fft_gain:.2f} dB')
        logger.debug(f'Noise figure (TD): {td_noise_figure:.2f} dB')
        logger.debug(f"Gain (TD): {td_gain:.2f} dB")
        
        # Detail results contain only FFT version of result for now
        return 'Noise Figure: {}, Gain: {}'.format(fft_noise_figure, fft_gain)

    def apply_mean_fft(
        self, iqdata: ndarray, fft_size: int, fft_window: ndarray, nffts: int, fft_window_cf: float
    ) -> ndarray:
        complex_fft = get_fft(
            time_data=iqdata,
            fft_size=fft_size,
            norm="forward",
            fft_window=fft_window,
            num_ffts=nffts,
            shift=True
        )
        # TESTING SCALING
        power_fft /= 2  # RF/baseband conversion
        complex_fft *= fft_window_cf  # Window correction
        power_fft = calculate_power_watts(complex_fft)
        mean_result = apply_power_detector(power_fft, self.fft_detector)
        return mean_result

    @property
    def description(self):
        # Get parameters; they may be single values or lists
        frequencies = get_parameter(FREQUENCY, self.parameters)
        nffts = get_parameter(NUM_FFTS, self.parameters)
        fft_size = get_parameter(FFT_SIZE, self.parameters)

        # Convert parameter lists to strings if needed
        if isinstance(frequencies, list):
            frequencies = utils.list_to_string(
                [f / 1e6 for f in get_parameter(FREQUENCY, self.parameters)]
            )
        if isinstance(nffts, list):
            nffts = utils.list_to_string(get_parameter(NUM_FFTS, self.parameters))
        if isinstance(fft_size, list):
            fft_size = utils.list_to_string(get_parameter(FFT_SIZE, self.parameters))
        
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

    def test_required_components(self):
        """Fail acquisition if a required component is not available."""
        if not self.sigan.is_available:
            msg = "acquisition failed: signal analyzer required but not available"
            raise RuntimeError(msg)


