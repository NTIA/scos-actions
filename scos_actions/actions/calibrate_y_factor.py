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
noise diode on, performs and M4 measurement, turns the noise diode off and
performs another M4 measurements. Uses the mean power on and mean power off
data to compute the noise figure and gain. For each M4 measurement, it applies
an M4S detector over {nffts} {fft_size}-pt FFTs at {frequencies} MHz.

# {name}

## Radio setup and sample acquisition

Each time this task runs, the following process is followed:
{acquisition_plan}

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

$$w(n) = &0.2156 - 0.4160 \cos{{(2 \pi n / M)}} + 0.2781 \cos{{(4 \pi n / M)}}
         - &0.0836 \cos{{(6 \pi n / M)}} + 0.0069 \cos{{(8 \pi n / M)}}$$

where $M = {fft_size}$ is the number of points in the window, is applied to
each row of the matrix.
"""

import logging
import time
from numpy.typing import NDArray
from scipy.constants import Boltzmann
from scos_actions.signal_processing.calibration import y_factor
from scos_actions import utils
from scos_actions.hardware import gps as mock_gps
from scos_actions.settings import sensor_calibration
from scos_actions.settings import SENSOR_CALIBRATION_FILE
from scos_actions.actions.interfaces.action import Action
from scos_actions.actions.action_utils import get_param
from scos_actions.actions.fft import (
    create_fft_detector, apply_fft_detector, get_fft, get_fft_window,
    get_fft_enbw
)
from scos_actions.actions.power_analysis import (convert_volts_to_watts)
from scos_actions.hardware import preselector
import os

logger = logging.getLogger(__name__)

RF_PATH = 'rf_path'
NOISE_DIODE_ON = {RF_PATH: 'noise_diode_on'}
NOISE_DIODE_OFF = {RF_PATH: 'noise_diode_off'}
SAMPLE_RATE = 'sample_rate'
FFT_SIZE = 'fft_size'


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
        logger.debug('Initializing calibration action')
        super().__init__(parameters, sigan, gps)
        # Pull parameters from action config
        self.fft_size = get_param('fft_size', self.parameter_map)
        self.nffts = get_param('nffts', self.parameter_map)
        self.nskip = get_param('nskip', self.parameter_map)
        # FFT setup
        self.fft_detector = create_fft_detector('FftMeanDetector', ['mean'])
        self.fft_window_type = 'flattop'
        self.fft_window = get_fft_window(self.fft_window_type, self.fft_size)
        self.num_samples = self.fft_size * self.nffts

    def __call__(self, schedule_entry_json, task_id):
        """This is the entrypoint function called by the scheduler."""
        self.test_required_components()
        start_time = utils.get_datetime_str_now()
        frequencies = self.parameter_map['frequency']
        detail = ''
        if isinstance(frequencies, list):
            for i in range(len(frequencies)):
                iteration_params = utils.get_parameters(i, self.parameter_map)
                if i == 0:
                    detail += self.calibrate(iteration_params)
                else:
                    detail += os.linesep + self.calibrate(iteration_params)
        elif isinstance(frequencies, float):
            detail = self.calibrate(self.parameters)

        end_time = utils.get_datetime_str_now()
        return detail

    def calibrate(self, params):
        # Set noise diode on
        logger.debug('Setting noise diode on')
        super().configure_preselector(NOISE_DIODE_ON)
        time.sleep(.25)

        # Debugging
        logger.debug('Before configuring, sigan preamp enable = '
                     + str(self.sigan.preamp_enable))

        # Configure signal analyzer
        self.sigan.preamp_enable = True
        super().configure_sigan(params)
        param_map = self.get_parameter_map(params)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('Preamp = ' + str(self.sigan.preamp_enable))
            logger.debug('Ref_level: ' + str(self.sigan.reference_level))
            logger.debug('Attenuation:' + str(self.sigan.attenuation))
        logger.debug('acquiring mean FFT')

        # Get noise diode on mean FFT result
        noise_on_measurement_result = self.sigan.acquire_time_domain_samples(
            self.num_samples, num_samples_skip=self.nskip, gain_adjust=False
        )
        sample_rate = noise_on_measurement_result['sample_rate']
        mean_on_watts = self.apply_mean_fft(noise_on_measurement_result)

        # Set noise diode off
        logger.debug('Setting noise diode off')
        self.configure_preselector(NOISE_DIODE_OFF)
        time.sleep(.25)

        # Get noise diode off mean FFT result
        logger.debug('Acquiring noise off mean FFT')
        noise_off_measurement_result = self.sigan.acquire_time_domain_samples(
            self.num_samples, num_samples_skip=self.nskip, gain_adjust=False
        )
        mean_off_watts = self.apply_mean_fft(noise_off_measurement_result)

        # Y-Factor
        enbw = get_fft_enbw(self.fft_window, sample_rate)
        enr = self.get_enr()
        temp_k, temp_c, _ = self.get_temperature()
        noise_figure, gain = y_factor(mean_on_watts, mean_off_watts, enr, enbw,
                                      T_room=temp_k)
        sensor_calibration.update(param_map, utils.get_datetime_str_now(),
                                  gain, noise_figure, temp_c,
                                  SENSOR_CALIBRATION_FILE)

        # Debugging
        noise_floor = Boltzmann * temp_k * enbw
        logger.debug('Noise floor: ' + str(noise_floor))
        enr = self.get_enr()
        logger.debug('ENR: ' + str(enr))
        logger.debug('Noise Figure:' + str(noise_figure))
        logger.debug('Gain: ' + str(gain))

        return 'Noise Figure:{}, Gain:{}'.format(noise_figure, gain)

    def apply_mean_fft(self, measurement_result: dict) -> NDArray:
        complex_fft = get_fft(measurement_result['data'], self.fft_size,
                              self.fft_window, self.nffts)
        power_fft = convert_volts_to_watts(complex_fft)
        mean_result = apply_fft_detector(power_fft, self.fft_detector)
        return mean_result

    def get_enr(self):
        # todo deal with multiple cal sources
        if len(preselector.cal_sources) == 0:
            raise Exception('No calibrations sources defined in preselector.')
        elif len(preselector.cal_sources) > 1:
            raise Exception(
                'Preselector contains multiple calibration sources.'
            )
        else:
            enr_dB = preselector.cal_sources[0].enr
            # enr_dB = 14.53
            linear_enr = 10 ** (enr_dB / 10.0)
            return linear_enr

    @property
    def description(self):

        if isinstance(self.parameter_map['frequency'], float):
            frequencies = self.parameter_map["frequency"] / 1e6
            nffts = self.parameter_map["nffts"]
            fft_size = self.parameter_map["fft_size"]
        else:
            frequencies = utils.list_to_string(self.parameter_map['frequency'])
            nffts = utils.list_to_string(self.parameter_map["nffts"])
            fft_size = utils.list_to_string(self.parameter_map["fft_size"])
        acq_plan = f"Performs a y-factor calibration at frequencies: " \
                   f"{frequencies}, nffts:{nffts}, fft_size: {fft_size}\n"
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
    def get_temperature(self):
        kelvin_temp = 290.0
        celsius_temp = kelvin_temp - 273.15
        fahrenheit = (celsius_temp * 9 / 5) + 32
        temp = preselector.get_sensor_value(1)
        logger.debug('Temp: ' + str(temp))
        if temp is None:
            logger.warning('Temperature is None. Using 290 K instead.')
        else:
            fahrenheit = float(temp)
            celsius_temp = ((5.0 * (fahrenheit - 32)) / 9.0)
            kelvin_temp = celsius_temp + 273.15
            logger.debug('Temperature: ' + str(kelvin_temp))
        return kelvin_temp, celsius_temp, fahrenheit

    def test_required_components(self):
        """Fail acquisition if a required component is not available."""
        if not self.sigan.is_available:
            msg = "Acquisition failed: signal analyzer required but not " \
                  + "available"
            raise RuntimeError(msg)
