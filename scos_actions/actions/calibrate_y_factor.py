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
Apply m4s detector over {nffts} {fft_size}-pt FFTs at {frequencies} MHz.

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

$$w(n) = &0.2156 - 0.4160 \cos{{(2 \pi n / M)}} + 0.2781 \cos{{(4 \pi n / M)}} -
         &0.0836 \cos{{(6 \pi n / M)}} + 0.0069 \cos{{(8 \pi n / M)}}$$

where $M = {fft_size}$ is the number of points in the window, is applied to
each row of the matrix.



"""
import copy
import logging
import time

from scipy.signal import windows
from scos_actions.signal_processing.utils import get_enbw
from scos_actions.signal_processing.calibration import y_factor
from scos_actions.signal_processing.utils import dbm_to_watts
from scos_actions import utils
from scos_actions.hardware import gps as mock_gps
from scos_actions.settings import sensor_calibration
from scos_actions.settings import SENSOR_CALIBRATION_FILE
from scos_actions.hardware import preselector
from scos_actions.actions.acquire_single_freq_fft import (
    SingleFrequencyFftAcquisition
)
from scos_actions.hardware import preselector
import os

logger = logging.getLogger(__name__)

RF_PATH = 'rf_path'
NOISE_DIODE_ON = {RF_PATH: 'noise_diode_on'}
NOISE_DIODE_OFF = {RF_PATH: 'noise_diode_off'}
SAMPLE_RATE = 'sample_rate'
FFT_SIZE = 'fft_size'

logger = logging.getLogger(__name__)


class YFactorCalibration(SingleFrequencyFftAcquisition):
    """Perform a single or stepped y-factor calibration.

    :param parameters: The dictionary of parameters needed for the action and the radio.

    The action will set any matching attributes found in the radio object. The following
    parameters are required by the action:

        name: name of the action
        frequency: center frequency in Hz
        fft_size: number of points in FFT (some 2^n)
        nffts: number of consecutive FFTs to pass to detector


    For the parameters required by the radio, see the documentation for the radio being used.

    :param radio: instance of RadioInterface
    """

    def __init__(self, parameters, sigan, gps=mock_gps):
        logger.debug('Initializing calibration action')
        super().__init__(parameters, sigan, gps)

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
        logger.info('Setting noise diode on')
        super().configure_preselector(NOISE_DIODE_ON)
        time.sleep(.25)
        logger.info('Before configure, Preamp = ' + str(self.sigan.preamp_enable))
        self.sigan.preamp_enable = True
        super().configure_sigan(params)
        param_map = self.get_parameter_map(params)
        logger.info('Preamp = ' + str(self.sigan.preamp_enable))
        logger.info('Ref_level: ' + str(self.sigan.reference_level))
        logger.info('Attenuation:' + str(self.sigan.attenuation))
        logger.info('acquiring m4')
        noise_on_measurement_result = super().acquire_data(param_map, apply_gain=False)
        mean_on_power_dbm = noise_on_measurement_result['data'][2]
        logger.info('Setting noise diode off')
        self.configure_preselector(NOISE_DIODE_OFF)
        time.sleep(.25)
        logger.info('Acquiring noise off M4')
        measurement_result = super().acquire_data(param_map, apply_gain=False)
        mean_off_power_dbm = measurement_result['data'][2]
        mean_on_watts = dbm_to_watts(mean_on_power_dbm)
        mean_off_watts = dbm_to_watts(mean_off_power_dbm)
        import numpy as np
        logger.info('Mean on dBm: ' + str(np.mean(mean_on_power_dbm)))
        logger.info('Mean off dBm:' + str(np.mean(mean_off_power_dbm)))
        window = windows.flattop(param_map[FFT_SIZE])
        enbw = get_enbw(window, param_map[SAMPLE_RATE])
        noise_floor = 1.38e-23 * 300 * enbw
        logger.info('Noise floor: ' + str(noise_floor))
        enr = self.get_enr()
        logger.info('ENR: ' + str(enr))
        temperature = self.get_temperature()
        noise_figure, gain = y_factor(mean_on_watts, mean_off_watts, enr, enbw, T_room=temperature)
        logger.info('Noise Figure:' + str(noise_figure))
        logger.info('Gain: ' + str(gain))
        sensor_calibration.update(param_map, utils.get_datetime_str_now(), gain, noise_figure, temperature, SENSOR_CALIBRATION_FILE)
        return 'Noise Figure:{}, Gain:{}'.format(noise_figure, gain)

    def get_enr(self):
        # todo deal with multiple cal sources
        if len(preselector.cal_sources) == 0:
            raise Exception('No calibrations sources defined in preselector.')
        elif len(preselector.cal_sources) > 1:
            raise Exception('Preselector contains multiple calibration sources.')
        else:
            enr_dB = preselector.cal_sources[0].enr
            # enr_dB = 14.53
            linear_enr = 10 ** (enr_dB / 10.0)
            return linear_enr

    @property
    def description(self):

        if(isinstance(self.parameter_map['frequency'], float)):
            frequencies = self.parameter_map["frequency"] / 1e6
            nffts = self.parameter_map["nffts"]
            fft_size = self.parameter_map["fft_size"]
        else:
            frequencies = utils.list_to_string(self.parameter_map['frequency'])
            nffts = utils.list_to_string(self.parameter_map["nffts"])
            fft_size = utils.list_to_string(self.parameter_map["fft_size"])
        acq_plan = f"Performs a y-factor calibration at frequencies: {frequencies}, nffts:{nffts}, fft_size: {fft_size}\n"
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
        temp = preselector.get_sensor_value(1)
        logger.info('Temp: ' + str(temp))
        if temp is None:
            logger.warning('Temperature is None. Using 290. instead.')
        else:
            kelvin_temp = ((5.0 * (float(temp) - 32)) / 9.0 ) + 273.15
            logger.info('Temperature: ' + str(kelvin_temp))
        return kelvin_temp
