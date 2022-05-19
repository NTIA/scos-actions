import numpy as np
from scipy.constants import Boltzmann as k_b
import logging


logger = logging.getLogger(__name__)

def y_factor(pwr_noise_on_watts, pwr_noise_off_watts, ENR, ENBW, T_room=290.):
    # Y-Factor calculations (element-wise from power arrays)
    logger.debug('ENR:{}'.format(ENR))
    logger.debug('ENBW:{}'.format(ENBW))
    logger.debug('mean power on: {}'.format(np.mean(pwr_noise_on_watts)))
    logger.debug('mean power off: {}'.format(np.mean(pwr_noise_off_watts)))
    y = pwr_noise_on_watts / pwr_noise_off_watts
    noise_factor = ENR / (y - 1)
    gain_watts = pwr_noise_on_watts / (k_b * T_room * ENBW * (ENR + noise_factor))
    # Get mean values from arrays and convert to dB
    noise_figure = 10. * np.log10(np.mean(noise_factor))  # dB
    gain = 10. * np.log10(np.mean(gain_watts))
    return noise_figure, gain

