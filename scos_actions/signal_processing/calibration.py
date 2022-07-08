import logging
from typing import Tuple

import numpy as np
from numpy import ndarray
from scipy.constants import Boltzmann

from scos_actions.signal_processing.unit_conversion import convert_linear_to_dB

logger = logging.getLogger(__name__)


def y_factor(
    pwr_noise_on_watts: ndarray,
    pwr_noise_off_watts: ndarray,
    enr_linear: float,
    enbw_hz: float,
    temp_kelvins: float = 300.0,
) -> Tuple[float, float]:
    """
    Perform Y-Factor calculations of noise figure and gain.

    Noise factor and linear gain are computed element-wise from
    the input arrays using the Y-Factor method. The linear values
    are then averaged and converted to dB.

    :param pwr_noise_on_watts: Array of power values, in Watts,
        recorded with the calibration noise source on.
    :param pwr_noise_off_watts: Array of power values, in Watts,
        recorded with the calibration noise source off.
    :param enr_linear: Calibration noise source excess noise
        ratio, in linear units.
    :param enbw_hz: Equivalent noise bandwidth, in Hz.
    :param temp_kelvins: Temperature, in Kelvins. If not given,
        a default value of 300 K is used.
    """
    logger.debug(f"ENR: {convert_linear_to_dB(enr_linear)} dB")
    logger.debug(f"ENBW: {enbw_hz} Hz")
    logger.debug(f"Mean power on: {np.mean(pwr_noise_on_watts)} W")
    logger.debug(f"Mean power off: {np.mean(pwr_noise_off_watts)} W")
    y = pwr_noise_on_watts / pwr_noise_off_watts
    noise_factor = enr_linear / (y - 1.0)
    gain_watts = pwr_noise_on_watts / (
        Boltzmann * temp_kelvins * enbw_hz * (enr_linear + noise_factor)
    )
    # Get mean values from arrays and convert to dB
    noise_figure = convert_linear_to_dB(np.mean(noise_factor))
    gain = convert_linear_to_dB(np.mean(gain_watts))
    return noise_figure, gain
