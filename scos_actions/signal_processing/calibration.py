import logging
from typing import Optional, Tuple

import numpy as np
from its_preselector.preselector import Preselector
from scipy.constants import Boltzmann
from scos_actions.signal_processing.unit_conversion import (
    convert_celsius_to_fahrenheit,
    convert_celsius_to_kelvins,
    convert_dB_to_linear,
    convert_linear_to_dB,
    convert_watts_to_dBm,
)

logger = logging.getLogger(__name__)


class CalibrationException(Exception):
    """Basic exception handling for calibration functions."""

    def __init__(self, msg):
        super().__init__(msg)


def y_factor(
    pwr_noise_on_watts: np.ndarray,
    pwr_noise_off_watts: np.ndarray,
    enr_linear: float,
    enbw_hz: float,
    temp_kelvins: float = 300.0,
) -> Tuple[float, float]:
    """
    Perform Y-Factor calculations of noise figure and gain.

    Noise factor and linear gain are computed from the input
    arrays using the Y-Factor method. The linear values are
    then averaged and converted to dB.

    :param pwr_noise_on_watts: Array of power values, in Watts,
        recorded with the calibration noise source on.
    :param pwr_noise_off_watts: Array of power values, in Watts,
        recorded with the calibration noise source off.
    :param enr_linear: Calibration noise source excess noise
        ratio, in linear units.
    :param enbw_hz: Equivalent noise bandwidth, in Hz.
    :param temp_kelvins: Temperature, in Kelvins. If not given,
        a default value of 300 K is used.
    :return: A tuple (noise_figure, gain) containing the calculated
        noise figure and gain, both in dB, from the Y-factor method.
    """
    mean_on_dBm = convert_watts_to_dBm(np.mean(pwr_noise_on_watts))
    mean_off_dBm = convert_watts_to_dBm(np.mean(pwr_noise_off_watts))
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"ENR: {convert_linear_to_dB(enr_linear)} dB")
        logger.debug(f"ENBW: {enbw_hz} Hz")
        logger.debug(f"Mean power on: {mean_on_dBm:.2f} dBm")
        logger.debug(f"Mean power off: {mean_off_dBm:.2f} dBm")
    y = convert_dB_to_linear(mean_on_dBm - mean_off_dBm)
    noise_factor = enr_linear / (y - 1.0)
    gain_dB = mean_on_dBm - convert_watts_to_dBm(
        Boltzmann * temp_kelvins * enbw_hz * (enr_linear + noise_factor)
    )
    noise_figure_dB = convert_linear_to_dB(noise_factor)
    return noise_figure_dB, gain_dB


def get_linear_enr(
    preselector: Preselector, cal_source_idx: Optional[int] = None
) -> float:
    """
    Get the excess noise ratio of a calibration source.

    Specifying `cal_source_idx` is optional as long as there is
    only one calibration source. It is required if multiple
    calibration sources are present.

    The preselector is loaded from `scos_actions.hardware`.

    :param preselector: The sensor preselector
    :param cal_source_idx: The index of the specified
        calibration source in `preselector.cal_sources`.
    :return: The excess noise ratio of the specified
        calibration source, in linear units.
    :raises CalibrationException: If multiple calibration sources are
        available but `cal_source_idx` is not specified.
    :raises IndexError: If the specified calibration source
        index is out of range for the current preselector.
    """
    if len(preselector.cal_sources) == 0:
        raise CalibrationException("No calibration sources defined in preselector.")
    elif len(preselector.cal_sources) == 1 and cal_source_idx is None:
        # Default to the only cal source available
        cal_source_idx = 0
    elif len(preselector.cal_sources) > 1 and cal_source_idx is None:
        # Must specify index if multiple sources available
        raise CalibrationException(
            "Preselector contains multiple calibration sources, "
            + "and the source index was not specified."
        )
    try:
        enr_dB = preselector.cal_sources[cal_source_idx].enr
    except IndexError:
        raise IndexError(
            f"Calibration source index {cal_source_idx} out of range "
            + "while trying to get ENR value."
        )
    enr_linear = convert_dB_to_linear(enr_dB)
    return enr_linear


def get_temperature(
    preselector: Preselector, sensor_idx: Optional[int] = None
) -> Tuple[float, float, float]:
    """
    Get the temperature from a preselector sensor.

    The preselector is expected to be configured to return the
    temperature in degrees Celsius.

    :param preselector: The sensor preselector.
    :param sensor_idx: The index of the desired temperature
        sensor in the preselector.
    :raises CalibrationException: If no sensor index is provided, or
        if no value is returned after querying the sensor.
    :return: A tuple of floats (temp_k, temp_c, temp_f) containing
        the retrieved temperature in Kelvins, degrees Celsius, and
        degrees Fahrenheit, respectively.
    """
    if sensor_idx is None:
        raise CalibrationException("Temperature sensor index not specified.")
    temp = preselector.get_sensor_value(sensor_idx)
    if temp is None:
        raise CalibrationException("Failed to get temperature from sensor.")
    logger.debug(f"Got temperature from sensor: {temp} deg. Celsius")
    temp_c = float(temp)
    temp_f = convert_celsius_to_fahrenheit(temp_c)
    temp_k = convert_celsius_to_kelvins(temp_c)
    return temp_k, temp_c, temp_f
