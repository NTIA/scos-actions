import os
import warnings
from typing import Union

import numexpr as ne
import numpy as np

from scos_actions.signal_processing import NUMEXPR_THRESHOLD


def suppress_divide_by_zero_when_testing():
    # If testing, don't output divide-by-zero warnings from log10
    # This handles NumPy and NumExpr warnings
    if "PYTEST_CURRENT_TEST" in os.environ:
        warnings.filterwarnings("ignore", message="divide by zero")
        np_error_settings_savepoint = np.seterr(divide="ignore")


def convert_watts_to_dBm(
    val_watts: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Convert from Watts to dBm.

    Calculation: ``10 * log10(val_watts) + 30``

    NumPy is used for scalar inputs and small arrays.
    NumExpr is used to speed up the operation for large arrays.

    :param val_watts: A value, or array of values, in Watts.
    :return: The input val_watts, converted to dBm.
    """
    suppress_divide_by_zero_when_testing()
    if np.isscalar(val_watts) or val_watts.size < NUMEXPR_THRESHOLD:
        val_dBm = 10.0 * np.log10(val_watts) + 30
    else:
        val_dBm = ne.evaluate("10*log10(val_watts)+30")
    return val_dBm


def convert_dBm_to_watts(val_dBm: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert from dBm to Watts.

    Calculation: ``10^((val_dBm - 30) / 10)``

    NumPy is used for scalar inputs and small arrays.
    NumExpr is used to speed up the operation for large arrays.

    :param val_dBm: A value, or array of values, in dBm.
    :return: The input val_dBm, converted to Watts.
    """
    if np.isscalar(val_dBm) or val_dBm.size < NUMEXPR_THRESHOLD:
        val_watts = 10.0 ** ((val_dBm - 30) / 10)
    else:
        val_watts = ne.evaluate("10**((val_dBm-30)/10)")
    return val_watts


def convert_linear_to_dB(
    val_linear: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Convert from linear units to dB.

    Calculation: ``10 * log10(val_linear)``

    NumPy is used for scalar inputs and small arrays.
    NumExpr is used to speed up the operation for large arrays.

    :param val_linear: A value, or array of values, in linear
        units.
    :return: The input val_linear, converted to dB.
    """
    suppress_divide_by_zero_when_testing()
    if np.isscalar(val_linear) or val_linear.size < NUMEXPR_THRESHOLD:
        val_dB = 10.0 * np.log10(val_linear)
    else:
        val_dB = ne.evaluate("10*log10(val_linear)")
    return val_dB


def convert_dB_to_linear(val_dB: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert from dB to linear units.

    Calculation: ``10^(val_dB / 10)``

    NumPy is used for scalar inputs and small arrays.
    NumExpr is used to speed up the operation for large arrays.

    :param val_dB: A value, or array of values, in dB.
    :return: The input val_dB, converted to corresponding
        linear units.
    """
    if np.isscalar(val_dB) or val_dB.size < NUMEXPR_THRESHOLD:
        val_linear = 10.0 ** (val_dB / 10.0)
    else:
        val_linear = ne.evaluate("10**(val_dB/10)")
    return val_linear


def convert_kelvins_to_celsius(
    val_kelvins: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Convert from Kelvins to degrees Celsius.

    Calculation: ``val_kelvins - 273.15``

    :param val_kelvins: A value, or array of values, in
        Kelvins.
    :return: The input val_kelvins, converted to degrees
        Celsius.
    """
    return val_kelvins - 273.15


def convert_celsius_to_kelvins(
    val_celsius: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Convert from degrees Celsius to Kelvins.

    Calculation: ``val_celsius + 273.15``

    :param val_celsius: A value, or array of values, in
        degrees Celsius.
    :return: The input val_celsius, converted to Kelvins.
    """
    return val_celsius + 273.15


def convert_fahrenheit_to_celsius(
    val_fahrenheit: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Convert from degrees Fahrenheit to degrees Celsius.

    Calculation: ``(val_fahrenheit - 32) * (5 / 9)``

    :param val_fahrenheit: A value, or array of values, in
        degrees Fahrenheit.
    :return: The input val_fahrenheit, converted to degrees
        Celsius.
    """
    return (val_fahrenheit - 32.0) * (5.0 / 9.0)


def convert_celsius_to_fahrenheit(
    val_celsius: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Convert from degrees Celsius to degrees Fahrenheit.

    Calculations: ``(val_celsius * (9 / 5)) + 32``

    :param val_celsius: A value, or an array of values, in
        degrees Celsius.
    :return: The input val_celsius, converted to degrees Fahrenheit.
    """
    return (val_celsius * (9.0 / 5.0)) + 32.0
