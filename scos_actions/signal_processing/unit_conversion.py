import os
import warnings
import numexpr as ne
import numpy as np
from numpy import ndarray
from typing import Union


def suppress_divide_by_zero_when_testing():
    # If testing, don't output divide-by-zero warnings from log10
    # This handles NumPy and NumExpr warnings
    if 'PYTEST_CURRENT_TEST' in os.environ:
        warnings.filterwarnings('ignore', message='divide by zero')
        np_error_settings_savepoint = np.seterr(divide='ignore')


def convert_watts_to_dBm(val_watts: Union[float, ndarray]) -> Union[float,
                                                                    ndarray]:
    """
    Convert from Watts to dBm.

    Calculation: 10 * log10(val_watts) + 30

    NumPy is used for scalar inputs.
    NumExpr is used to speed up the operation for arrays.

    :param val_watts: A value, or array of values, in Watts.
    :returns: The input val_watts, converted to dBm.
    """
    suppress_divide_by_zero_when_testing()
    if np.isscalar(val_watts):
        val_dBm = 10. * np.log10(val_watts) + 30
    else:
        val_dBm = ne.evaluate('10*log10(val_watts)+30')
    return val_dBm


def convert_dBm_to_watts(val_dBm: Union[float, ndarray]) -> Union[float,
                                                                  ndarray]:
    """
    Convert from dBm to Watts.

    Calculation: 10^((val_dBm - 30) / 10)

    NumPy is used for scalar inputs.
    NumExpr is used to speed up the operation for arrays.

    :param val_dBm: A value, or array of values, in dBm.
    :returns: The input val_dBm, converted to Watts.
    """
    if np.isscalar(val_dBm):
        val_watts = 10. ** ((val_dBm - 30) / 10)
    else:
        val_watts = ne.evaluate('10**((val_dBm-30)/10)')
    return val_watts


def convert_linear_to_dB(val_linear: Union[float, ndarray]) -> Union[float,
                                                                     ndarray]:
    """
    Convert from linear units to dB.

    Calculation: 10 * log10(val_linear)

    NumPy is used for scalar inputs.
    NumExpr is used to speed up the operation for arrays.

    :param val_linear: A value, or array of values, in linear
        units.
    :returns: The input val_linear, convert to dB.
    """
    suppress_divide_by_zero_when_testing()
    if np.isscalar(val_linear):
        val_dB = 10. * np.log10(val_linear)
    else:
        val_dB = ne.evaluate('10*log10(val_linear)')
    return val_dB


def convert_dB_to_linear(val_dB: Union[float, ndarray]) -> Union[float,
                                                                 ndarray]:
    """
    Convert from dB to linear units.

    Calculation: 10^(val_dB / 10)

    NumPy is used for scalar inputs.
    NumExpr is used to speed up the operation for arrays.

    :param val_dB: A value, or array of values, in dB.
    :returns: The input val_dB, in corresponding linear units.
    """
    if np.isscalar(val_dB):
        val_linear = 10. ** (val_dB / 10.)
    else:
        val_linear = ne.evaluate('10**(val_dB/10)')
    return val_linear
