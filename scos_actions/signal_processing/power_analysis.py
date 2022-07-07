import numexpr as ne
import os
import warnings


def convert_volts_to_watts(val_volts, impedance_ohms: float = 50.):
    """
    Convert a value, or array of values, from Volts to Watts.

    Calculation: (abs(val_volts)^2) / impedance_ohms

    NumExpr is used to speed up the operation for large arrays.
    Calculation assumes 50 Ohm impedance by default.

    :param val_volts: A value, or array of values, in Volts.
    :param impedance_ohms: The impedance value to use when
        converting from Volts to Watts.
    :returns: The input val_volts, converted to Watts.
    """
    return ne.evaluate('(abs(val_volts)**2)/impedance_ohms')


def convert_watts_to_dBm(val_watts):
    """
    Convert a value, or array of values, from Watts to dBm.

    Calculation: 10 * log10(val_watts) + 30

    NumExpr is used to speed up the operation for large arrays.

    :param val_watts: A value, or array of values, in Watts.
    :returns: The input val_watts, converted to dBm.
    """
    # If testing, don't output divide-by-zero warnings from log10
    if 'PYTEST_CURRENT_TEST' in os.environ:
        warnings.filterwarnings('ignore', message='divide by zero')
    return ne.evaluate('10*log10(val_watts)+30')


def convert_dBm_to_watts(val_dBm):
    """
    Convert a value, or array of values, from dBm to Watts.

    Calculation: 10^((val_dBm - 30) / 10)

    NumExpr is used to speed up the operation for large arrays.

    :param val_dBm: A value, or array of values, in dBm.
    :returns: The input val_dBm, converted to Watts.
    """
    return ne.evaluate('10**((val_dBm-30)/10)')
