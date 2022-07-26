import logging
from enum import Enum, EnumMeta

import numexpr as ne
import numpy as np

logger = logging.getLogger(__name__)


def calculate_power_watts(val_volts, impedance_ohms: float = 50.0):
    """
    Calculate power in Watts from time domain samples in Volts.

    Calculation: (abs(val_volts)^2) / impedance_ohms

    NumPy is used for scalar inputs.
    NumExpr is used to speed up the operation for arrays.

    The calculation assumes 50 Ohm impedance by default.

    :param val_volts: A value, or array of values, in Volts.
        The input may be complex or real.
    :param impedance_ohms: The impedance value to use when
        converting from Volts to Watts.
    :return: The input val_volts, converted to Watts. The
        returned quantity is always real.
    """
    if np.isscalar(val_volts):
        power = (np.abs(val_volts) ** 2.) / impedance_ohms
    else:
        power = ne.evaluate("(abs(val_volts).real**2)/impedance_ohms")
    return power


def calculate_pseudo_power(val):
    """
    Calculate the 'pseudo-power' (magnitude-squared) of input samples.

    Calculation: abs(val)^2

    'Pseudo-power' is useful in certain applications to avoid
    computing the power of many samples before data reduction
    by a detector.

    NumPy is used for scalar inputs.
    NumExpr is used to speed up the operation for arrays.

    :param val: A value, or array of values, to be converted
        to 'pseudo-power' (magnitude-squared). The input may be
        complex or real.
    :return: The input val, converted to 'pseudo-power' (magnitude-
        squared). The returned quantity is always real.
    """
    if np.isscalar(val):
        ps_pwr = np.abs(val) ** 2.
    else:
        ps_pwr = ne.evaluate("abs(val).real**2")
    return ps_pwr


def create_power_detector(name: str, detectors: list) -> EnumMeta:
    """
    Construct a power detector based on a list of selected detectors.

    This allows for constructing new detectors while preserving the
    order of the 5 possible detector types in all instances. The five
    possible detector types to include are min, max, mean, median, and
    sample.

    The returned enumeration can be passed to ``apply_power_detector()``.

    :param name: The name of the returned detector enumeration.
    :param detectors: A list of strings specifying the detectors. Valid
        contents are: 'min', 'max', 'mean', 'median', and 'sample'.

    :return: The detector enumeration created based on the input parameters.
    """
    # Construct 2-tuples to create enumeration
    _args = []
    if "min" in detectors:
        _args.append(("min", "min_power"))
    if "max" in detectors:
        _args.append(("max", "max_power"))
    if "mean" in detectors:
        _args.append(("mean", "mean_power"))
    if "median" in detectors:
        _args.append(("median", "median_power"))
    if "sample" in detectors:
        _args.append(("sample", "sample_power"))
    return Enum(name, tuple(_args))


def apply_power_detector(
    data: np.ndarray, detector: EnumMeta, dtype: type = None, ignore_nan: bool = False,
) -> np.ndarray:
    """
    Apply statistical detectors to a 2-D array of samples.

    Statistical detectors are applied along axis 0 (column-wise),
    and the sample detector selects a single row from the 2-D
    array at random.

    If the input samples are power FFT samples, they are expected
    to be packed in the shape (N_FFTs, N_Bins).

    The shape of the output depends on the number of detectors
    specified. The order of the results always follows min, max, mean,
    median, sample - regardless of which detectors are used. This
    ordering matches that of the detector enumerations.

    Create a detector using ``create_power_detector()``

    :param data: A 2-D array of real, linear samples.
    :param detector: A detector enumeration containing any combination
        of 'min', 'max', 'mean', 'median', and 'sample'. Also see the
        create_fft_detector and create_time_domain_detector documentation.
    :param dtype: Data type of values within the returned array. If not
        provided, the type is determined by NumPy as the minimum type
        required to hold the values (see numpy.array).
    :param ignore_nan: If true, statistical detectors (min/max/mean/median)
        will ignore any NaN values. NaN values may still appear in the
        random sample detector result.
    :return: A 2-D array containing the selected detector results
        as the specified dtype. The number of rows is equal to the
        number of detectors applied, and the number of columns is equal
        to the number of columns in the input array.
    """
    # Currently this is identical to apply_fft_detector: make general?
    # Get detector names from detector enumeration
    detectors = [d.name for _, d in enumerate(detector)]

    if ignore_nan:
        detector_functions = [np.nanmin, np.nanmax, np.nanmean, np.nanmedian]
    else:
        detector_functions = [np.min, np.max, np.mean, np.median]
        
    # Get functions based on specified detector
    if "min" in detectors:
        detector_functions.append(detector_functions[0])
    if "max" in detectors:
        detector_functions.append(detector_functions[1])
    if "mean" in detectors:
        detector_functions.append(detector_functions[2])
    if "median" in detectors:
        detector_functions.append(detector_functions[3])
    # Apply statistical detectors
    result = [d(data, axis=0) for d in detector_functions]
    # Add sample detector result if configured
    if "sample" in detectors:
        rng = np.random.default_rng()
        result.append(data[rng.integers(0, data.shape[0], 1)][0])
        del rng
    return np.array(result, dtype=dtype)


def filter_quantiles(x: np.ndarray, q_lo: float, q_hi: float) -> np.ndarray:
    """
    Replace values outside specified quantiles with NaN.

    :param x: Input N-dimensional data array.
    :param q_lo: Lower quantile, 0 <= q_lo < q_hi.
    :param q_hi: Upper quantile, q_lo < q_hi <= 1.
    :return: The input data array, with values outside the
        specified quantile replaced with NaN (numpy.nan).
    """
    lo, hi = np.quantile(x, [q_lo, q_hi])  # Works on flattened array
    nan = np.nan
    return ne.evaluate("x + where((x<=lo)|(x>hi), nan, 0)")
