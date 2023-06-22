from enum import Enum, EnumMeta

import numexpr as ne
import numpy as np

from scos_actions.signal_processing import NUMEXPR_THRESHOLD


def calculate_power_watts(val_volts, impedance_ohms: float = 50.0):
    """
    Calculate power in Watts from time domain samples in Volts.

    Calculation: (abs(val_volts)^2) / impedance_ohms

    NumPy is used for scalar inputs and small arrays.
    NumExpr is used to speed up the operation for large arrays.

    The calculation assumes 50 Ohm impedance by default.

    :param val_volts: A value, or array of values, in Volts.
        The input may be complex or real.
    :param impedance_ohms: The impedance value to use when
        converting from Volts to Watts.
    :return: The input val_volts, converted to Watts. The
        returned quantity is always real.
    """
    if np.isscalar(val_volts) or val_volts.size < NUMEXPR_THRESHOLD:
        power = (np.abs(val_volts) ** 2.0) / impedance_ohms
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

    NumPy is used for scalar inputs and small arrays.
    NumExpr is used to speed up the operation for large arrays.

    :param val: A value, or array of values, to be converted
        to 'pseudo-power' (magnitude-squared). The input may be
        complex or real.
    :return: The input val, converted to 'pseudo-power' (magnitude-
        squared). The returned quantity is always real.
    """
    if np.isscalar(val) or val.size < NUMEXPR_THRESHOLD:
        ps_pwr = np.abs(val) ** 2.0
    else:
        ps_pwr = ne.evaluate("abs(val).real**2")
    return ps_pwr


def create_statistical_detector(name: str, detectors: list) -> EnumMeta:
    """
    Construct a statistical detector based on a list of selected detectors.

    This allows for constructing new detectors while preserving the
    order of the 5 possible detector types in all instances. The five
    possible detector types to include are min, max, mean, median, and
    sample.

    The returned enumeration can be passed to ``apply_statistical_detector()``.

    :param name: The name of the returned detector enumeration.
    :param detectors: A list of strings specifying the detectors. Valid
        contents are: 'min', 'max', 'mean', 'median', and 'sample'.

    :return: The detector enumeration created based on the input parameters.
    :raises ValueError: If ``detectors`` contains unknown detector types.
    """
    # Construct 2-tuples to create enumeration
    _args = []
    # Check for invalid inputs
    allowed_detectors = ["min", "max", "mean", "median", "sample"]
    if not set(allowed_detectors) >= set(detectors):
        raise ValueError(f"Detectors must be one of {allowed_detectors}")
    # Use separate conditions to maintain ordering
    if "min" in detectors:
        _args.append(("min", "minimum"))
    if "max" in detectors:
        _args.append(("max", "maximum"))
    if "mean" in detectors:
        _args.append(("mean", "mean"))
    if "median" in detectors:
        _args.append(("median", "median"))
    if "sample" in detectors:
        _args.append(("sample", "sample"))
    return Enum(name, tuple(_args))


def apply_statistical_detector(
    data: np.ndarray,
    detector: EnumMeta,
    dtype: type = None,
    ignore_nan: bool = False,
    axis: int = 0,
) -> np.ndarray:
    """
    Apply statistical detectors to a 1- or 2-D array of samples.

    For 2-D input data, statistical detectors are applied along
    axis 0 (column-wise), and the sample detector selects a single
    row at random.

    For 1-D input data, statistical detectors are applied along the
    array, producing a single value output for each selected detector,
    and the sample detector selects a single value from the input at
    random.

    If the input samples are power FFT samples, stored in a 2-D array,
    they are expected to be packed in the shape (N_FFTs, N_Bins).

    The shape of the output depends on the number of detectors
    specified. The order of the results always follows min, max, mean,
    median, sample - regardless of which detectors are used. This
    ordering matches that of the detector enumerations.

    Create a detector using ``create_statistical_detector()``

    :param data: A 1- or 2-D array of real-valued samples in linear units.
    :param detector: A detector enumeration containing any combination
        of 'min', 'max', 'mean', 'median', and 'sample'. Also see the
        create_fft_detector and create_time_domain_detector documentation.
    :param dtype: Data type of values within the returned array. If not
        provided, the type is determined by NumPy as the minimum type
        required to hold the values (see numpy.array).
    :param ignore_nan: If true, statistical detectors (min/max/mean/median)
        will ignore any NaN values. NaN values may still appear in the
        random sample detector result.
    :param axis: Axis of ``data`` over which to apply detectors, defaults
        to 0.
    :return: A 1- or 2-D array containing the selected detector results
        as the specified dtype. For 1-D inputs, the 1-D output length is
        equal to the number of detectors applied. For 2-D inputs, the number
        of rows is equal to the number of detectors applied, and the number
        of columns is equal to the number of columns in the input array.
    :raises ValueError: If NaN values exist in the data and ``ignore_nan`` is
        False. In this case, all detectors will always return NaN if NaN is
        present in the input data, except for the sample detector. If this
        error is encountered, check that your input data is what you expect, and
        optionally enable ``ignore_nan``.
    """
    # Get detector names from detector enumeration
    detectors = [d.name for _, d in enumerate(detector)]

    if ignore_nan:
        detector_functions = [np.nanmin, np.nanmax, np.nanmean, np.nanmedian]
    else:
        if np.isnan(data).any():
            raise ValueError("Data contains NaN values but ``ignore_nan`` is False.")
        detector_functions = [np.min, np.max, np.mean, np.median]

    # Get functions based on specified detector
    applied_detectors = []
    if "min" in detectors:
        applied_detectors.append(detector_functions[0])
    if "max" in detectors:
        applied_detectors.append(detector_functions[1])
    if "mean" in detectors:
        applied_detectors.append(detector_functions[2])
    if "median" in detectors:
        applied_detectors.append(detector_functions[3])
    # Apply statistical detectors
    result = [d(data, axis=axis) for d in applied_detectors]
    # Add sample detector result if configured
    if "sample" in detectors:
        rng = np.random.default_rng()
        if axis == 0:
            # Pick out a random entire row
            sample_result = data[rng.integers(0, data.shape[0], 1)][0]
        elif axis == 1:
            # Pick out a random entire column
            sample_result = data.T[rng.integers(0, data.shape[0], 1)][0]
        else:
            raise NotImplementedError(
                "Sample detector not implemented for axes above 1"
            )
        result.append(sample_result)
        del rng
    result = np.array(result, dtype=dtype)
    return result


def filter_quantiles(x: np.ndarray, q_lo: float, q_hi: float) -> np.ndarray:
    """
    Replace values outside specified quantiles with NaN.

    :param x: Input N-dimensional data array. Complex valued arrays
        are not supported.
    :param q_lo: Lower quantile, 0 <= q_lo < q_hi.
    :param q_hi: Upper quantile, q_lo < q_hi <= 1.
    :return: The input data array, with values outside the
        specified quantile replaced with NaN (numpy.nan).
    :raises ValueError: If either ``q_lo`` or ``q_hi`` is not
        within its valid range (listed above).
    :raises TypeError: If ``x`` is not a real-valued NumPy array
        with a size greater than 1.
    """
    if q_lo < 0 or q_lo >= q_hi:
        raise ValueError("q_lo must satistfy 0 <= q_lo < q_hi")
    if q_hi > 1 or q_hi <= q_lo:
        raise ValueError("q_hi must satisfy q_lo < q_hi <= 1")
    if not isinstance(x, np.ndarray):
        raise TypeError("Input data must be a NumPy array")
    if x.size <= 1:
        raise TypeError("Input data must have length greater than 1")
    if np.iscomplexobj(x):
        raise TypeError("Input data must be real, not complex")
    lo, hi = np.quantile(x, [q_lo, q_hi])  # Works on flattened array
    if x.size < NUMEXPR_THRESHOLD:
        x = np.where((x <= lo) | (x > hi), np.nan, x)
    else:
        nan = np.nan
        x = ne.evaluate("x + where((x<=lo)|(x>hi), nan, 0)")
    return x
