import logging
import numexpr as ne
import numpy as np
from enum import Enum, EnumMeta
from numpy import ndarray

logger = logging.getLogger(__name__)


def calculate_power_watts(val_volts, impedance_ohms: float = 50.):
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
    :returns: The input val_volts, converted to Watts. The
        returned quantity is always real.
    """
    if np.isscalar(val_volts):
        power = (np.abs(val_volts) ** 2) / impedance_ohms
    else:
        power = ne.evaluate('(abs(val_volts)**2)/impedance_ohms')
    if np.iscomplexobj(power):
        # NumExpr returns complex type for complex input
        power = np.real(power)
    return power


def create_time_domain_detector(name: str, detectors: list) -> EnumMeta:
    """
    Construct a time domain detector based on a list of selected detectors.

    This allows for constructing new time domain detectors while preserving
    the order of the 5 possible detector types in all instances. The five
    possible detector types to include are min, max, mean, median, and
    sample.

    The returned enumeration can be passed to apply_detector.

    :param name: The name of the returned detector enumeration.
    :param detectors: A list of strings specifying the detectors. Valid
        contents are: 'min', 'max', 'mean', 'median', and 'sample'.

    :returns: The detector enumeration created based on the input parameters.
    """
    # Construct 2-tuples to create enumeration
    _args = []
    if 'min' in detectors:
        _args.append(('min', 'time_domain_min_power'))
    if 'max' in detectors:
        _args.append(('max', 'time_domain_max_power'))
    if 'mean' in detectors:
        _args.append(('mean', 'time_domain_mean_power'))
    if 'median' in detectors:
        _args.append(('median', 'time_domain_median_power'))
    if 'sample' in detectors:
        _args.append(('sample', 'time_domain_sample_power'))
    return Enum(name, tuple(_args))


def apply_power_detector(data: ndarray, detector: EnumMeta,
                         dtype: type = None) -> ndarray:
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

    Create a detector using fft.create_fft_detector or
    create_time_domain_detector.

    :param data: A 2-D array of real, linear samples.
    :param detector: A detector enumeration containing any combination
        of 'min', 'max', 'mean', 'median', and 'sample'. Also see the
        create_fft_detector and create_time_domain_detector documentation.
    :param dtype: Data type of values within the returned array. If not
        provided, the type is determined by NumPy as the minimum type
        required to hold the values (see numpy.array).
    :returns: A 2-D array containing the selected detector results
        as the specified dtype. The number of rows is equal to the
        number of detectors applied, and the number of columns is equal
        to the number of columns in the input array.
    """
    # Currently this is identical to apply_fft_detector: make general?
    # Get detector names from detector enumeration
    detectors = [d.name for _, d in enumerate(detector)]
    # Get functions based on specified detector
    detector_functions = []
    if 'min' in detectors:
        detector_functions.append(np.min)
    if 'max' in detectors:
        detector_functions.append(np.max)
    if 'mean' in detectors:
        detector_functions.append(np.mean)
    if 'median' in detectors:
        detector_functions.append(np.median)
    # Apply statistical detectors
    result = [d(data, axis=0) for d in detector_functions]
    # Add sample detector result if configured
    if 'sample' in detectors:
        rng = np.random.default_rng()
        result.append(data[rng.integers(0, data.shape[0], 1)][0])
        del rng
    return np.array(result, dtype=dtype)


def filter_quantiles(x: ndarray, q_lo: float, q_hi: float) -> ndarray:
    """
    Replace values outside specified quantiles with NaN.

    :param x: Input N-dimensional data array.
    :param q_lo: Lower quantile, 0 <= q_lo < q_hi.
    :param q_hi: Upper quantile, q_lo < q_hi <= 1.
    :returns: The input data array, with values outside the
        specified quantile replaced with NaN (numpy.nan).
    """
    lo, hi = np.quantile(x, [q_lo, q_hi])  # Works on flattened array
    nan = np.nan
    return ne.evaluate('x + where((x<=lo)|(x>hi), nan, 0)')


def data_to_blocks(data: ndarray, num_blocks: int = None,
                   block_size: int = None) -> ndarray:
    """
    Convert 1-D array of samples in to 2-D array of sequential blocks.

    The returned array will have shape (num_blocks, block_size).

    If only one of num_blocks or block_size are specified, the other
    is calculated based on the length of the data.

    If data truncation occurs by either specified parameters or
    when specifying only one of the shape parameters, a warning
    is raised, but the data is truncated to the specified length
    before being reshaped.

    If there are not enough samples in data to satisfy the input
    values of num_blocks and block_size, an

    :param data: The input 1-D array of samples.
    :param num_blocks:
    :param block_size: The length, in samples, of each block.
    :returns:
    :raises SomeException
    :raises SomeException: When there are not enough samples in
        data to satisfy the values of num_blocks and block_size.
    """
    if block_size is None and num_blocks is None:
        # If neither parameter is specified
        raise Exception
    if block_size is None and num_blocks is not None:
        # If only num_blocks is specified
        block_size = len(data) // num_blocks
    elif num_blocks is None and block_size is not None:
        # If only block size is specified
        num_blocks = len(data) // block_size
    elif block_size * num_blocks > len(data):
        # If both parameters are specified, and there are not
        # enough samples in data to satisfy parameters.
        raise Exception
    # Check for truncation and warn if truncating
    if block_size * num_blocks < len(data):
        # Raise a warning but continue with truncation
        lost_samples = len(data) - (block_size * num_blocks)
        msg = "Specified block_size and num_blocks passed to data_to_blocks" \
              + f"will result in data truncation. {lost_samples} will be " \
              + f"thrown away, out of {len(data)} original samples."
        logger.warning(msg)
        data = data[:num_blocks * block_size]
    # Return reshaped data
    return np.reshape(data, (num_blocks, block_size))
