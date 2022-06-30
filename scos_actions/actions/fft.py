import os
from enum import Enum
import logging
import numpy as np
from numpy.typing import NDArray
from scipy.signal import get_window

logger = logging.getLogger(__name__)


class M4sDetector(Enum):
    min = "fft_min_power"
    max = "fft_max_power"
    mean = "fft_mean_power"
    median = "fft_median_power"
    sample = "fft_sample_power"


def m4s_detector(array):
    """Take min, max, mean, median, and random sample of n-dimensional array.

    Detector is applied along each column.

    :param array: an (m x n) array of real frequency-domain linear power values
    :returns: a (5 x n) in the order min, max, mean, median, sample in the case
              that `detector` is `m4s`, otherwise a (1 x n) array

    """
    amin = np.min(array, axis=0)
    amax = np.max(array, axis=0)
    mean = np.mean(array, axis=0)
    median = np.median(array, axis=0)
    random_sample = array[np.random.randint(0, array.shape[0], 1)][0]
    m4s = np.array([amin, amax, mean, median, random_sample], dtype=np.float32)

    return m4s


def mean_detector(array):
    mean = np.mean(array, axis=0)
    return mean


def get_frequency_domain_data(time_data, sample_rate, fft_size, fft_window, fft_window_acf):
    logger.debug(
        'Converting {} samples at {} to freq domain with fft_size {}'.format(len(time_data), sample_rate, fft_size))
    # Resize time data for FFTs
    num_ffts = int(len(time_data) / fft_size)
    time_data = np.resize(time_data, (num_ffts, fft_size))
    # Apply the FFT window
    data = time_data * fft_window
    # Take and shift the fft (center frequency)
    complex_fft = np.fft.fft(data)
    complex_fft = np.fft.fftshift(complex_fft)
    complex_fft /= 2
    # Convert from V/Hz to V
    complex_fft /= fft_size
    # Apply the window's amplitude correction factor
    complex_fft *= fft_window_acf
    return complex_fft


def convert_volts_to_watts(complex_fft):
    # Convert to power P=V^2/R
    power_fft = np.abs(complex_fft)
    power_fft = np.square(power_fft)
    power_fft /= 50
    return power_fft


def apply_detector(power_fft):
    # Run the M4S detector
    power_fft_m4s = m4s_detector(power_fft)
    return power_fft_m4s


def convert_watts_to_dbm(power_fft):
    # If testing, don't flood output with divide-by-zero warnings from np.log10
    # if settings.RUNNING_TESTS:
    if "PYTEST_CURRENT_TEST" in os.environ:
        np_error_settings_savepoint = np.seterr(divide="ignore")
    # Convert to dBm dBm = dB +30; dB = 10log(W)
    power_fft_dbm = 10 * np.log10(power_fft) + 30
    return power_fft_dbm


def get_fft_window(window_type: str, window_length: int) -> np.ndarray:
    """
    Generate a periodic window of the specified length.

    Supported values for window_type: boxcar, triang, blackman, hamming,
    hann (also "Hanning" supported for backwards compatibility),
    bartlett, flattop, parzen, bohman, blackmanharris, nuttall, barthann,
    cosine, exponential, tukey, and taylor.

    If an invalid window type is specified, a boxcar (rectangular)
    window will be used instead.

    :param window_type: A string supported by scipy.signal.get_window.
        Only windows which do not require additional parameters are
        supported. Whitespace and capitalization are ignored.
    :param window_length: The number of samples in the window.
    :returns: An array of window samples, of length window_length and
        type window_type.
    """
    # String formatting for backwards-compatibility
    window_type = window_type.lower().strip().replace(' ', '')

    # Catch Hanning window for backwards-compatibility
    if window_type == 'hanning':
        window_type = 'hann'

    # Get window samples
    try:
        window = get_window(window_type, window_length)
    except ValueError:
        logger.debug('Error generating FFT window. Attempting to'
                     + ' use a rectangular window...')
        window = get_window('boxcar', window_length)

    # Return the window
    return window


def get_fft_window_correction(window: NDArray,
                              correction_type: str = "amplitude") -> float:
    """
    Get the amplitude or energy correction factor for a window.

    :param window: The array of window samples.
    :param correction_type: Which correction factor to return.
        Must be one of 'amplitude' or 'energy'.

    :returns: The specified window correction factor.
    :raises ValueError: If the correction type is neither 'energy'
        nor 'amplitude'.
    """
    if correction_type == 'amplitude':
        window_correction = 1. / np.mean(window)
    elif correction_type == 'energy':
        window_correction = np.sqrt(1. / np.mean(window ** 2))
    else:
        raise ValueError(f"Invalid window correction type: {correction_type}")

    return window_correction

def get_fft_frequencies(fft_size, sample_rate, center_frequency):
    time_step = 1 / sample_rate
    frequencies = np.fft.fftfreq(fft_size, time_step)
    frequencies = np.fft.fftshift(frequencies) + center_frequency
    return frequencies


def get_m4s_watts(fft_size, measurement_result, fft_window, fft_window_acf):
    complex_fft = get_frequency_domain_data(
        measurement_result["data"], measurement_result["sample_rate"], fft_size, fft_window, fft_window_acf
    )
    power_fft = convert_volts_to_watts(complex_fft)
    power_fft_m4s = apply_detector(power_fft)
    return power_fft_m4s


def get_m4s_dbm(fft_size, measurement_result, fft_window, fft_window_acf):
    m4_watts = get_m4s_watts(fft_size, measurement_result, fft_window, fft_window_acf)
    power_fft_dbm = convert_watts_to_dbm(m4_watts)
    return power_fft_dbm


def get_mean_detector_watts(fft_size, measurement_result, fft_window, fft_window_acf):
    complex_fft = get_frequency_domain_data(
        measurement_result["data"], measurement_result["sample_rate"], fft_size, fft_window, fft_window_acf
    )
    power_fft = convert_volts_to_watts(complex_fft)
    return mean_detector(power_fft)


def get_enbw(sample_rate, fft_size, fft_window_enbw):
    enbw = sample_rate
    enbw *= fft_window_enbw
    enbw /= fft_size
    return enbw

def get_fft_window_correction_factors(fft_window):
    fft_window_acf = get_fft_window_correction(fft_window, "amplitude")
    fft_window_ecf = get_fft_window_correction(fft_window, "energy")
    fft_window_enbw = (fft_window_acf / fft_window_ecf) ** 2
    return fft_window_acf, fft_window_ecf, fft_window_enbw
