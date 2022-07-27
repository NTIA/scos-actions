import logging
import os

import numexpr as ne
import numpy as np
from scipy.fft import fft as sp_fft
from scipy.signal import get_window

logger = logging.getLogger(__name__)


def get_fft(
    time_data: np.ndarray,
    fft_size: int,
    norm: str = "forward",
    fft_window: np.ndarray = None,
    num_ffts: int = 0,
    shift: bool = True,
    workers: int = os.cpu_count() // 2,
) -> np.ndarray:
    """
    Compute the 1-D DFT using the FFT algorithm.

    The input time domain samples are reshaped based on fft_size
    and num_ffts. Then, the window is applied. The FFT is performed
    and frequencies are shifted. The FFT is calculated using the
    scipy.fft.fft method, which is leveraged for parallelization.

    This function only scales the FFT output by 1/fft_size. No
    other power scaling, including RF/baseband conversion, is applied.
    It is recommended to first apply statistical detectors, if any,
    and apply power scaling as needed after converting values to dB,
    if applicable - this approach generally results in faster
    computation, and keeps power scaling details contained within
    individual actions.

    By default, as many FFTs as possible are computed, based on the
    length of the time_data input and the fft_size. If num_ffts is
    specified, the time domain data will be truncated to length
    (fft_size * num_ffts) before FFTs are computed. If num_ffts is not
    specified, but the length of the input time domain data is not
    evenly divisible by fft_size, the time domain data will still be
    truncated.

    :param time_data: An array of time domain samples, which can be
        complex.
    :param fft_size: Length of FFT (N_Bins).
    :param norm: Normalization mode. Valid options are 'backward',
        'ortho', and 'forward'. Backward applies no normalization,
        while 'forward' applies 1/fft_size scaling and 'ortho' applies
        1/sqrt(fft_size) scaling. Defaults to 'forward'.
    :param fft_window: An array of window samples (see get_fft_window).
        If not given, no windowing is performed (equivalent to rectangular
        windowing).
    :param num_ffts: The number of FFTs of length fft_size to compute.
        Setting this to zero or a negative number results in "as many
        as possible" behavior, which is also the default behavior if
        num_ffts is not specified.
    :param shift: If True, shift the zero-frequency component to the
        center of the spectrum.
    :param workers: Maximum number of workers to use for parallel
        computation. See scipy.fft.fft for more details.
    :return: The transformed input, scaled based on the specified
        normalization mode.
    """
    # Get num_ffts for default case: as many as possible
    if num_ffts <= 0:
        num_ffts = int(len(time_data) // fft_size)

    # Determine if truncation will occur and raise a warning if so
    if len(time_data) != fft_size * num_ffts:
        thrown_away_samples = len(time_data) - (fft_size * num_ffts)
        msg = "Time domain data length is not divisible by num_ffts.\nTime"
        msg += "domain data will be truncated; Throwing away last "
        msg += f"{thrown_away_samples} sample(s)."
        logger.warning(msg)

    # Resize time data for FFTs
    time_data = np.reshape(time_data[: num_ffts * fft_size], (num_ffts, fft_size))

    # Apply the FFT window if provided
    if fft_window is not None:
        time_data = ne.evaluate("time_data*fft_window")

    # Take the FFT
    complex_fft = sp_fft(time_data, norm=norm, workers=workers)

    # Shift the frequencies if desired (only along second axis)
    if shift:
        complex_fft = np.fft.fftshift(complex_fft, axes=(1,))
    return complex_fft


def get_fft_window(window_type: str, window_length: int) -> np.ndarray:
    """
    Generate a periodic window of the specified length.

    Supported values for window_type: boxcar, triang, blackman,
    hamming, hann (also "Hanning" supported for backwards compatibility),
    bartlett, flattop, parzen, bohman, blackmanharris, nuttall, barthann,
    cosine, exponential, tukey, and taylor.

    If an invalid window type is specified, a boxcar (rectangular)
    window will be used instead.

    :param window_type: A string supported by scipy.signal.get_window.
        Only windows which do not require additional parameters are
        supported. Whitespace and capitalization are ignored.
    :param window_length: The number of samples in the window.
    :return: An array of window samples, of length window_length and
        type window_type.
    """
    # String formatting for backwards-compatibility
    window_type = window_type.lower().strip().replace(" ", "")

    # Catch Hanning window for backwards-compatibility
    if window_type == "hanning":
        window_type = "hann"

    # Get window samples
    try:
        window = get_window(window_type, window_length)
    except ValueError:
        logger.debug(
            "Error generating FFT window. Attempting to"
            + " use a rectangular window..."
        )
        window = get_window("boxcar", window_length)

    # Return the window
    return window


def get_fft_window_correction(window: np.ndarray, correction_type: str) -> float:
    """
    Get the amplitude or energy correction factor for a window.

    :param window: The array of window samples.
    :param correction_type: Which correction factor to return.
        Must be one of 'amplitude' or 'energy'.
    :return: The specified window correction factor.
    :raises ValueError: If the correction type is neither 'energy'
        nor 'amplitude'.
    """
    if correction_type == "amplitude":
        window_correction = 1.0 / np.mean(window)
    elif correction_type == "energy":
        window_correction = np.sqrt(1.0 / np.mean(window**2))
    else:
        raise ValueError(f"Invalid window correction type: {correction_type}")

    return window_correction


def get_fft_frequencies(
    fft_size: int, sample_rate: float, center_frequency: float
) -> list:
    """
    Get the frequency axis for an FFT.

    The units of sample_rate and center_frequency should be the same:
    if both are given in Hz, the returned frequency values will be in
    Hz. If both are in MHz, the returned frequency values will be in
    MHz. It is recommended to keep them both in Hz.

    :param fft_size: The length, in samples, of the FFT (N_Bins).
    :param sample_rate: The sample rate for the transformed time domain
        samples, in Hz.
    :param center_frequency: The center frequency, in Hz.
    :return: A list of values representing the frequency axis of the
        FFT.
    """
    time_step = 1.0 / sample_rate
    frequencies = np.fft.fftfreq(fft_size, time_step)
    frequencies = np.fft.fftshift(frequencies) + center_frequency
    return frequencies.tolist()


def get_fft_enbw(fft_window: np.ndarray, sample_rate: float) -> float:
    """
    Get the equivalent noise bandwidth of an FFT bin.

    The FFT size is inferred from the number of samples
    in the input window.

    :param fft_window: An array of window samples.
    :param sample_rate: The sampling rate, in Hz.
    """
    # window_enbw is (amplitude_correction/energy_correction)^2
    # Here, get_fft_window_correction is not used in order to
    # simplify the calculation.
    window_enbw = np.mean(fft_window**2) / (np.mean(fft_window) ** 2)
    return (sample_rate / len(fft_window)) * window_enbw
