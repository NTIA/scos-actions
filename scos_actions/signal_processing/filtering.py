import logging
import numpy as np
from scipy.signal import ellip, ellipord, kaiserord, firwin

logger = logging.getLogger(__name__)

def generate_elliptic_iir_low_pass_filter(
        rp_dB: float,
        rs_dB: float,
        cutoff_Hz: float,
        width_Hz: float,
        sample_rate_Hz: float,
) -> np.ndarray:
    """
    Generate an elliptic IIR low pass filter.

    Apply this filter to data using scipy.signal.sosfilt or
    scipy.signal.sosfiltfilt (for forwards-backwards filtering).

    :param rp_dB: Maximum passband ripple below unity gain, in dB.
    :param rs_dB: Minimum stopband attenuation, in dB.
    :param cutoff_Hz: Filter cutoff frequency, in Hz.
    :param width_Hz: Passband-to-stopband transition width, in Hz.
    :param sample_rate_Hz: Sampling rate, in Hz.
    :return: Second-order sections representation of the IIR filter.
    """
    ord, wn = ellipord(cutoff_Hz, cutoff_Hz + width_Hz, rp_dB, rs_dB, False, sample_rate_Hz)
    sos = ellip(ord, rp_dB, rs_dB, wn, 'lowpass', False, 'sos', sample_rate_Hz)
    logger.debug(f'Generated low-pass IIR filter with order {ord}.')
    return sos


def generate_fir_low_pass_filter(
        attenuation_dB: float,
        width_Hz: float,
        cutoff_Hz: float,
        sample_rate_Hz: float
) -> np.ndarray:
    """
    Generate a FIR low pass filter using the Kaiser window method.

    Apply this filter to data using scipy.signal.lfilter or
    scipy.signal.filtfilt (for forwards-backwards filtering).
    In either case, use the coefficients output by this method as
    the numerator parameter, with a denominator of 1.0.

    :param atten_dB: Minimum stopband attenuation, in dB.
    :param width_Hz: Width of the transition region, in Hz.
    :param cutoff_Hz: Filter cutoff frequency, in Hz.
    :param sample_rate_Hz: Sampling rate, in Hz.
    :return: Coeffiecients of the FIR low pass filter.
    """
    ord, beta = kaiserord(attenuation_dB, width_Hz / (0.5 * sample_rate_Hz))
    taps = firwin(ord + 1, cutoff_Hz, width_Hz, ('kaiser', beta), 'lowpass', True, fs=sample_rate_Hz)
    logger.debug(f"Generated Type {'I' if ord % 2 == 0 else 'II'} low-pass FIR filter with order {ord} and length {ord + 1}.")
    return taps 