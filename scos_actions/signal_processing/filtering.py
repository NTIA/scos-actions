import logging
import numpy as np
from scipy.signal import ellip, ellipord

logger = logging.getLogger(__name__)

def generate_elliptic_iir_low_pass_filter(
        rp_dB: float,
        rs_dB: float,
        cutoff_Hz: float,
        width_Hz: float,
        sample_rate_Hz: float
) -> np.ndarray:
    """
    Generate an elliptic IIR low pass filter.

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