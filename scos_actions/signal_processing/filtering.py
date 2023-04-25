from typing import Tuple, Union

import numpy as np
from scipy.signal import ellip, ellipord, firwin, kaiserord, sos2zpk, sosfreqz


def generate_elliptic_iir_low_pass_filter(
    gpass_dB: float,
    gstop_dB: float,
    pb_edge_Hz: float,
    sb_edge_Hz: float,
    sample_rate_Hz: float,
) -> np.ndarray:
    """
    Generate an elliptic IIR low pass filter.

    This method generates a second-order sections representation of
    the lowest order digital elliptic filter which loses no more than
    ``gpass_dB`` dB in the passband and has at least ``gstop_dB``
    attenuation in the stopband. The passband and stopband are defined
    by their edge frequencies, ``pb_edge_Hz`` and ``sb_edge_Hz``.

    Apply this filter to data using scipy.signal.sosfilt or
    scipy.signal.sosfiltfilt (for forwards-backwards filtering).

    :param gpass_dB: Maximum passband ripple below unity gain, in dB.
    :param gstop_dB: Minimum stopband attenuation, in dB.
    :param pb_edge_Hz: Filter passband edge frequency, in Hz.
    :param sb_edge_Hz: Filter stopband edge frequency, in Hz.
    :param sample_rate_Hz: Sampling rate, in Hz.
    :return: Second-order sections representation of the IIR filter.
    """
    if sb_edge_Hz <= pb_edge_Hz:
        raise ValueError(
            f"Stopband edge frequency {sb_edge_Hz} Hz is not greater than passband"
            + f"edge frequency {pb_edge_Hz} Hz."
        )
    ord, wn = ellipord(
        pb_edge_Hz, sb_edge_Hz, gpass_dB, gstop_dB, False, sample_rate_Hz
    )
    sos = ellip(ord, gpass_dB, gstop_dB, wn, "lowpass", False, "sos", sample_rate_Hz)
    return sos


def generate_fir_low_pass_filter(
    attenuation_dB: float, width_Hz: float, cutoff_Hz: float, sample_rate_Hz: float
) -> np.ndarray:
    """
    Generate a FIR low pass filter using the Kaiser window method.

    This method computes the coefficients of a finite impulse
    response filter, with linear phase,

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
    taps = firwin(
        ord + 1,
        cutoff_Hz,
        width_Hz,
        ("kaiser", beta),
        "lowpass",
        True,
        fs=sample_rate_Hz,
    )
    return taps


def get_iir_frequency_response(
    sos: np.ndarray, worN: Union[int, np.ndarray], sample_rate_Hz: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the frequency response of an IIR filter.

    :param sos: Second-order sections representation of the IIR filter.
    :param worN: If a single integer, then compute at that many frequencies.
        If an array is supplied, it should be the frequencies at which to
        compute the frequency response (in Hz).
    :param sample_rate_Hz: Sampling rate, in Hz.
    :return: A tuple containing two NumPy arrays. The first is the array of
        frequencies, in Hz, for which the frequency response was calculated.
        The second is the array containing the frequency response values, which
        are complex values in linear units.
    """
    w, h = sosfreqz(sos, worN, whole=True, fs=sample_rate_Hz)
    return w, h


def get_iir_phase_response(
    sos: np.ndarray, worN: Union[int, np.ndarray], sample_rate_Hz: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the phase response of an IIR filter.

    :param sos: Second-order sections representation of the IIR filter.
    :param worN: If a single integer, then compute at that many frequencies.
        If an array is supplied, it should be the frequencies at which to
        compute the phase response (in Hz).
    :param sample_rate_Hz: Sampling rate, in Hz.
    :return: A tuple containing two NumPy arrays. The first is the array of
        frequencies, in Hz, for which the phase response was calculated.
        The second is the array containing the phase response values, in radians.
    """
    w, h = sosfreqz(sos, worN, whole=False, fs=sample_rate_Hz)
    angles = np.unwrap(np.angle(h))
    return w, angles


def get_iir_enbw(
    sos: np.ndarray, worN: Union[int, np.ndarray], sample_rate_Hz: float
) -> float:
    """
    Get the equivalent noise bandwidth of an IIR filter.

    :param sos: Second-order sections representation of the IIR filter.
    :param worN: If a single integer, then compute at that many frequencies.
        If an array is supplied, it should be the frequencies at which to
        compute the frequency response (in Hz) to estimate the ENBW. The frequencies
        should span from ``-sample_rate_Hz / 2`` to ``+sample_rate_Hz / 2``.
    :param sample_rate_Hz: Sampling rate, in Hz.
    :return: The equivalent noise bandwidth of the input filter, in Hz.
    """
    if isinstance(worN, float) and worN.is_integer():
        worN = int(worN)
    if isinstance(worN, int):
        worN = np.linspace(-sample_rate_Hz / 2, sample_rate_Hz / 2, num=worN)
    if not isinstance(worN, np.ndarray):
        raise TypeError(f"Parameter worN must be int or np.ndarray, not {type(worN)}.")
    if min(worN) < -sample_rate_Hz / 2 or max(worN) > sample_rate_Hz / 2:
        raise ValueError(
            "Supplied frequency values must fall within +/- Nyquist frequency at baseband."
        )
    w, h = get_iir_frequency_response(sos, worN, sample_rate_Hz)
    dw = np.mean(np.diff(w))
    h = np.abs(h) ** 2.0
    enbw = np.sum(h) * (dw / h.max())
    return enbw


def is_stable(sos: np.ndarray) -> bool:
    """
    Check IIR filter stability using Z-plane analysis.

    An IIR filter is stable if its poles lie within the
    unit circle on the Z-plane.

    :param sos: Second-order sections representation of the IIR filter.
    :return: True if the filter is stable, False if not.
    """
    _, poles, _ = sos2zpk(sos)
    stable = all([True if p < 1 else False for p in np.square(np.abs(poles))])
    return stable
