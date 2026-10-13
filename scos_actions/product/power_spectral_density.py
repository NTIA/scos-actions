
"""Power Spectral Density"""

import logging
from enum import EnumMeta

import numpy as np
import ray

from scos_actions.signal_processing.fft import (
    get_fft, get_fft_window, get_fft_window_correction,
)
from scos_actions.signal_processing.power_analysis import (
    apply_statistical_detector,
    calculate_pseudo_power,
)

logger = logging.getLogger(__name__)


@ray.remote
class PowerSpectralDensity:
    """
    Compute data product mean/max PSD results from IQ samples.

    The result is a PSD, with amplitudes in dBm/Hz referenced to the
    calibration terminal (assuming IQ data has been scaled for calibration).
    The result is truncated in frequency to the middle 10 MHz (middle 625 out
    of 875 total DFT samples).
    """

    def __init__(
        self,
        sample_rate_Hz: float,
        num_ffts: int,
        fft_size: int,
        fft_window_type: str,
        fft_window_correction_type: str,
        detector: EnumMeta,
        percentiles: np.ndarray,
        impedance_ohms: float,
    ):
        self.detector = detector
        self.percentiles = percentiles
        self.fft_size = fft_size
        self.fft_window = get_fft_window(fft_window_type, fft_size)
        fft_window_correction = get_fft_window_correction(self.fft_window, fft_window_correction_type)
        self.num_ffts = num_ffts
        # Get truncation points: truncate FFT result to middle 125 samples (middle 10 MHz from 14 MHz)
        self.bin_start = int(fft_size / 7)  # bin_start = 25 with FFT_SIZE 175
        self.bin_end = fft_size - self.bin_start  # bin_end = 150 with FFT_SIZE 175
        # Compute the amplitude shift for PSD scaling. The FFT result
        # is in pseudo-power log units and must be scaled to a PSD.
        self.fft_scale_factor = (
            -10.0 * np.log10(impedance_ohms)  # Pseudo-power to power
            + 27.0  # Watts to dBm (+30) and baseband to RF (-3)
            - 10.0 * np.log10(sample_rate_Hz * fft_size)  # PSD scaling
            + 20.0 * np.log10(fft_window_correction)  # Window energy correction
        )

    def run(self, iq: ray.ObjectRef) -> np.ndarray:
        """
        Compute power spectral densities from IQ samples.

        :param iq: Complex-valued input waveform samples (which should
            already exist in the Ray object store).
        :return: A 2D NumPy array of statistical detector results computed
            from PSD amplitudes, ordered (max, mean).
        """
        fft_amplitudes = get_fft(
            iq, self.fft_size, "backward", self.fft_window, self.num_ffts, False, 1
        )
        # Power in Watts
        fft_amplitudes = calculate_pseudo_power(fft_amplitudes)
        fft_result = apply_statistical_detector(
            fft_amplitudes, self.detector
        )  # (max, mean, median)
        percentile_result = np.percentile(fft_amplitudes, self.percentiles, axis=0)
        fft_result = np.vstack((fft_result, percentile_result))
        fft_result = np.fft.fftshift(fft_result, axes=(1,))  # Shift frequencies
        fft_result = fft_result[
            :, self.bin_start : self.bin_end
        ]  # Truncation to middle bins
        fft_result = 10.0 * np.log10(fft_result) + self.fft_scale_factor

        # Returned order is (max, mean, median, 25%, 75%, 90%, 95%, 99%, 99.9%, 99.99%)
        # Total of 10 arrays, each of length 125 (output shape (10, 125))
        # Percentile computation linearly interpolates. See numpy documentation.
        return fft_result