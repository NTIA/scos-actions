"""Power Spectral Density"""

import logging

import numpy as np
import ray

from scos_actions.product.interfaces import ProductInterface
from scos_actions.signal_processing.fft import (
    get_fft,
    get_fft_window,
    get_fft_window_correction,
)
from scos_actions.signal_processing.power_analysis import (
    apply_statistical_detector,
    calculate_pseudo_power,
    create_statistical_detector,
)

logger = logging.getLogger(__name__)


class PowerSpectralDensity(ProductInterface):
    """
    Power Spectral Density
    """

    # FFT parameter keys
    NUM_FFTS = "nffts"
    FFT_SIZE = "fft_size"
    SAMPLE_RATE = "sample_rate"

    # FFT constants
    FFT_PERCENTILES = np.array([25, 75, 90, 95, 99, 99.9, 99.99])
    FFT_WINDOW_TYPE = "flattop"
    FFT_WINDOW_CORRECTION_TYPE = "energy"
    IMPEDANCE_OHMS = 50.0

    def __init__(self, params: dict):
        super().__init__(params)

    def _configure_from_params(self, params: dict) -> None:
        self.detector = create_statistical_detector(
            "FftM3Detector", ["max", "mean", "median"]
        )
        self.percentiles = np.asarray(self.FFT_PERCENTILES)
        self.fft_size = params[self.FFT_SIZE]
        self.fft_window = get_fft_window(self.FFT_WINDOW_TYPE, self.fft_size)
        fft_window_correction = get_fft_window_correction(
            self.fft_window, self.FFT_WINDOW_CORRECTION_TYPE
        )
        self.num_ffts = params[self.NUM_FFTS]
        # Get truncation points: truncate FFT result to middle 125 samples (middle 10 MHz from 14 MHz)
        sample_rate_Hz = params[self.SAMPLE_RATE]
        self.bin_start = int(self.fft_size / 7)     # bin_start = 25 with FFT_SIZE 175
        self.bin_end = self.fft_size - self.bin_start       # bin_end = 150 with FFT_SIZE 175
        # Compute the amplitude shift for PSD scaling. The FFT result
        # is in pseudo-power log units and must be scaled to a PSD.
        self.fft_scale_factor = (
            -10.0 * np.log10(self.IMPEDANCE_OHMS)       # Pseudo-power to power
            + 27.0          # Watts to dBm (+30) and baseband to RF (-3)
            - 10.0 * np.log10(sample_rate_Hz * self.fft_size)       # PSD scaling
            + 20.0 * np.log10(fft_window_correction)        # Window energy correction
        )

    def compute(self, iq: np.ndarray) -> np.ndarray:
        """
        Compute data product mean/max PSD results from IQ samples.

        The result is a PSD, with amplitudes in dBm/Hz referenced to the
        calibration terminal (assuming IQ data has been scaled for calibration).
        The result is truncated in frequency to the middle 10 MHz.

        :param iq: Complex-valued input waveform samples.
        :return: A 2D NumPy array of statistical detector results computed
            from PSD amplitudes, ordered (max, mean, median, percentiles).
        """
        fft_amplitudes = get_fft(
            iq, self.fft_size, "backward", self.fft_window, self.num_ffts, False, 1
        )
        # Power in Watts
        fft_amplitudes = calculate_pseudo_power(fft_amplitudes)
        fft_result = apply_statistical_detector(fft_amplitudes, self.detector)      # (max, mean, median)
        percentile_result = np.percentile(
            fft_amplitudes, self.percentiles, axis=0
        )
        fft_result = np.vstack((fft_result, percentile_result))
        fft_result = np.fft.fftshift(fft_result, axes=(1,))     # Shift frequencies
        fft_result = fft_result[:, self.bin_start : self.bin_end]       # Truncation to middle bins
        fft_result = 10.0 * np.log10(fft_result) + self.fft_scale_factor

        # Returned order is (max, mean, median, 25%, 75%, 90%, 95%, 99%, 99.9%, 99.99%)
        # Total of 10 arrays, each of length 125 (output shape (10, 125))
        # Percentile computation linearly interpolates. See numpy documentation.
        return fft_result


@ray.remote
class PowerSpectralDensityRayActor(PowerSpectralDensity):
    """
    Ray Wrapper (Only handles the distributed execution)
    """

    def __init__(self, params: dict):
        """
        less indirection during actor startup: Ray is creating and hydrating the actor in a separate worker process;
        keeping actor __init__ flat reduces moving parts
        avoids relying on MRO/parent init behavior inside Ray-managed process startup
        """
        self._configure_from_params(params)

    def run(self, iq: ray.ObjectRef) -> np.ndarray:
        """
        Calls local compute method internally on the worker node
        """
        return self.compute(iq)