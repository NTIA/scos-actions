"""Power Spectral Density"""

import logging
from typing import Optional

import numpy as np

from scos_actions.metadata.structs import ntia_algorithm
from scos_actions.data_product.interfaces import (
    DataProductInterface,
    DataProductResult,
    DataProductGraphInfo,
)
from scos_actions.signal_processing.fft import get_fft, get_fft_window, get_fft_window_correction, get_fft_enbw, \
    get_fft_frequencies
from scos_actions.signal_processing.power_analysis import (
    apply_statistical_detector,
    calculate_pseudo_power,
)

logger = logging.getLogger(__name__)


# FFT parameter keys
SAMPLE_RATE = "sample_rate"
DURATION_MS = "duration_ms"
FFT_SIZE = "fft_size"


class PowerSpectralDensity(DataProductInterface):
    """
    Power Spectral Density
    """

    def __init__(self, consts: dict):
        super().__init__(consts)

    def _configure_from_constants(self, consts: dict) -> None:
        self.percentiles = consts["FFT_PERCENTILES"]
        self.fft_window_type = consts["FFT_WINDOW_TYPE"]
        self.fft_window_correction_type = consts["FFT_WINDOW_CORRECTION_TYPE"]
        self.detector = consts["FFT_M3_DETECTOR"]
        self.impedance_ohms = consts["IMPEDANCE_OHMS"]

    def set_parameters(self, params: dict) -> None:
        self.sample_rate = params[SAMPLE_RATE]
        self.duration_ms = params[DURATION_MS]
        self.fft_size = int(params[FFT_SIZE])
        self.fft_window = get_fft_window(self.fft_window_type, self.fft_size)
        self.fft_window_correction = get_fft_window_correction(
            self.fft_window, self.fft_window_correction_type
        )
        self.num_ffts = int(self.sample_rate * self.duration_ms * 1e-3 / self.fft_size)

        # Keep the center 5/7 of the FFT bins, covering 10 MHz of a 14 MHz capture.
        self.bin_start = int(self.fft_size / 7)  # bin_start = 25 with FFT_SIZE 175
        self.bin_end = self.fft_size - self.bin_start  # bin_end = 150 with FFT_SIZE 175
        # Compute the amplitude shift for PSD scaling. The FFT result
        # is in pseudo-power log units and must be scaled to a PSD.
        self.fft_scale_factor = (
                -10.0 * np.log10(self.impedance_ohms)  # Pseudo-power to power
                + 27.0  # Watts to dBm (+30) and baseband to RF (-3)
                - 10.0 * np.log10(self.sample_rate * self.fft_size)  # PSD scaling
                + 20.0 * np.log10(self.fft_window_correction)  # Window energy correction
        )

    def compute(
        self, iq: np.ndarray, recording_id: int
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Compute power spectral densities from IQ samples.

        :param iq: Complex-valued input waveform samples (which should
            already exist in the Ray object store).
        :return: PSD data, no summary, and graph metadata.
        """
        logger.debug("PowerSpectralDensity started: %s", recording_id)

        fft_amplitudes = get_fft(
            iq, self.fft_size, "backward", self.fft_window, self.num_ffts, False, 1
        )
        # Convert FFT magnitudes to pseudo-power before applying power detectors.
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

        logger.debug("PowerSpectralDensity completed: %s", recording_id)
        # Returned order is (max, mean, median, 25%, 75%, 90%, 95%, 99%, 99.9%, 99.99%)
        # Total of 10 arrays, each of length 125 (output shape (10, 125))
        # Percentile computation linearly interpolates. See numpy documentation.
        return fft_result, None

    def get_dft(self) -> ntia_algorithm.DFT:
        dft_obj = ntia_algorithm.DFT(
            id="psd_fft",
            equivalent_noise_bandwidth=round(get_fft_enbw(self.fft_window, self.sample_rate), 2),
            samples=self.fft_size,
            dfts=self.num_ffts,
            window=self.fft_window_type,
            baseband=True,
            description=f"First and last {int(self.fft_size / 7)} samples from {self.fft_size}-point FFT discarded",
        )

        return dft_obj

    def get_graph(self) -> DataProductGraphInfo:
        dft_obj = self.get_dft()
        psd_length = int(self.fft_size * (5 / 7))
        #psd_bin_start = int(self.fft_size / 7)  # bin_start = 125 with FFT_SIZE 875
        #psd_bin_end = self.fft_size - psd_bin_start  # bin_end = 750 with FFT_SIZE 875
        psd_x_axis__Hz = get_fft_frequencies(self.fft_size, self.sample_rate, 0.0)  # Baseband
        psd_graph = ntia_algorithm.Graph(
            name="Power Spectral Density",
            series=[d.value for d in self.detector]
                   + [
                       f"{int(p)}th_percentile" if p.is_integer() else f"{p}th_percentile"
                       for p in self.percentiles
                   ],  # ["max", "mean", "median", "25th_percentile", "75th_percentile", ... "99.99th_percentile"]
            length=psd_length,
            x_units="Hz",
            x_start=[psd_x_axis__Hz[self.bin_start]],
            x_stop=[psd_x_axis__Hz[self.bin_end - 1]],  # -1 for zero-indexed array
            x_step=[self.sample_rate / self.fft_size],
            y_units="dBm/Hz",
            processing=[dft_obj.id],
            #reference=data_products_reference,
            description=(
                    "Results of statistical detectors (max, mean, median, 25th_percentile, 75th_percentile, "
                    + "90th_percentile, 95th_percentile, 99th_percentile, 99.9th_percentile, 99.99th_percentile) "
                    + f"applied to power spectral density samples, with the first and last {int(self.fft_size / 7)} "
                    + "samples discarded. FFTs computed on IIR-filtered data."
            ),
        )

        psd_total_length = psd_length * (len(self.detector) + len(self.percentiles))

        return DataProductGraphInfo(psd_length, psd_total_length, psd_graph, dft_obj)