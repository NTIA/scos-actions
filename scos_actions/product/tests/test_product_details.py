from enum import Enum
from unittest.mock import Mock

import numpy as np
import pytest

from scos_actions.product import (
    amplitude_probability_distribution as apd_module,
    periodic_frame_power as pfp_module,
    power_spectral_density as psd_module,
)


class PSDDetector(Enum):
    MAX = "max"
    MEAN = "mean"
    MEDIAN = "median"


class PFPDetector(Enum):
    MIN = "min"
    MAX = "max"
    MEAN = "mean"


def test_apd_compute_converts_bin_limits_and_builds_y_axis(monkeypatch):
    """Verify APD converts dBm limits before computing its probability data."""
    probability = np.array([0.75, 0.25])
    get_apd_mock = Mock(return_value=(probability, None))
    monkeypatch.setattr(apd_module, "get_apd", get_apd_mock)
    product = apd_module.AmplitudeProbabilityDistribution({"IMPEDANCE_OHMS": 50.0})

    iq = np.array([1.0 + 1.0j])
    result = product.compute(
        iq,
        {
            "recording_id": 7,
            "apd_bin_size_dB": 2.0,
            "apd_min_bin_dBm": -100.0,
            "apd_max_bin_dBm": -94.0,
        },
    )

    get_apd_mock.assert_called_once_with(iq, 2.0, -127.0, -121.0, 50.0)
    assert result.data_result.tolist() == [[0.75, 0.25]]
    assert result.graph_info.graph.y_start == [-100.0]
    assert result.graph_info.graph.y_stop == [-94.0]
    assert result.graph_info.graph.y_step == [2.0]
    assert result.graph_info.data_length == 4
    assert result.graph_info.total_length == 4
    assert result.graph_info.dft is None


def test_psd_parameters_and_graph_metadata(monkeypatch):
    """Ensure PSD parameters produce the expected graph axes and DFT metadata."""
    window = np.ones(7)
    frequencies = np.arange(7, dtype=float)
    monkeypatch.setattr(psd_module, "get_fft_window", lambda *_args: window)
    monkeypatch.setattr(
        psd_module, "get_fft_window_correction", lambda *_args: 2.0
    )
    monkeypatch.setattr(psd_module, "get_fft_enbw", lambda *_args: 1.234)
    monkeypatch.setattr(
        psd_module, "get_fft_frequencies", lambda *_args: frequencies
    )
    product = psd_module.PowerSpectralDensity(
        {
            "FFT_PERCENTILES": np.array([25.0, 75.0]),
            "FFT_WINDOW_TYPE": "boxcar",
            "FFT_WINDOW_CORRECTION_TYPE": "energy",
            "FFT_M3_DETECTOR": PSDDetector,
            "IMPEDANCE_OHMS": 50.0,
        }
    )

    product.set_parameters(
        {
            "sample_rate": 1400,
            "duration_ms": 10,
            "fft_size": 7,
        }
    )
    graph_info = product.get_graph()

    assert product.fft_window is window
    assert product.fft_window_correction == 2.0
    assert product.num_ffts == 2.0
    assert (product.bin_start, product.bin_end) == (1, 6)
    assert graph_info.graph.series == [
        "max",
        "mean",
        "median",
        "25th_percentile",
        "75th_percentile",
    ]
    assert graph_info.graph.x_start == [1.0]
    assert graph_info.graph.x_stop == [5.0]
    assert graph_info.graph.x_step == [200.0]
    assert graph_info.data_length == 5
    assert graph_info.total_length == 25
    assert graph_info.dft.equivalent_noise_bandwidth == 1.23
    assert graph_info.dft.samples == 7
    assert graph_info.dft.dfts == 2.0


@pytest.mark.parametrize(
    "overrides",
    [
        {"pfp_frame_period_ms": 2.5},
        {"detector_period_s": 0.0015},
    ],
)
def test_pfp_rejects_periods_not_aligned_to_sample_rate(overrides):
    """Ensure PFP rejects frame and detector periods that do not align to samples."""
    detector_period_s = overrides.get("detector_period_s", 0.001)
    product = pfp_module.PeriodicFramePower(
        {
            "PFP_M3_DETECTOR": PFPDetector,
            "IMPEDANCE_OHMS": 50.0,
            "PFP_FRAME_RESOLUTION_S": detector_period_s,
        }
    )
    params = {
        "recording_id": 1,
        "sample_rate": 1000,
        "duration_ms": 4,
        "td_bin_size_ms": 1,
        "pfp_frame_period_ms": overrides.get("pfp_frame_period_ms", 2),
    }

    with pytest.raises(ValueError, match="integer multiple of sampling period"):
        product.set_parameters(params)


def test_pfp_parameters_derive_frame_dimensions():
    """Ensure PFP derives frame dimensions and power scaling from its parameters."""
    product = pfp_module.PeriodicFramePower(
        {
            "PFP_M3_DETECTOR": PFPDetector,
            "IMPEDANCE_OHMS": 50.0,
            "PFP_FRAME_RESOLUTION_S": 1e-3,
        }
    )

    product.set_parameters(
        {
            "sample_rate": 1000,
            "duration_ms": 4,
            "td_bin_size_ms": 1,
            "pfp_frame_period_ms": 2,
        }
    )

    assert product.n_frames == 2
    assert product.n_points == 2
    assert product.n_detectors == 3
    assert np.isclose(product.pfp_scale_factor, -10 * np.log10(50.0) + 27.0)
