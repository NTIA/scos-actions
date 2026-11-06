"""Contract tests for product data, summaries, and graph metadata."""

from enum import Enum

import numpy as np

from scos_actions.data_product import amplitude_probability_distribution as apd_module
from scos_actions.data_product import periodic_frame_power as pfp_module
from scos_actions.data_product import power_spectral_density as psd_module
from scos_actions.data_product import power_vs_time as pvt_module
from scos_actions.data_product.interfaces import DataProductResult, DataProductGraphInfo


class PSDDetector(Enum):
    MAX = "max"
    MEAN = "mean"
    MEDIAN = "median"


class PVTDetector(Enum):
    MAX = "max"
    MEAN = "mean"


class PFPDetector(Enum):
    MIN = "min"
    MAX = "max"
    MEAN = "mean"


def test_psd_compute_returns_data_and_no_summary(monkeypatch):
    """Verify PSD data, no summary, and graph metadata use the shared contract."""
    monkeypatch.setattr(psd_module, "get_fft_window", lambda *_args: np.ones(7))
    monkeypatch.setattr(
        psd_module, "get_fft_window_correction", lambda *_args: 1.0
    )
    monkeypatch.setattr(
        psd_module,
        "get_fft",
        lambda *_args: np.ones((1, 7), dtype=np.complex64),
    )
    monkeypatch.setattr(
        psd_module, "calculate_pseudo_power", lambda _values: np.ones((1, 7))
    )
    monkeypatch.setattr(
        psd_module,
        "apply_statistical_detector",
        lambda *_args: np.ones((3, 7)),
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

    params = {
        "recording_id": 0,
        "sample_rate": 14e6,
        "duration_ms": 1,
        "fft_size": 7,
    }
    iq = np.ones(7, dtype=np.complex64)
    product.set_parameters(params)
    result = product.compute(iq, params["recording_id"])

    assert isinstance(result, DataProductResult)
    assert isinstance(result.graph_info, DataProductGraphInfo)
    assert result.data_result.shape == (5, 5)
    assert result.summary is None
    assert result.graph_info.data_length == 5
    assert result.graph_info.total_length == 25
    assert result.graph_info.dft is not None
    assert result.graph_info.graph.series == [
        "max",
        "mean",
        "median",
        "25th_percentile",
        "75th_percentile",
    ]


def test_pvt_compute_returns_data_and_summary(monkeypatch):
    """Verify PVT returns detector data and its four power summary statistics."""
    monkeypatch.setattr(
        pvt_module,
        "calculate_power_watts",
        lambda *_args: np.array([[1.0], [2.0], [3.0], [4.0]]),
    )
    monkeypatch.setattr(
        pvt_module,
        "apply_statistical_detector",
        lambda *_args, **_kwargs: np.array(
            [[4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0]]
        ),
    )
    product = pvt_module.PowerVsTime(
        {"TD_DETECTOR": PVTDetector, "IMPEDANCE_OHMS": 50.0}
    )

    params = {
        "recording_id": 0,
        "sample_rate": 1000,
        "duration_ms": 4,
        "td_bin_size_ms": 1,
    }
    iq = np.ones(4, dtype=np.complex64)
    product.set_parameters(params)
    result = product.compute(iq, params["recording_id"])

    assert isinstance(result, DataProductResult)
    assert result.data_result.shape == (2, 4)
    assert result.summary.shape == (4,)
    expected_summary = 10 * np.log10([4.0, 2.5, 2.5, 2.5]) + 27.0
    np.testing.assert_allclose(result.summary, expected_summary)
    assert result.graph_info.data_length == 4
    assert result.graph_info.total_length == 8
    assert result.graph_info.graph.series == ["max", "mean"]
    assert result.graph_info.graph.x_start == [0.0]
    assert result.graph_info.graph.x_stop == [0.003]
    assert result.graph_info.graph.x_step == [0.001]


def test_pfp_compute_returns_data_and_no_summary(monkeypatch):
    """Verify PFP output rows and graph series correspond to configured detectors."""
    product = pfp_module.PeriodicFramePower(
        {
            "PFP_M3_DETECTOR": PFPDetector,
            "IMPEDANCE_OHMS": 50.0,
            "PFP_FRAME_RESOLUTION_S": 1e-3,
        }
    )
    monkeypatch.setattr(
        pfp_module,
        "calculate_pseudo_power",
        lambda values: np.ones(values.shape),
    )
    monkeypatch.setattr(
        pfp_module,
        "apply_statistical_detector",
        lambda *_args, **_kwargs: np.ones((3, 2)),
    )

    params = {
        "recording_id": 0,
        "sample_rate": 1000,
        "duration_ms": 4,
        "td_bin_size_ms": 1,
        "pfp_frame_period_ms": 2,
    }
    iq = np.ones((4, 1), dtype=np.complex64)
    product.set_parameters(params)
    result = product.compute(iq, params["recording_id"])

    assert isinstance(result, DataProductResult)
    assert result.data_result.shape == (6, 2)
    assert result.summary is None
    assert result.graph_info.data_length == 2
    assert result.graph_info.total_length == 12
    assert result.graph_info.graph.series == [
        "mean_min",
        "mean_max",
        "mean_mean",
        "max_min",
        "max_max",
        "max_mean",
    ]


def test_apd_compute_returns_data_and_no_summary(monkeypatch):
    """Verify APD data and graph lengths are returned without a summary."""
    monkeypatch.setattr(
        apd_module,
        "get_apd",
        lambda *_args: (np.array([0.25, 0.75]), None),
    )
    product = apd_module.AmplitudeProbabilityDistribution({"IMPEDANCE_OHMS": 50.0})

    params = {
        "recording_id": 0,
        "apd_bin_size_dB": 1.0,
        "apd_min_bin_dBm": -100.0,
        "apd_max_bin_dBm": 0.0,
    }
    iq = np.ones(2, dtype=np.complex64)
    product.set_parameters(params)
    result = product.compute(iq, params["recording_id"])

    assert isinstance(result, DataProductResult)
    assert result.data_result.shape == (1, 2)
    assert result.summary is None
    assert result.graph_info.data_length == 101
    assert result.graph_info.total_length == 101
