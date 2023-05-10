"""
Unit test for scos_actions.signal_processing.power_analysis
"""
from enum import EnumMeta

import numpy as np
import pytest

from scos_actions.signal_processing import NUMEXPR_THRESHOLD
from scos_actions.signal_processing import power_analysis as pa

# Test inputs: all values equal (4 + 3j) for testing
TEST_VAL_CPLX = 4 + 3j


@pytest.fixture
def large_array():
    return np.ones(NUMEXPR_THRESHOLD * 2) * TEST_VAL_CPLX


@pytest.fixture
def large_readonly_array():
    x = np.ones(NUMEXPR_THRESHOLD * 2) * TEST_VAL_CPLX
    x.setflags(write=False)
    return x


@pytest.fixture
def small_array():
    return np.ones(NUMEXPR_THRESHOLD // 2) * TEST_VAL_CPLX


@pytest.fixture
def scalar_array():
    return np.ones(1) * TEST_VAL_CPLX


@pytest.fixture
def complex_scalar():
    return TEST_VAL_CPLX


@pytest.fixture
def real_scalar():
    return np.abs(TEST_VAL_CPLX)


@pytest.fixture
def true_scalars(complex_scalar, real_scalar):
    return [complex_scalar, real_scalar]


@pytest.fixture
def all_arrays(large_array, small_array, scalar_array, large_readonly_array):
    return [large_array, small_array, scalar_array, large_readonly_array]


def test_calculate_power_watts(all_arrays, true_scalars):
    # (4 + 3j) Volt IQ has magnitude 5, result is 0.5 W @ 50 Ohms
    answer = 0.5
    for volts in all_arrays:
        watts = pa.calculate_power_watts(volts)
        np.testing.assert_array_equal(watts, answer * np.ones_like(watts))
    for volts in true_scalars:
        watts = pa.calculate_power_watts(volts)
        assert watts == answer


def test_calculate_pseudo_power(all_arrays, true_scalars):
    # (4 + 3j) Volt IQ has pseudo-power 25.0
    answer = 25.0
    for val in all_arrays:
        ps_pwr = pa.calculate_pseudo_power(val)
        np.testing.assert_array_equal(ps_pwr, answer * np.ones_like(val))
    for val in true_scalars:
        ps_pwr = pa.calculate_pseudo_power(val)
        assert ps_pwr == answer


def test_create_power_detector():
    # Ensure construction works with multiple inputs
    dets_correct_order = ["min", "max", "mean", "median", "sample"]
    dets_random_order = ["median", "max", "min", "sample", "mean"]
    # Detectors should always come out in "correct" order
    dets_correct_subset = ["max", "median", "sample"]
    dets_random_subset = ["max", "sample", "median"]
    det_enum = pa.create_power_detector("test", dets_random_order)
    det_subset_enum = pa.create_power_detector("subset", dets_random_subset)
    # Check enum lengths
    assert isinstance(det_enum, EnumMeta)
    assert isinstance(det_subset_enum, EnumMeta)
    assert len(det_enum) == len(dets_correct_order)
    assert len(det_subset_enum) == len(dets_correct_subset)
    # Check enum names
    assert dets_correct_order == [d.name for d in det_enum]
    assert dets_correct_subset == [d.name for d in det_subset_enum]
    # Check enum values
    assert [f"{d}_power" for d in dets_correct_order] == [d.value for d in det_enum]
    assert [f"{d}_power" for d in dets_correct_subset] == [
        d.value for d in det_subset_enum
    ]
    # Invalid input causes ValueError
    dets_with_invalid = ["max", "mean", "invalid"]
    with pytest.raises(ValueError):
        _ = pa.create_power_detector("invalid", dets_with_invalid)


def test_apply_power_detector():
    data_dim0_len = 10
    data_dim1_len = 5
    n_detectors = 5
    test_data = np.arange(data_dim0_len)
    test_data_odd = np.arange(data_dim0_len + 1)
    correct_result = np.array(
        [0, 9, 4.5, 4.5]
    )  # Do not include unpredictable sample det.
    correct_result_odd = np.array([0, 10, 5, 5])
    test_data_2D = np.arange(data_dim0_len * data_dim1_len).reshape(
        data_dim1_len, data_dim0_len
    )
    correct_result_2D_ax0 = np.array([np.arange(10) + i for i in [0, 40, 20, 20]])
    correct_result_2D_ax1 = np.array(
        [correct_result + 10 * i for i in range(data_dim1_len)]
    ).T
    det = pa.create_power_detector("test", ["min", "max", "mean", "median", "sample"])
    det_r = pa.apply_power_detector(test_data, det)
    det_r_odd = pa.apply_power_detector(test_data_odd, det)
    det_r_2D_ax0 = pa.apply_power_detector(test_data_2D.copy(), det)
    det_r_2D_ax1 = pa.apply_power_detector(test_data_2D.copy(), det, axis=1)
    np.testing.assert_array_equal(correct_result, det_r[: n_detectors - 1])
    np.testing.assert_array_equal(correct_result_odd, det_r_odd[: n_detectors - 1])
    assert det_r_2D_ax0.shape == (n_detectors, data_dim0_len)
    assert det_r_2D_ax1.shape == (n_detectors, data_dim1_len)
    np.testing.assert_array_equal(
        correct_result_2D_ax0, det_r_2D_ax0[: n_detectors - 1, :]
    )
    np.testing.assert_array_equal(
        correct_result_2D_ax1, det_r_2D_ax1[: n_detectors - 1, :]
    )
    # Sample detector result should be in the input data
    assert det_r[-1] in test_data
    assert det_r_odd[-1] in test_data_odd
    assert det_r_2D_ax0[n_detectors - 1] in test_data_2D
    assert det_r_2D_ax1[n_detectors - 1] in test_data_2D.T

    # Test nan failure if not ignored
    test_nan_data = np.array([1, 2, 3, np.nan])
    ignorenan_result = pa.apply_power_detector(test_nan_data, det, ignore_nan=True)
    np.testing.assert_array_equal(
        ignorenan_result[: n_detectors - 1], np.array([1, 3, 2, 2])
    )
    with pytest.raises(
        ValueError, match="Data contains NaN values but ``ignore_nan`` is False."
    ):
        _ = pa.apply_power_detector(test_nan_data, det)


def test_filter_quantiles():
    test_data = np.arange(100)
    large_test_data = np.arange(NUMEXPR_THRESHOLD)
    bad_inputs = [100.0, 5, "invalid", np.array(100.0)]
    lo_q, hi_q = 0.05, 0.95
    bad_lo_q, bad_hi_q = [0.96, -0.1], [0.04, 1.01]
    filtered = pa.filter_quantiles(test_data, lo_q, hi_q)
    np.testing.assert_array_equal(
        filtered,
        np.hstack((np.ones(5) * np.nan, np.arange(90) + 5, np.ones(5) * np.nan)),
    )
    # Large arrays are computed with NumExpr
    large_filtered = pa.filter_quantiles(large_test_data, lo_q, hi_q)
    large_data_range = int(NUMEXPR_THRESHOLD * lo_q), int(NUMEXPR_THRESHOLD * hi_q)
    large_correct_result = np.hstack(
        (
            np.ones(large_data_range[0]) * np.nan,
            large_test_data[large_data_range[0] : large_data_range[1]],
            np.ones(len(large_test_data) - large_data_range[1]) * np.nan,
        )
    )
    np.testing.assert_array_almost_equal(large_filtered, large_correct_result)
    # Scalar input or 1-element array should cause TypeError
    for i in bad_inputs:
        with pytest.raises(TypeError):
            _ = pa.filter_quantiles(i, lo_q, hi_q)
    for q in bad_lo_q:
        with pytest.raises(ValueError):
            _ = pa.filter_quantiles(test_data, q, hi_q)
    for q in bad_hi_q:
        with pytest.raises(ValueError):
            _ = pa.filter_quantiles(test_data, lo_q, q)
    # Complex input should raise TypeError
    test_complex_data = test_data + 1j * test_data
    with pytest.raises(TypeError):
        _ = pa.filter_quantiles(test_complex_data, lo_q, hi_q)
