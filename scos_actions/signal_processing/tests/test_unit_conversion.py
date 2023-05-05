"""
Unit test for scos_actions.signal_processing.unit_conversion
"""
import numpy as np
import pytest

from scos_actions.signal_processing import NUMEXPR_THRESHOLD
from scos_actions.signal_processing import unit_conversion as uc


@pytest.fixture
def small_array_len():
    return 10


def test_suppress_divide_by_zero_when_testing():
    uc.suppress_divide_by_zero_when_testing()
    _ = np.ones(5) / 0


def test_convert_watts_to_dBm(small_array_len):
    # 1 Watt is 30 dBm
    # 0.001 Watts is 0 dBm
    test_inputs = [1.0, 0.001]
    correct_results = [30.0, 0.0]

    def test_convert(val, correct_val):
        r = uc.convert_watts_to_dBm(val)
        if isinstance(val, np.ndarray):
            np.testing.assert_array_equal(r, np.ones_like(val) * correct_val)
        elif isinstance:
            assert r == correct_val

    for v, c in zip(test_inputs, correct_results):
        test_convert(v, c)
        if v == 1:
            test_convert(int(v), c)
        test_convert(np.ones(small_array_len) * v, c)
        test_convert(np.ones(NUMEXPR_THRESHOLD) * v, c)


def test_convert_dBm_to_watts(small_array_len):
    # 30 dBm is 1 Watt
    # 0 dBm is 0.001 Watts
    test_inputs = [30.0, 0.0]
    correct_results = [1.0, 0.001]

    def test_convert(val, correct_val):
        r = uc.convert_dBm_to_watts(val)
        if isinstance(val, np.ndarray):
            np.testing.assert_array_equal(r, np.ones_like(val) * correct_val)
        elif isinstance:
            assert r == correct_val

    for v, c in zip(test_inputs, correct_results):
        test_convert(v, c)
        if v == 30:
            test_convert(int(v), c)
        test_convert(np.ones(small_array_len) * v, c)
        test_convert(np.ones(NUMEXPR_THRESHOLD) * v, c)


def test_convert_linear_to_dB(small_array_len):
    # 10log10(1) is 0
    # 10log10(10) is 10
    test_inputs = [1.0, 10.0]
    correct_results = [0.0, 10.0]

    def test_convert(val, correct_val):
        r = uc.convert_linear_to_dB(val)
        if isinstance(val, np.ndarray):
            np.testing.assert_array_equal(r, np.ones_like(val) * correct_val)
        elif isinstance:
            assert r == correct_val

    for v, c in zip(test_inputs, correct_results):
        test_convert(v, c)
        if v == 1:
            test_convert(int(v), c)
        test_convert(np.ones(small_array_len) * v, c)
        test_convert(np.ones(NUMEXPR_THRESHOLD) * v, c)


def test_convert_dB_to_linear(small_array_len):
    # 10log10(1) is 0
    # 10log10(10) is 10
    test_inputs = [0.0, 10.0]
    correct_results = [1.0, 10.0]

    def test_convert(val, correct_val):
        r = uc.convert_dB_to_linear(val)
        if isinstance(val, np.ndarray):
            np.testing.assert_array_equal(r, np.ones_like(val) * correct_val)
        elif isinstance:
            assert r == correct_val

    for v, c in zip(test_inputs, correct_results):
        test_convert(v, c)
        if v == 1:
            test_convert(int(v), c)
        test_convert(np.ones(small_array_len) * v, c)
        test_convert(np.ones(NUMEXPR_THRESHOLD) * v, c)


def test_convert_kelvins_to_celsius(small_array_len):
    # 273.15 K is 0 C
    # 373.15 K is 100 C
    test_inputs = [273.15, 373.15]
    correct_results = [0.0, 100.0]

    def test_convert(val, correct_val):
        r = uc.convert_kelvins_to_celsius(val)
        if isinstance(val, np.ndarray):
            np.testing.assert_array_equal(r, np.ones_like(val) * correct_val)
        elif isinstance:
            assert r == correct_val

    for v, c in zip(test_inputs, correct_results):
        test_convert(v, c)
        if v == 1:
            test_convert(int(v), c)
        test_convert(np.ones(small_array_len) * v, c)


def test_convert_celsius_to_kelvins(small_array_len):
    # 273.15 K is 0 C
    # 373.15 K is 100 C
    test_inputs = [0.0, 100.0]
    correct_results = [273.15, 373.15]

    def test_convert(val, correct_val):
        r = uc.convert_celsius_to_kelvins(val)
        if isinstance(val, np.ndarray):
            np.testing.assert_array_equal(r, np.ones_like(val) * correct_val)
        elif isinstance:
            assert r == correct_val

    for v, c in zip(test_inputs, correct_results):
        test_convert(v, c)
        if v == 1:
            test_convert(int(v), c)
        test_convert(np.ones(small_array_len) * v, c)


def test_convert_fahrenheit_to_celsius(small_array_len):
    # 32 F is 0 C
    # 212 F is 100 C
    test_inputs = [32.0, 212.0]
    correct_results = [0.0, 100.0]

    def test_convert(val, correct_val):
        r = uc.convert_fahrenheit_to_celsius(val)
        if isinstance(val, np.ndarray):
            np.testing.assert_array_equal(r, np.ones_like(val) * correct_val)
        elif isinstance:
            assert r == correct_val

    for v, c in zip(test_inputs, correct_results):
        test_convert(v, c)
        if v == 1:
            test_convert(int(v), c)
        test_convert(np.ones(small_array_len) * v, c)


def test_convert_celsius_to_fahrenheit(small_array_len):
    # 32 F is 0 C
    # 212 F is 100 C
    test_inputs = [0.0, 100.0]
    correct_results = [32.0, 212.0]

    def test_convert(val, correct_val):
        r = uc.convert_celsius_to_fahrenheit(val)
        if isinstance(val, np.ndarray):
            np.testing.assert_array_equal(r, np.ones_like(val) * correct_val)
        elif isinstance:
            assert r == correct_val

    for v, c in zip(test_inputs, correct_results):
        test_convert(v, c)
        if v == 1:
            test_convert(int(v), c)
        test_convert(np.ones(small_array_len) * v, c)
