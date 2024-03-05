"""
Unit test for scos_actions.signal_processing.filtering

Since most filtering functions are thin wrappers around SciPy, these
tests mostly exist to ensure that tests will fail if substantial changes
are made to the wrappers.
"""

import numpy as np
import pytest
from scipy.signal import ellip, ellipord, firwin, kaiserord, sos2zpk, sosfreqz

from scos_actions.signal_processing import filtering


@pytest.fixture
def gpass():  # Passband ripple, dB
    return 0.1


@pytest.fixture
def gstop():  # Stopband attenuation, dB
    return 10


@pytest.fixture
def pb():  # Passband edge frequency, Hz
    return 5000


@pytest.fixture
def sb():  # Stopband edge frequency, Hz
    return 7050


@pytest.fixture
def sr():  # Sample rate, Hz
    return 10e5


@pytest.fixture
def example_sos(gpass, gstop, pb, sb, sr):
    o, w = ellipord(pb, sb, gpass, gstop, False, sr)
    return ellip(o, gpass, gstop, w, "lowpass", False, "sos", sr)


def test_generate_elliptic_iir_low_pass_filter(example_sos, gpass, gstop, pb, sb, sr):
    with pytest.raises(ValueError):
        # Stopband edge must be a higher frequency than passband edge
        _ = filtering.generate_elliptic_iir_low_pass_filter(1, 3, 1000, 900, 1e6)
    test_sos = filtering.generate_elliptic_iir_low_pass_filter(gpass, gstop, pb, sb, sr)
    assert isinstance(test_sos, np.ndarray)
    np.testing.assert_array_equal(example_sos, test_sos)


def test_generate_fir_low_pass_filter():
    # Same approach as above for IIR: basically duplicate the functionality here
    att, wid, xoff, sr = 10, 100, 1000, 10e3
    o, b = kaiserord(att, wid / (0.5 * sr))
    true_taps = firwin(o + 1, xoff, wid, ("kaiser", b), "lowpass", True, fs=sr)
    test_taps = filtering.generate_fir_low_pass_filter(att, wid, xoff, sr)
    assert isinstance(test_taps, np.ndarray)
    assert test_taps.shape == (o + 1,)
    np.testing.assert_array_equal(test_taps, true_taps)


def test_get_iir_frequency_response(example_sos, pb, sb, sr):
    for worN in [100, np.linspace(pb - 500, sb + 500, 3050)]:
        true_w, true_h = sosfreqz(example_sos, worN, True, sr)
        test_w, test_h = filtering.get_iir_frequency_response(example_sos, worN, sr)
        if isinstance(worN, int):
            assert all(len(x) == worN for x in [test_w, test_h])
        elif isinstance(worN, np.ndarray):
            assert all(len(x) == len(worN) for x in [test_w, test_h])
        np.testing.assert_array_equal(true_h, test_h)
        np.testing.assert_array_equal(true_w, test_w)


def test_get_iir_phase_response(example_sos, pb, sb, sr):
    for worN in [100, np.linspace(pb - 500, sb + 500, 3050)]:
        true_w, h = sosfreqz(example_sos, worN, False, sr)
        true_angles = np.unwrap(np.angle(h))
        test_w, test_angles = filtering.get_iir_phase_response(example_sos, worN, sr)
        if isinstance(worN, int):
            assert all(len(x) == worN for x in [test_w, test_angles])
        elif isinstance(worN, np.ndarray):
            assert all(len(x) == len(worN) for x in [test_w, test_angles])
        np.testing.assert_array_equal(true_w, test_w)
        np.testing.assert_array_equal(true_angles, test_angles)


def test_get_iir_enbw(example_sos, sr):
    with pytest.raises(TypeError):
        _ = filtering.get_iir_enbw(example_sos, "invalid", sr)
    with pytest.raises(ValueError):
        _ = filtering.get_iir_enbw(example_sos, np.linspace(-sr, sr, 100), sr)
    enbw_test = filtering.get_iir_enbw(example_sos, 1000, sr)
    assert isinstance(enbw_test, float)
    assert enbw_test > 0


def test_is_stable(example_sos):
    stable_test = filtering.is_stable(example_sos)
    assert isinstance(stable_test, bool)
    assert stable_test is True
    _, poles, _ = sos2zpk(example_sos)
    stable_true = all([p < 1 for p in np.square(np.abs(poles))])
    assert stable_true == stable_test
