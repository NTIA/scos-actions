"""
Unit test for scos_actions.signal_processing.fft
"""
import numpy as np
import pytest
from scipy.signal import get_window

from scos_actions.signal_processing import fft

# Define the correct calculations. Simplified expressions are used
# in fft.get_fft_window_correction and fft.get_fft_enbw


def window_amplitude_correction(window: np.ndarray) -> float:
    return len(window) / np.sum(window)


def window_energy_correction(window: np.ndarray) -> float:
    return np.sqrt(len(window) / np.sum(np.square(window)))


def fft_window_enbw(window: np.ndarray) -> float:
    return (
        window_amplitude_correction(window) / window_energy_correction(window)
    ) ** 2.0


def fft_bin_enbw(window: np.ndarray, sample_rate__Hz: float) -> float:
    return (sample_rate__Hz / len(window)) * fft_window_enbw(window)


def test_get_fft():
    """
    Test the fft.get_fft method. This test does not check for mathematical correctness
    of the computed FFT, since that is handled by the SciPy function scipy.fft.fft.
    Instead, this test checks other aspects of the behavior of the fft.get_fft method
    """
    # Check that ValueErrors are correctly raised
    valerr_msg = "{var} must be an integer, not {type}."
    non_integers = ["invalid", 10.5, False]
    for x in non_integers:
        with pytest.raises(
            ValueError, match=valerr_msg.format(var="fft_size", type=type(x))
        ):
            _ = fft.get_fft(np.ones(10), x)
        with pytest.raises(
            ValueError, match=valerr_msg.format(var="num_ffts", type=type(x))
        ):
            _ = fft.get_fft(np.ones(10), 10, num_ffts=x)
    with pytest.raises(ValueError):
        _ = fft.get_fft(np.ones(10), 5, num_ffts=1)

    # Generate some data to use as time domain samples
    fft_size = 1024
    num_ffts = 5
    signal_amplitude = 500
    window = get_window("flattop", fft_size, True)
    window_acf = window_amplitude_correction(window)

    # Generated signal is constant: the FFT should be zero in all bins except
    # the first (without shifting) or center (with shifting), which should be signal_amplitude.
    def iq(length=num_ffts * fft_size, amplitude=signal_amplitude):
        return np.ones(length) * amplitude

    # Test with no window or normalization
    result = fft.get_fft(iq(), fft_size, "forward", None, num_ffts, False)
    # Check return type/shape just once
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.complex128
    assert result.shape == (num_ffts, fft_size)
    # Results here should be signal_amplitude in DC bin, zero elsewhere
    np.testing.assert_allclose(result[:, 0], np.ones(num_ffts) * signal_amplitude)
    np.testing.assert_allclose(result[:, 1:], np.zeros((num_ffts, fft_size - 1)))

    # Test window provided case
    result = fft.get_fft(iq(), fft_size, "forward", window, num_ffts, False)
    # Results here should be signal_amplitude * window_acf in DC bin
    np.testing.assert_allclose(
        result[:, 0] * window_acf, np.ones(num_ffts) * signal_amplitude
    )

    # Test num_ffts not provided case (with valid num_ffts, fft_size, unlike ValueError case above)
    result = fft.get_fft(iq(), fft_size)
    assert result.shape == (num_ffts, fft_size)

    # Test frequency shift
    result = fft.get_fft(iq(), fft_size, "forward", None, num_ffts, True)
    assert result.shape == (num_ffts, fft_size)

    np.testing.assert_allclose(
        result[:, fft_size // 2], np.ones(num_ffts) * signal_amplitude
    )
    np.testing.assert_allclose(
        result[:, : fft_size // 2], np.zeros((num_ffts, fft_size // 2))
    )
    np.testing.assert_allclose(
        result[:, fft_size // 2 + 1 :], np.zeros((num_ffts, fft_size // 2 - 1))
    )


def test_get_fft_window():
    # These window types are supported with SciPy >=1.8.0
    supported_window_types = [
        "boxcar",
        "triang",
        "blackman",
        "hamming",
        "hann",
        "bartlett",
        "flattop",
        "parzen",
        "bohman",
        "blackmanharris",
        "nuttall",
        "barthann",
        "cosine",
        "exponential",
        "tukey",
        "taylor",
    ]

    # These window types are not supported, since they require extra parameters
    unsupported_window_types = [
        "kaiser",
        "gaussian",
        "general_cosine",
        "general_gaussian",
        "general_hamming",
        "dpss",
        "chebwin",
    ]

    # As of SciPy 1.10.1, the following window is supported but not
    # tested for: "lanczos". The following window is unsupported, but not
    # tested for: "kaiser_bessel_derived".

    # These inputs don't work directly as inputs to scipy.signal.get_window,
    # but formatting is handled by scos_actions.signal_processing.fft.get_fft_window
    window_alt_format_map = {
        # "string supported by get_fft_window": "correct window type from SciPy"
        "Flat Top": "flattop",
        "hanning": "hann",
        "BOXCAR": "boxcar",
        " flat top ": "flattop",
    }

    # Check that all supported windows are correctly
    # generated using SciPy
    for w_type in supported_window_types:
        win = fft.get_fft_window(w_type, 1024)
        true_win = get_window(w_type, 1024, True)
        assert isinstance(win, np.ndarray)
        assert win.size == 1024
        assert np.array_equal(win, true_win)

    # These unsupported windows should cause a
    # ValueError when passed to SciPy without their
    # additional required parameters.
    for w_type in unsupported_window_types:
        with pytest.raises(
            ValueError,
            match=f"The '{w_type}' window needs one or more parameters -- pass a tuple.",
        ):
            _ = fft.get_fft_window(w_type, 1024)

    # Check that input formatting works as expected
    for bad_w_type, true_w_type in window_alt_format_map.items():
        win = fft.get_fft_window(bad_w_type, 1024)
        true_win = get_window(true_w_type, 1024, True)
        assert np.array_equal(win, true_win)
        with pytest.raises(ValueError, match="Unknown window type."):
            _ = get_window(bad_w_type, 1024, True)


def test_get_fft_window_correction():
    # Test a few common windows:
    test_windows = ["flattop", "hanning", "bartlett", "hamming", "blackmanharris"]

    for w_type in test_windows:
        window = fft.get_fft_window(w_type, 1024)
        true_amplitude_correction = window_amplitude_correction(window)
        true_energy_correction = window_energy_correction(window)
        test_amplitude_correction = fft.get_fft_window_correction(window, "amplitude")
        test_energy_correction = fft.get_fft_window_correction(window, "energy")
        assert np.isclose(true_amplitude_correction, test_amplitude_correction)
        assert np.isclose(true_energy_correction, test_energy_correction)
        assert isinstance(test_amplitude_correction, float)
        assert isinstance(test_energy_correction, float)

    # Function under test should raise a ValueError for invalid correction type
    bad_correction_type = ["amp", "e", "both", "other"]
    test_good_window = get_window("flattop", 1024, True)
    for ct in bad_correction_type:
        with pytest.raises(ValueError, match=f"Invalid window correction type: {ct}"):
            _ = fft.get_fft_window_correction(test_good_window, ct)


def test_get_fft_frequencies():
    # Construct a correct list of frequencies to check against
    sample_rate__Hz = 14e6
    fft_size = 875
    delta_f__Hz = sample_rate__Hz / fft_size
    center_frequency__Hz = 3555e6
    first_frequency = (
        center_frequency__Hz - (delta_f__Hz * fft_size // 2) + (delta_f__Hz / 2)
    )
    last_frequency = (
        center_frequency__Hz + (delta_f__Hz * fft_size // 2) - (delta_f__Hz / 2)
    )
    true_frequencies = np.linspace(
        first_frequency, last_frequency, fft_size, endpoint=True
    )
    assert all(d == delta_f__Hz for d in np.diff(true_frequencies))

    # Now run the test
    test_frequencies = fft.get_fft_frequencies(
        fft_size, sample_rate__Hz, center_frequency__Hz
    )
    assert isinstance(test_frequencies, list)
    assert len(test_frequencies) == fft_size
    np.testing.assert_allclose(np.array(test_frequencies), true_frequencies)


def test_get_fft_enbw():
    window_types = ["flattop", "hann", "blackmanharris"]
    fft_size = 1024
    sample_rate__Hz = 14e6
    for w_type in window_types:
        window = get_window(w_type, fft_size, True)
        true_fft_bin_enbw__Hz = fft_bin_enbw(window, sample_rate__Hz)
        test_fft_bin_enbw__Hz = fft.get_fft_enbw(window, sample_rate__Hz)
        assert isinstance(test_fft_bin_enbw__Hz, float)
        assert np.isclose(test_fft_bin_enbw__Hz, true_fft_bin_enbw__Hz)
