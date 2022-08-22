import numpy as np
import pytest

from scos_actions.signal_processing import apd

rng = np.random.default_rng()


@pytest.fixture
def example_iq_data():
    """
    Generate complex samples, with real and imaginary parts
    being independent normally distributed random variables,
    with mean zero and variance 1/2.
    """
    n_samps = 10000
    std_dev = np.sqrt(2) / 2.0
    samps = rng.normal(0, std_dev, n_samps) + 1j * rng.normal(0, std_dev, n_samps)
    return samps


def test_get_apd_nan_handling():
    # All zero amplitudes should be converted to NaN
    # Peak amplitude 0 count should be replaced with NaN
    zero_amps = np.zeros(10) * (1 + 1j)
    p, a = apd.get_apd(zero_amps)
    assert np.isnan(p[-1])
    assert all(np.isnan(a))


def test_get_apd_no_downsample(example_iq_data):
    bin_sizes = [None, 0]
    for bin_size in bin_sizes:
        apd_result = apd.get_apd(example_iq_data, bin_size)
        assert isinstance(apd_result, tuple)
        assert len(apd_result) == 2
        assert all(isinstance(x, np.ndarray) for x in apd_result)
        assert all(len(x) == len(example_iq_data) for x in apd_result)
        p, a = apd_result
        assert not any(x == 0 for x in a)
        np.testing.assert_equal(a, np.real(a))
        assert all(a[i] <= a[i + 1] for i in range(len(a) - 1))
        assert max(p) < 1
        assert min(p) > 0
        assert all(p[i + 1] <= p[i] for i in range(len(p) - 2))
        assert np.isnan(p[-1])


def test_get_apd_downsample(example_iq_data):
    bin_sizes = [0.5, 0.25, 0.15]
    for bin_size in bin_sizes:
        p, a = apd.get_apd(example_iq_data, bin_size)
        assert len(p) == len(a)
        assert len(p) < len(example_iq_data)
        assert not any(x == 0 for x in a)
        np.testing.assert_equal(a, np.real(a))
        assert all(a[i] <= a[i + 1] for i in range(len(a) - 1))
        np.testing.assert_allclose(
            np.diff(a), np.ones(len(a) - 1) * bin_size, rtol=1e-14, atol=0
        )
        assert max(p) < 1
        assert min(p) > 0
        assert all(p[i + 1] <= p[i] for i in range(len(p) - 2))
        assert np.isnan(p[-1])


def test_sample_ccdf():
    example_ccdf_bins = np.arange(0, 51, 1)
    example_ccdf_data = np.linspace(0, 50, 50)
    ccdf = apd.sample_ccdf(example_ccdf_data, example_ccdf_bins, density=False)
    ccdf_d = apd.sample_ccdf(example_ccdf_data, example_ccdf_bins, density=True)
    assert len(ccdf) == len(example_ccdf_bins)
    assert len(ccdf_d) == len(example_ccdf_bins)
    assert isinstance(ccdf, np.ndarray)
    assert isinstance(ccdf_d, np.ndarray)
    np.testing.assert_equal(ccdf_d, ccdf / len(example_ccdf_data))
    assert all(ccdf[i + 1] <= ccdf[i] for i in range(len(ccdf) - 1))
    assert all(ccdf_d[i + 1] <= ccdf_d[i] for i in range(len(ccdf_d) - 1))
