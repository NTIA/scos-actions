"""
Unit test for scos_actions.signal_processing.calibration
"""

import numpy as np
from scipy.constants import Boltzmann

from scos_actions.signal_processing import calibration


def test_y_factor():
    """
    Test Y-factor calculation of noise figure and gain.
    The test generates arrays of samples at mean power values
    calculated for a set of input gain and noise figure values.
    """
    # Set desired results for a few test cases
    excess_noise_ratio__dB = 15.0
    noise_figure__dB = [5.0, 10.0]
    gain__dB = [30.0, 40.0]

    # Other parameters for generating power samples
    number_of_samples = 1000
    temperature__K = 300.0
    bandwidth__Hz = 10e6
    excess_noise_ratio = 10.0 ** (excess_noise_ratio__dB / 10.0)
    ones_arr = np.ones(number_of_samples)
    ktb = Boltzmann * temperature__K * bandwidth__Hz

    # Iterate and test
    for nf__dB, g__dB in zip(noise_figure__dB, gain__dB):
        noise_factor = 10.0 ** (nf__dB / 10.0)
        gain = 10.0 ** (g__dB / 10.0)
        # Generated arrays have identical values, all representing
        # the value of mean received power. The function under
        # test computes the mean of these arrays, which is needed for
        # real-world data, but redundant for this test.
        pwr_noise_off__W = ones_arr * ktb * noise_factor * gain
        pwr_noise_on__W = ones_arr * ktb * (excess_noise_ratio + noise_factor) * gain

        nf_test, g_test = calibration.y_factor(
            pwr_noise_on__W,
            pwr_noise_off__W,
            excess_noise_ratio,
            bandwidth__Hz,
            temperature__K,
        )
        assert isinstance(nf_test, float)
        assert isinstance(g_test, float)
        assert np.isclose(nf_test, nf__dB)
        assert np.isclose(g_test, g__dB)


# NOT IMPLEMENTED: Requires connected preselector
# def test_get_linear_enr():
#     pass

# NOT IMPLEMENTED: Requires connected preselector with temperature sensor.
# def test_get_temperature():
#     pass
