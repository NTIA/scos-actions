
"""Amplitude Probability Distribution"""

import logging

import numpy as np
import ray

from scos_actions.product.interfaces import ProductInterface
from scos_actions.signal_processing.apd import get_apd

logger = logging.getLogger(__name__)


class AmplitudeProbabilityDistribution(ProductInterface):
    """
    Compute a downsampled amplitude probability distribution from IQ samples.

    The result is the probability axis of the downsampled APD. Downsampling is
    configurable through defining a bin size, minimum bin value, and maximum bin
    value for binning amplitudes. The amplitude data can be reconstructed if these
    three parameters are known.
    """

    # APD parameter keys
    APD_BIN_SIZE_DB = "apd_bin_size_dB"
    APD_MIN_BIN_DBM = "apd_min_bin_dBm"
    APD_MAX_BIN_DBM = "apd_max_bin_dBm"

    # APD constants
    IMPEDANCE_OHMS = 50.0

    def __init__(self, params: dict):
        super().__init__(params)

    def _configure_from_params(self, params: dict) -> None:
        self.bin_size_dB = params[self.APD_BIN_SIZE_DB]
        self.impedance_ohms = self.IMPEDANCE_OHMS
        # get_apd requires amplitude bin edge values in dBW
        # Scale input to get_apd to account for:
        #     dBm -> dBW (-30)
        #     baseband -> RF power reference (+3)
        self.min_bin_dBW_RF = params[self.APD_MIN_BIN_DBM] - 27.0
        self.max_bin_dBW_RF = params[self.APD_MAX_BIN_DBM] - 27.0

    def compute(self, iq: np.ndarray) -> np.ndarray:
        """
        Generate the downsampled APD result from IQ samples.

        :param iq: Complex-valued input waveform samples.
        :return: A NumPy array containing the APD probability axis as percentages.
        """
        p, _ = get_apd(
            iq,
            self.bin_size_dB,
            self.min_bin_dBW_RF,
            self.max_bin_dBW_RF,
            self.impedance_ohms,
        )
        return p


@ray.remote
class AmplitudeProbabilityDistributionRayActor:
    """
    Compute a downsampled amplitude probability distribution from IQ samples.

    The result is the probability axis of the downsampled APD. Downsampling is
    configurable through defining a bin size, minimum bin value, and maximum bin
    value for binning amplitudes. The amplitude data can be reconstructed if these
    three parameters are known.
    """

    def __init__(
        self,
        bin_size_dB: float,
        min_bin_dBm_baseband: float,
        max_bin_dBm_baseband: float,
        impedance_ohms: float,
    ):
        self.bin_size_dB = bin_size_dB
        self.impedance_ohms = impedance_ohms
        # get_apd requires amplitude bin edge values in dBW
        # Scale input to get_apd to account for:
        #     dBm -> dBW (-30)
        #     baseband -> RF power reference (+3)
        self.min_bin_dBW_RF = min_bin_dBm_baseband - 27.0
        self.max_bin_dBW_RF = max_bin_dBm_baseband - 27.0

    def run(self, iq: ray.ObjectRef) -> np.ndarray:
        """
        Generate the downsampled APD result from IQ samples.

        :param iq: Complex-valued input waveform samples (which should
            already exist in the Ray object store).
        :return: A NumPy array containing the APD probability axis as percentages.
        """
        p, _ = get_apd(
            iq,
            self.bin_size_dB,
            self.min_bin_dBW_RF,
            self.max_bin_dBW_RF,
            self.impedance_ohms,
        )
        return p