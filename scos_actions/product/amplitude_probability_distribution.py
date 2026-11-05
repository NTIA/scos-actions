
"""Amplitude Probability Distribution"""

import logging

import numpy as np

from scos_actions.metadata.structs import ntia_algorithm
from scos_actions.product.interfaces import (
    DataProductInterface,
    DataProductResult,
    ProductGraphInfo,
)
from scos_actions.signal_processing.apd import get_apd

logger = logging.getLogger(__name__)

# APD parameter keys
APD_BIN_SIZE_DB = "apd_bin_size_dB"
APD_MIN_BIN_DBM = "apd_min_bin_dBm"
APD_MAX_BIN_DBM = "apd_max_bin_dBm"


class AmplitudeProbabilityDistribution(DataProductInterface):
    """
    Compute a downsampled amplitude probability distribution from IQ samples.

    The result is the probability axis of the downsampled APD. Downsampling is
    configurable through defining a bin size, minimum bin value, and maximum bin
    value for binning amplitudes. The amplitude data can be reconstructed if these
    three parameters are known.
    """

    def __init__(self, constants: dict):
        super().__init__(constants)

    def _configure_from_constants(self, constants: dict) -> None:
        self.impedance_ohms = constants["IMPEDANCE_OHMS"]

    def set_parameters(self, params: dict) -> None:
        self.bin_size_dB = params[APD_BIN_SIZE_DB]
        # get_apd requires amplitude bin edge values in dBW
        # Scale input to get_apd to account for:
        #     dBm -> dBW (-30)
        #     baseband -> RF power reference (+3)
        self.apd_min_bin_dbm = params[APD_MIN_BIN_DBM]
        self.apd_max_bin_dbm = params[APD_MAX_BIN_DBM]
        self.min_bin_dBW_RF = self.apd_min_bin_dbm - 27.0
        self.max_bin_dBW_RF = self.apd_max_bin_dbm - 27.0

    def compute(
        self, iq: np.ndarray, params: dict
    ) -> DataProductResult:
        """
        Generate the downsampled APD result from IQ samples.

        :param iq: Complex-valued input waveform samples (which should
            already exist in the Ray object store).
        :return: APD data, no summary, and graph metadata.
        """
        recording_id = params["recording_id"]
        logger.debug("AmplitudeProbabilityDistribution started: %s", recording_id)

        self.set_parameters(params)

        p, _ = get_apd(
            iq,
            self.bin_size_dB,
            self.min_bin_dBW_RF,
            self.max_bin_dBW_RF,
            self.impedance_ohms,
        )

        logger.debug("AmplitudeProbabilityDistribution completed: %s", recording_id)
        return DataProductResult(p.reshape(1, p.shape[0]), None, self.get_graph())

    def get_graph(self) -> ProductGraphInfo:
        apd_y_axis__dBm = np.arange(
            self.apd_min_bin_dbm,
            self.apd_max_bin_dbm + self.bin_size_dB,
            self.bin_size_dB,
        )
        apd_graph = ntia_algorithm.Graph(
            name="Amplitude Probability Distribution",
            length=apd_y_axis__dBm.size,
            x_units="percent",
            y_units="dBm",
            y_start=[apd_y_axis__dBm[0]],
            y_stop=[apd_y_axis__dBm[-1]],
            y_step=[apd_y_axis__dBm[1] - apd_y_axis__dBm[0]],
            #reference=data_products_reference,
            description=(
                    f"Estimate of the APD, using a {self.bin_size_dB} dB "
                    + "bin size for amplitude values. The data payload includes"
                    + " probability values, as percentages, indicating the "
                    + "probability that a given IQ sample exceeds the corresponding"
                    + " amplitudes, the y-axis values recorded by this metadata object."
            ),
        )

        apd_length = apd_graph.length

        return ProductGraphInfo(apd_length, apd_length, apd_graph, None)