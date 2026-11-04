
"""Power vs. Time"""

import logging

import numpy as np

from scos_actions.metadata.structs import ntia_algorithm
from scos_actions.product.interfaces import (
    DataProductInterface,
    DataProductResult,
    ProductGraphInfo,
)
from scos_actions.signal_processing.power_analysis import (
    apply_statistical_detector,
    calculate_power_watts,
)

logger = logging.getLogger(__name__)

# PVT parameter keys
SAMPLE_RATE = "sample_rate"
TD_BIN_SIZE_MS = "td_bin_size_ms"
DURATION_MS = "duration_ms"


class PowerVsTime(DataProductInterface):
    """
    Compute mean/max PVT results and summary statistics from IQ samples.

    The result includes the PVT detector data, four summary statistics, and
    graph metadata.
    """
    def __init__(self, consts: dict):
        super().__init__(consts)

    def _configure_from_constants(self, consts: dict) -> None:
        self.detector = consts["TD_DETECTOR"]
        self.impedance_ohms = consts["IMPEDANCE_OHMS"]

    def set_parameters(self, params: dict) -> None:
        self.sample_rate = params[SAMPLE_RATE]
        self.duration_ms = params[DURATION_MS]
        self.td_bin_size_ms = params[TD_BIN_SIZE_MS]
        self.block_size = int(self.td_bin_size_ms * self.sample_rate * 1e-3)

    def compute(
        self, iq: np.ndarray, params: dict
    ) -> DataProductResult:
        """
        Compute power versus time results from IQ samples.

        :param iq: Complex-valued input waveform samples (which should
            already exist in the Ray object store).
        :return: PVT data, four summary statistics, and graph metadata.
        """
        # Reshape IQ data into blocks and calculate power
        recording_id = params["recording_id"]
        logger.warning("PowerVsTime started: %s", recording_id)

        self.set_parameters(params)

        n_blocks = len(iq) // self.block_size
        iq_pwr = calculate_power_watts(
            iq.reshape((n_blocks, self.block_size)), self.impedance_ohms
        )
        # Get true median power
        pvt_median = np.median(iq_pwr.flatten())
        # Apply max/mean detectors
        pvt_result = apply_statistical_detector(iq_pwr, self.detector, axis=1)
        # Get single value statistics: (max-of-max, median-of-mean, mean, median)
        pvt_summary = np.array(
            [
                pvt_result[0].max(),
                np.median(pvt_result[1]),
                pvt_result[1].mean(),
                pvt_median,
            ]
        )
        # Convert to dBm and account for RF/baseband power difference
        # Note: convert_watts_to_dBm is not used to avoid NumExpr usage
        # for the relatively small arrays
        pvt_result, pvt_summary = (
            10.0 * np.log10(x) + 27.0 for x in [pvt_result, pvt_summary]
        )

        logger.warning("PowerVsTime completed: %s", recording_id)
        # Return order ((max array, mean array), (max-of-max, median-of-mean, mean, median))
        return DataProductResult(pvt_result, pvt_summary, self.get_graph())

    def get_graph(self) -> ProductGraphInfo:
        num_iq_samples = int(self.sample_rate * self.duration_ms * 1e-3)
        pvt_length = int(self.duration_ms / self.td_bin_size_ms)
        pvt_x_axis__s = np.arange(pvt_length) * (self.duration_ms * 1e-3 / pvt_length)
        pvt_graph = ntia_algorithm.Graph(
            name="Power vs. Time",
            series=[d.value for d in self.detector],  # ["max", "mean"]
            length=pvt_length,
            x_units="s",
            x_start=[pvt_x_axis__s[0]],
            x_stop=[pvt_x_axis__s[-1]],
            x_step=[pvt_x_axis__s[1] - pvt_x_axis__s[0]],
            y_units="dBm",
            #reference=data_products_reference,
            description=(
                    "Max- and mean-detected channel power vs. time, with "
                    + f"an integration time of {self.td_bin_size_ms} ms. "
                    + "Each data point represents the result of a statistical "
                    + f"detector applied over the previous {self.td_bin_size_ms}."
                    + f" In total, {num_iq_samples} IQ samples were used as the input."
            ),
        )

        pvt_total_length = pvt_length * len(self.detector)

        return ProductGraphInfo(pvt_length, pvt_total_length, pvt_graph, None)