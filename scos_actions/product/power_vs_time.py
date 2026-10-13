
"""Power vs. Time"""

import logging
from enum import EnumMeta

import numpy as np
import ray

from scos_actions.signal_processing.power_analysis import (
    apply_statistical_detector,
    calculate_power_watts,
)

logger = logging.getLogger(__name__)


@ray.remote
class PowerVsTime:
    """
    Compute mean/max PVT results and summary statistics from IQ samples.

    The results are two NumPy arrays. The first contains the full PVT detector
    results (``detector`` results with ``bin_size_ms`` resolution). The second
    contains two values: the maximum of the first detector result, and the median
    of the second detector result. These are used to obtain max-of-max and median-of-
    mean channel power summary statistics.
    """

    def __init__(
        self,
        sample_rate_Hz: float,
        bin_size_ms: float,
        impedance_ohms: float,
        detector: EnumMeta,
    ):
        self.block_size = int(bin_size_ms * sample_rate_Hz * 1e-3)
        self.detector = detector
        self.impedance_ohms = impedance_ohms

    def run(self, iq: ray.ObjectRef) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute power versus time results from IQ samples.

        :param iq: Complex-valued input waveform samples (which should
            already exist in the Ray object store).
        :return: Two NumPy arrays: the first has shape (2, 400) and
            the second is 1D with length 2. The first array contains
            the (max, mean) detector results and the second array contains
            the (max-of-max, median-of-mean, mean, median) single-valued
            summary statistics.
        """
        # Reshape IQ data into blocks and calculate power
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
        # Return order ((max array, mean array), (max-of-max, median-of-mean, mean, median))
        return pvt_result, pvt_summary