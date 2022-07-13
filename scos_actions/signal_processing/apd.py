import logging
from typing import Tuple

import numexpr as ne
import numpy as np

from scos_actions.signal_processing.unit_conversion import convert_linear_to_dB

logger = logging.getLogger(__name__)


def get_apd(
    time_data: np.ndarray, bin_size_dB: float = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate the APD by sampling the CCDF.

    The size of the output depends on ``bin_size_dB``, which
    determines the effective downsampling of IQ data into an APD dataset.
    Higher bin sizes will lead to smaller data sizes but less resolution.
    Inversely, smaller bin sizes will result in larger data size output
    with higher resolution.

    Not setting ``bin_size_dB`` will result in no downsampling of the data
    and will output the same data size as ``time_data``.

    No additional scaling is applied, so resulting amplitude units are
    dBV. Typical applications will require converting this result to
    power units.

    :param time_data: Input complex baseband IQ samples.
    :param bin_size_dB: Amplitude granularity, in dB, for estimating the APD.
        If not specified, the APD will not be downsampled (default behavior).
    :return: A tuple (p, a) of NumPy arrays, where p contains the APD
        probabilities, and a contains the APD amplitudes.
    """
    # Convert IQ to amplitudes
    all_amps = ne.evaluate("abs(time_data).real")

    # Replace any 0 value amplitudes with NaN
    all_amps[all_amps == 0] = np.nan

    # Convert amplitudes from V to dBV
    all_amps = convert_linear_to_dB(all_amps)

    if bin_size_dB is None:
        # No downsampling
        a = np.sort(all_amps)
        p = 1 - ((np.arange(len(a)) + 1) / len(a))
    else:
        # Generate bins based on bin_size_dB for downsampling
        a = np.arange(
            np.nanmin(all_amps), np.nanmax(all_amps) + bin_size_dB, bin_size_dB
        )
        # Get counts of amplitudes exceeding each bin value
        p = sample_ccdf(all_amps, a)

    # Replace peak amplitude 0 count with NaN
    p[-1] = np.nan
    logger.debug(f"APD result length: {len(a)} samples.")

    return p, a


def sample_ccdf(a: np.ndarray, edges: np.ndarray, density: bool = True) -> np.ndarray:
    """
    Computes the fraction (or total number) of samples in `a` that
    exceed each edge value.

    :param a: the vector of input samples
    :param edges: sample threshold values at which to characterize the distribution
    :param density: if True, the sample counts are normalized by `a.size`
    :return: The empirical complementary cumulative distribution
    """
    # 'left' makes the bin interval open-ended on the left side
    # (the CCDF is "number of samples exceeding interval")
    edge_inds = np.searchsorted(edges, a, side="left")
    bin_counts = np.bincount(edge_inds, minlength=edges.size + 1)
    ccdf = (a.size - bin_counts.cumsum())[:-1]

    if density:
        ccdf = ccdf.astype("float64")
        ccdf /= a.size

    return ccdf
