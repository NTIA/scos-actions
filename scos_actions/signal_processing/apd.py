import logging
from typing import Tuple

import numexpr as ne
import numpy as np

from scos_actions.signal_processing.unit_conversion import convert_linear_to_dB

logger = logging.getLogger(__name__)


def get_apd(
    time_data: np.ndarray,
    bin_size_dB: float = None,
    min_bin: float = None,
    max_bin: float = None,
    impedance_ohms: float = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate the APD by sampling the CCDF.

    The size of the output depends on ``bin_size_dB``, which
    determines the effective downsampling of IQ data into an APD dataset.
    Higher bin sizes will lead to smaller data sizes but less resolution.
    Inversely, smaller bin sizes will result in larger data size output
    with higher resolution.

    Not setting ``bin_size_dB`` will result in no downsampling of the data
    and will output the same data size as ``time_data``.

    If ``impedance_ohms`` is not provided, output units are determined by
    ``20*log10(input units)`` (dBV for ``time_data`` in Volts).
    If ``impedance_ohms`` is provided, output units are determined by
    ``10*log10(input units squared / Ohms)``. (dBW for ``time_data`` in Volts).

    :param time_data: Input complex baseband IQ samples.
    :param bin_size_dB: Amplitude granularity, in dB, for estimating the APD.
        If not specified, the APD will not be downsampled (default behavior).
        Setting this to zero will also result in no downsampling.
    :param min_bin: The minimum bin edge value for downsampling, in the
        same units as the output (dBW for Volts input).
    :param max_bin: The maximum bin edge value for downsampling, in the
        same units as the output (dBW for Volts input).
    :param impedance_ohms: The system's input impedance, in Ohms. Used
        to calculate power if provided.
    :return: A tuple (p, a) of NumPy arrays, where p contains the APD
        probabilities, and a contains the APD amplitudes.
    """
    # Convert IQ to amplitudes
    all_amps = ne.evaluate("abs(x).real", {"x": time_data})

    # Replace any 0 value amplitudes with NaN
    all_amps[all_amps == 0] = np.nan

    # Convert amplitudes from V to pseudo-power
    # Do not use utils.convert_linear_to_dB since the input
    # here is always an array, and generally a large one.
    ne.evaluate("20*log10(all_amps)", out=all_amps)

    if bin_size_dB is None or bin_size_dB == 0:
        # No downsampling
        if min_bin is not None or max_bin is not None:
            logger.warning(
                f"APD bin edge specified but no downsampling is being performed"
            )
        a = np.sort(all_amps)
        p = 1 - ((np.arange(len(a)) + 1) / len(a))
    else:
        # Dynamically get bin edges if necessary
        if min_bin is None:
            logger.debug("Setting APD minimum bin edge to minimum recorded amplitude")
            min_bin = np.nanmin(all_amps)
        if max_bin is None:
            logger.debug("Setting APD maximum bin edge to maximum recorded amplitude")
            max_bin = np.nanmax(all_amps)
        if min_bin >= max_bin:
            logger.error(
                f"Minimum APD bin {min_bin} is not less than maximum {max_bin}"
            )
        # Scale bin edges to the correct units if necessary
        if impedance_ohms is not None:
            min_bin, max_bin = (
                b + 10.0 * np.log10(impedance_ohms) for b in [min_bin, max_bin]
            )
        # Generate bins based on bin_size_dB for downsampling
        a = np.arange(min_bin, max_bin + bin_size_dB, bin_size_dB)
        # Get counts of amplitudes exceeding each bin value
        p = sample_ccdf(all_amps, a)

    # Replace peak amplitude 0 count with NaN
    p[-1] = np.nan

    # Scale to power if impedance value provided
    if impedance_ohms is not None:
        ne.evaluate("a-(10*log10(impedance_ohms))", out=a)

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
        ne.evaluate("ccdf/a_size", {"ccdf": ccdf, "a_size": a.size}, out=ccdf)

    return ccdf
