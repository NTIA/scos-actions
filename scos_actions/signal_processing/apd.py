import logging

import numexpr as ne
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def get_apd(time_data, bin_size_dB: float = 0.5, gain=0):
    """Estimate the APD by sampling the CCDF.

    The size of the output depends on the parameter bin_size_dB,
    which determines the effective downsampling of IQ data into an APD dataset.
    Higher bin sizes will lead to smaller data sizes but less resolution,
    inversley smaller bin sizes will reslut in larger data size output with more resolution.

    Setting the bin size to 0 will result in no downsampling of the data
    and will output the same data size as the input time_data.

    The gain arg will correct for added rf front end gain in order to get more accurate values
    of actual received power returned from this function.

    Parameters
    ----------
    time_data: Input complex baseband IQ samples.
    bin_size_dB: Amplitude granularity, in dB, for estimating the APD. A value of 0 will result in no downsampling of the apd.
    gain: correction value used to correct for rf front end gain

    Returns
    -------
    p: APD probabilities, scaled from 0 to 1.
    a: APD amplitudes.
    """
    # Convert IQ to amplitudes
    all_amps = ne.evaluate("abs(time_data).real")
    del time_data

    # Replace any 0 value amplitudes with NaN
    all_amps[all_amps == 0] = np.nan

    # Convert to dB(V^2)
    ne.evaluate("20*log10(all_amps)", out=all_amps)

    # Create amplitude bins
    if bin_size_dB != 0:
        a = np.arange(
            np.nanmin(all_amps), np.nanmax(all_amps) + bin_size_dB, bin_size_dB
        )
        # Get counts of amplitudes exceeding each bin value
        p = sample_ccdf(all_amps, a)
    else:
        a = np.sort(all_amps)
        p = 1 - ((np.arange(len(a)) + 1) / len(a))

    # Replace peak amplitude 0 count with NaN
    p[-1] = np.nan
    logger.debug(f"APD result length: {len(a)} samples.")

    # Gain corrections (dB(V^2) --> dBm, RF/baseband (-3 dB), system gain, impedance)
    a = a + 27 - gain - (10.0 * np.log10(50))

    return p, a


def sample_ccdf(a: np.array, edges: np.array, density: bool = True) -> np.array:
    """
    Computes the fraction (or total number) of samples in `a` that
    exceed each edge value.

    Args:
        a: the vector of input samples
        edges: sample threshold values at which to characterize the distribution
        density: if True, the sample counts are normalized by `a.size`
    Returns:
        the empirical complementary cumulative distribution
    """

    # 'left' makes the bin interval open-ended on the left side
    # (the CCDF is "number of samples exceeding interval", and not equal to)
    edge_inds = np.searchsorted(edges, a, side="left")
    bin_counts = np.bincount(edge_inds, minlength=edges.size + 1)
    ccdf = (a.size - bin_counts.cumsum())[:-1]

    if density:
        ccdf = ccdf.astype("float64")
        ccdf /= a.size

    return ccdf
