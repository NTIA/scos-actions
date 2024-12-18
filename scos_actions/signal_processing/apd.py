import numexpr as ne
import numpy as np

from scos_actions.signal_processing import NUMEXPR_THRESHOLD


def get_apd(
    time_data: np.ndarray,
    bin_size_dB: float = None,
    min_bin: float = None,
    max_bin: float = None,
    impedance_ohms: float = None,
) -> tuple[np.ndarray, np.ndarray]:
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
    if time_data.size < NUMEXPR_THRESHOLD:
        all_amps = np.abs(time_data)
    else:
        all_amps = ne.evaluate("abs(x).real", {"x": time_data})

    # Replace any 0 value amplitudes with NaN
    all_amps[all_amps == 0] = np.nan

    # Convert amplitudes from V to pseudo-power
    if time_data.size < NUMEXPR_THRESHOLD:
        all_amps = 20.0 * np.log10(all_amps)
    else:
        ne.evaluate("20*log10(all_amps)", out=all_amps)

    if bin_size_dB is None or bin_size_dB == 0:
        downsampling = False
        # No downsampling. min_bin and max_bin ignored.
        a = np.sort(all_amps)
        p = 1 - ((np.arange(len(a)) + 1) / len(a))
        # Replace peak amplitude 0 count with NaN
        p[-1] = np.nan
    else:
        downsampling = True
        # Dynamically get bin edges if necessary
        if min_bin is None:
            min_bin = np.nanmin(all_amps)
        if max_bin is None:
            max_bin = np.nanmax(all_amps)
        if min_bin >= max_bin:
            raise ValueError(
                f"Minimum APD bin {min_bin} is not less than maximum {max_bin}"
            )
        # Check that downsampling range is evenly spanned by bins
        if not ((max_bin - min_bin) / bin_size_dB).is_integer():
            raise ValueError(
                "APD downsampling range is not evenly spanned by configured bin size."
            )
        # Scale bin edges to the correct units if necessary
        if impedance_ohms is not None:
            min_bin, max_bin = (
                b + 10.0 * np.log10(impedance_ohms) for b in [min_bin, max_bin]
            )
        # Generate bins based on bin_size_dB for downsampling
        a = np.arange(min_bin, max_bin + bin_size_dB, bin_size_dB, dtype=float)
        assert np.isclose(
            a[1] - a[0], bin_size_dB
        )  # Checks against undesired arange behavior

        # Get counts of amplitudes exceeding each bin value
        p = sample_ccdf(all_amps, a)
        # Replace any 0 probabilities with NaN
        p[p == 0] = np.nan

    # Scale to power if impedance value provided
    if impedance_ohms is not None:
        if downsampling:
            a -= 10.0 * np.log10(impedance_ohms)
        else:
            if a.size < NUMEXPR_THRESHOLD:
                a -= 10.0 * np.log10(impedance_ohms)
            else:
                ne.evaluate("a-(10*log10(impedance_ohms))", out=a)

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
        if a.size < NUMEXPR_THRESHOLD:
            ccdf /= a.size
        else:
            ne.evaluate("ccdf/a_size", {"ccdf": ccdf, "a_size": a.size}, out=ccdf)

    return ccdf
