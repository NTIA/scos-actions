def easy_gain(sample_rate: float, frequency: float, gain: int) -> float:
    """
    Create an easily interpolated calibration gain value for testing.

    :param sample_rate: Sample rate in samples per second
    :param frequency: Frequency in hertz
    :param gain: Signal analyzer gain setting in dB
    """
    return gain + (sample_rate / 1e6) + (frequency / 1e9)


def is_close(a, b, tolerance):
    """Handle floating point comparisons"""
    return abs(a - b) <= tolerance
