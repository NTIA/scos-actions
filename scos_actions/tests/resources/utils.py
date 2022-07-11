def easy_gain(sample_rate, frequency, gain):
    """Create an easily interpolated calibration gain value for testing.

    :type sample_rate: float
    :param sample_rate: Sample rate in samples per second

    :type frequency: float
    :param frequency: Frequency in hertz

    :type gain: int
    :param gain: Signal analyzer gain setting in dB

    :rtype: float
    """
    return gain + (sample_rate / 1e6) + (frequency / 1e9)
