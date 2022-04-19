import datetime

from scos_usrp.hardware.calibration import Calibration


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
    return (gain) + (sample_rate / 1e6) + (frequency / 1e9)


def is_close(a, b, tolerance):
    """Handle floating point comparisons"""
    return abs(a - b) <= tolerance


def create_dummy_calibration(empty_cal=False):
    """Create a dummy calibration object"""

    # Define the calibration file steps
    sample_rates = [10e6, 15.36e6, 40e6]
    clock_frequencies = [40e6, 30.72e6, 40e6]
    gains = [20, 40, 60]
    frequencies = [1e9, 2e9, 3e9, 4e9]

    # Create the datetime
    calibration_datetime = "{}Z".format(datetime.datetime.utcnow().isoformat())

    # Create the sample/clock rate lookup
    calibration_sample_clock_rate_lookup = []
    for i in range(len(sample_rates)):
        calibration_sample_clock_rate_lookup.append(
            {
                "sample_rate": int(sample_rates[i]),
                "clock_frequency": int(clock_frequencies[i]),
            }
        )

    # Create the frequency divisions
    calibration_frequency_divisions = []

    # Create the actual data if not an empty cal file
    calibration_data = {}
    if not empty_cal:
        for sr in sample_rates:
            for f in frequencies:
                for g in gains:
                    # initialize dictionaries
                    if sr not in calibration_data.keys():
                        calibration_data[sr] = {}
                    if f not in calibration_data[sr].keys():
                        calibration_data[sr][f] = {}
                    calibration_data[sr][f][g] = {
                        "gain_sigan": easy_gain(sr, f, g),
                        "gain_preselector": -10,
                        "gain_sensor": easy_gain(sr, f, g) - 10,
                        "1db_compression_sensor": 1,
                    }
    else:  # Create an empty calibration file
        calibration_data = {  # Empty calibration file data
            10e6: {1e9: {40: {}, 60: {}}, 2e9: {40: {}, 60: {}}}
        }

    return Calibration(
        calibration_datetime,
        calibration_data,
        calibration_sample_clock_rate_lookup,
        calibration_frequency_divisions,
    )
