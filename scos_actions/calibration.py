import json
import logging
from scos_actions.settings import SENSOR_CALIBRATION_FILE

logger = logging.getLogger(__name__)


class Calibration(object):
    def __init__(
        self,
        calibration_datetime,
        calibration_data,
        clock_rate_lookup_by_sample_rate,
        calibration_frequency_divisions=None,
    ):
        self.calibration_datetime = calibration_datetime
        self.calibration_data = calibration_data
        self.clock_rate_lookup_by_sample_rate = clock_rate_lookup_by_sample_rate
        self.calibration_frequency_divisions = sorted(
            calibration_frequency_divisions,
            key=lambda division: division["lower_bound"],
        )

    def get_clock_rate(self, sample_rate):
        """Find the clock rate (Hz) using the given sample_rate (samples per second)"""
        for mapping in self.clock_rate_lookup_by_sample_rate:
            mapped = freq_to_compare(mapping["sample_rate"])
            actual = freq_to_compare(sample_rate)
            if mapped == actual:
                return mapping["clock_frequency"]
        return sample_rate

    def get_calibration_dict(self, sample_rate, lo_frequency, setting_value):
        """Find the calibration points closest to the current frequency/setting value (gain, attenuation,ref_level...)."""

        # Check if the sample rate was calibrated
        sr = freq_to_compare(sample_rate)
        srs = sorted(self.calibration_data.keys())
        if sr not in srs:
            logger.warning("Requested sample rate was not calibrated!")
            logger.warning("Assuming default sample rate:")
            logger.warning("    Requested sample rate: {}".format(sr))
            logger.warning("    Assumed sample rate:   {}".format(srs[0]))
            sr = srs[0]

        # Get the nearest calibrated frequency and its index
        f = lo_frequency
        fs = sorted(self.calibration_data[sr].keys())
        f_i = -1
        bypass_freq_interpolation = True
        if f < fs[0]:  # Frequency below calibrated range
            f_i = 0
            logger.warning("Tuned frequency is below calibrated range!")
            logger.warning("Assuming lowest frequency:")
            logger.warning("    Tuned frequency:   {}".format(f))
            logger.warning("    Assumed frequency: {}".format(fs[f_i]))
        elif f > fs[-1]:  # Frequency above calibrated range
            f_i = len(fs) - 1
            logger.warning("Tuned frequency is above calibrated range!")
            logger.warning("Assuming highest frequency:")
            logger.warning("    Tuned frequency:   {}".format(f))
            logger.warning("    Assumed frequency: {}".format(fs[f_i]))
        else:
            # Ensure we use frequency interpolation
            bypass_freq_interpolation = False
            # Check if we are within a frequency division
            for div in self.calibration_frequency_divisions:
                if f > div["lower_bound"] and f < div["upper_bound"]:
                    logger.warning("Tuned frequency within a division!")
                    logger.warning("Assuming frequency at lower bound:")
                    logger.warning("    Tuned frequency:   {}".format(f))
                    logger.warning(
                        "    Lower bound:       {}".format(div["lower_bound"])
                    )
                    logger.warning(
                        "    Upper bound:       {}".format(div["upper_bound"])
                    )
                    logger.warning(
                        "    Assumed frequency: {}".format(div["lower_bound"])
                    )
                    f = div[
                        "lower_bound"
                    ]  # Interpolation will force this point; no interpolation error
            # Determine the index associated with the closest frequency less than or equal to f
            for i in range(len(fs) - 1):
                f_i = i
                # If the next frequency is larger, we're done
                if fs[i + 1] > f:
                    break

        # Get the nearest calibrated gain and its index
        specified_setting_value = setting_value
        calibrated_values = sorted(self.calibration_data[sr][fs[f_i]].keys())
        g_i = -1
        diff_specified_vs_calibrated = 0
        bypass_interpolation = True
        if specified_setting_value < calibrated_values[0]:  # Specified value below calibrated range
            g_i = 0
            diff_specified_vs_calibrated = specified_setting_value - calibrated_values[0]
            logger.warning("Current gain is below calibrated range!")
            logger.warning("Assuming lowest gain and extending:")
            logger.warning("    Current gain: {}".format(specified_setting_value))
            logger.warning("    Assumed gain: {}".format(calibrated_values[0]))
            logger.warning("    Fudge factor: {}".format(diff_specified_vs_calibrated))
        elif specified_setting_value > calibrated_values[-1]:  # Gain above calibrated range
            g_i = len(calibrated_values) - 1
            diff_specified_vs_calibrated = specified_setting_value - calibrated_values[-1]
            logger.warning("Current gain is above calibrated range!")
            logger.warning("Assuming lowest gain and extending:")
            logger.warning("    Current gain: {}".format(specified_setting_value))
            logger.warning("    Assumed gain: {}".format(calibrated_values[-1]))
            logger.warning("    Fudge factor: {}".format(diff_specified_vs_calibrated))
        else:
            # Ensure we use gain interpolation
            bypass_interpolation = False
            # Determine the index associated with the closest gain less than or equal to specified_setting_value
            for i in range(len(calibrated_values) - 1):
                g_i = i
                # If the next gain is larger, we're done
                if calibrated_values[i + 1] > specified_setting_value:
                    break

        # Get the list of calibration factors
        calibration_factors = self.calibration_data[sr][fs[f_i]][calibrated_values[g_i]].keys()

        # Interpolate as needed for each calibration point
        interpolated_calibration = {}
        for cal_factor in calibration_factors:
            if bypass_interpolation and bypass_freq_interpolation:
                factor = self.calibration_data[sr][fs[f_i]][calibrated_values[g_i]][cal_factor]
            elif bypass_freq_interpolation:
                factor = self.interpolate_1d(
                    specified_setting_value,
                    calibrated_values[g_i],
                    calibrated_values[g_i + 1],
                    self.calibration_data[sr][fs[f_i]][calibrated_values[g_i]][cal_factor],
                    self.calibration_data[sr][fs[f_i]][calibrated_values[g_i + 1]][cal_factor],
                )
            elif bypass_interpolation:
                factor = self.interpolate_1d(
                    f,
                    fs[f_i],
                    fs[f_i + 1],
                    self.calibration_data[sr][fs[f_i]][calibrated_values[g_i]][cal_factor],
                    self.calibration_data[sr][fs[f_i + 1]][calibrated_values[g_i]][cal_factor],
                )
            else:
                factor = self.interpolate_2d(
                    f,
                    specified_setting_value,
                    fs[f_i],
                    fs[f_i + 1],
                    calibrated_values[g_i],
                    calibrated_values[g_i + 1],
                    self.calibration_data[sr][fs[f_i]][calibrated_values[g_i]][cal_factor],
                    self.calibration_data[sr][fs[f_i + 1]][calibrated_values[g_i]][cal_factor],
                    self.calibration_data[sr][fs[f_i]][calibrated_values[g_i + 1]][cal_factor],
                    self.calibration_data[sr][fs[f_i + 1]][calibrated_values[g_i + 1]][cal_factor],
                )

            # Apply the setting fudge factor based off the calibration type
            if "gain" in cal_factor:
                factor += diff_specified_vs_calibrated
            if "noise_figure" in cal_factor:
                factor -= diff_specified_vs_calibrated
            if "compression" in cal_factor:
                factor -= diff_specified_vs_calibrated

            # Add the calibration factor to the interpolated list
            interpolated_calibration[cal_factor] = factor

        # Return the interpolated calibration factors
        return interpolated_calibration

    def interpolate_1d(self, x, x1, x2, y1, y2):
        """Interpolate between points in one dimension."""
        return y1 * (x2 - x) / (x2 - x1) + y2 * (x - x1) / (x2 - x1)

    def interpolate_2d(self, x, y, x1, x2, y1, y2, z11, z21, z12, z22):
        """Interpolate between points in two dimensions."""
        z_y1 = self.interpolate_1d(x, x1, x2, z11, z21)
        z_y2 = self.interpolate_1d(x, x1, x2, z12, z22)
        return self.interpolate_1d(y, y1, y2, z_y1, z_y2)

    def update(self, sample_rate, lo_frequency, setting_value, gain, noise_figure):
        sample_rates = self.calibration_data['sample_rates']
        updated = False
        for sr_cal in sample_rates:
            if sr_cal['sample_rate'] == sample_rate:
                cal_data = sr_cal['calibration_data']
                frequencies = cal_data['frequencies']
                for freq_cal in frequencies:
                    if freq_cal['frequency'] == lo_frequency:
                        cal_data = freq_cal['calibration_data']
                        setting_values = freq_cal['setting_values']
                        for setting_cal in setting_values:
                            if setting_cal['setting_value'] == setting_value:
                                updated = True
                                cal = setting_cal['calibration_data']
                                cal['gain_sensor'] = gain
                                cal['noise_figure_sensor'] = noise_figure
        if not updated:
             raise Exception('Sensor calibration file does not contain parameters to update.')

        else:
            with open(SENSOR_CALIBRATION_FILE, 'w') as outfile:
                json.dump(json_string, outfile)


def freq_to_compare(f):
    """Allow a frequency of type [float] to be compared with =="""
    f = int(round(f))
    return f


def load_from_json(fname):
    with open(fname) as file:
        calibration = json.load(file)

    # Check that the required fields are in the dict
    assert "calibration_datetime" in calibration
    assert "calibration_frequency_divisions" in calibration
    assert "calibration_data" in calibration
    assert "clock_rate_lookup_by_sample_rate" in calibration

    # Load all the calibration data
    calibration_data = {}
    for sample_rate_row in calibration["calibration_data"]["sample_rates"]:
        sr = freq_to_compare(sample_rate_row["sample_rate"])
        for frequency_row in sample_rate_row["calibration_data"]["frequencies"]:
            f = frequency_row["frequency"]
            for setting_value_row in frequency_row["calibration_data"]["setting_values"]:
                g = setting_value_row["setting_value"]
                cal_point = setting_value_row["calibration_data"]

                # initialize dictionaries
                if sr not in calibration_data.keys():
                    calibration_data[sr] = {}
                if f not in calibration_data[sr].keys():
                    calibration_data[sr][f] = {}
                calibration_data[sr][f][g] = cal_point

    # Create and return the Calibration object
    return Calibration(
        calibration["calibration_datetime"],
        calibration_data,
        calibration["clock_rate_lookup_by_sample_rate"],
        calibration["calibration_frequency_divisions"],
    )



