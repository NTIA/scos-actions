"""Test aspects of ScaleFactors."""

import datetime
import json
import random
from copy import deepcopy
from math import isclose
from pathlib import Path

import pytest
from scos_actions.calibration.sensor_calibration import SensorCalibration
from scos_actions.calibration.utils import CalibrationException, filter_by_parameter
from scos_actions.tests.resources.utils import easy_gain
from scos_actions.utils import get_datetime_str_now, parse_datetime_iso_format_str


class TestSensorCalibrationFile:
    # Ensure we load the test file
    setup_complete = False

    def rand_index(self, l):
        """Get a random index for a list"""
        return random.randint(0, len(l) - 1)

    def check_duplicate(self, sr, f, g):
        """Check if a set of points was already tested"""
        for pt in self.pytest_points:
            duplicate_f = f == pt["frequency"]
            duplicate_g = g == pt["setting_value"]
            duplicate_sr = sr == pt["sample_rate"]
            if duplicate_f and duplicate_g and duplicate_sr:
                return True

    def run_pytest_point(self, sr, f, g, reason, sr_m=False, f_m=False, g_m=False):
        """Test the calculated value against the algorithm
        Parameters:
            sr, f, g -> Set values for the mock USRP
            reason: Test case string for failure reference
            sr_m, f_m, g_m -> Set values to use when calculating the expected value
                              May differ in from actual set points in edge cases
                              such as tuning in divisions or uncalibrated sample rate"""
        # Check that the setup was completed
        assert self.setup_complete, "Setup was not completed"

        # If this point was tested before, skip it (triggering a new one)
        if self.check_duplicate(sr, f, g):
            return False

        # If the point doesn't have modified inputs, use the algorithm ones
        if not f_m:
            f_m = f
        if not g_m:
            g_m = g
        if not sr_m:
            sr_m = sr

        # Calculate what the scale factor should be
        calc_gain_sigan = easy_gain(sr_m, f_m, g_m)

        # Get the scale factor from the algorithm
        interp_cal_data = self.sample_cal.get_calibration_dict([sr, f, g])
        interp_gain_siggan = interp_cal_data["gain"]

        # Save the point so we don't duplicate
        self.pytest_points.append(
            {
                "sample_rate": int(sr),
                "frequency": f,
                "setting_value": g,
                "gain": calc_gain_sigan,
                "test": reason,
            }
        )

        # Check if the point was calculated correctly
        tolerance = 1e-5
        msg = "Scale factor not correctly calculated!\r\n"
        msg = f"{msg}    Expected value:   {calc_gain_sigan}\r\n"
        msg = f"{msg}    Calculated value: {interp_gain_siggan}\r\n"
        msg = f"{msg}    Tolerance: {tolerance}\r\n"
        msg = f"{msg}    Test: {reason}\r\n"
        msg = f"{msg}    Sample Rate: {sr / 1e6}({sr_m / 1e6})\r\n"
        msg = f"{msg}    Frequency: {f / 1e6}({f_m / 1e6})\r\n"
        msg = f"{msg}    Gain: {g}({g_m})\r\n"
        msg = (
            "{}    Formula: -1 * (Gain - Frequency[GHz] - Sample Rate[MHz])\r\n".format(
                msg
            )
        )
        if not isclose(calc_gain_sigan, interp_gain_siggan, abs_tol=tolerance):
            interp_cal_data = self.sample_cal.get_calibration_dict([sr, f, g])

        assert isclose(calc_gain_sigan, interp_gain_siggan, abs_tol=tolerance), msg
        return True

    @pytest.fixture(autouse=True)
    def setup_calibration_file(self, tmpdir):
        """Create the dummy calibration file in the pytest temp directory"""

        # Only setup once
        if self.setup_complete:
            return

        # Create and save the temp directory and file
        self.tmpdir = tmpdir.strpath
        self.calibration_file = "{}".format(tmpdir.join("dummy_cal_file.json"))

        # Setup variables
        self.dummy_noise_figure = 10
        self.dummy_compression = -20
        self.test_repeat_times = 3

        # Sweep variables
        self.sample_rates = [10e6, 15.36e6, 40e6]
        self.gain_min = 40
        self.gain_max = 60
        self.gain_step = 10
        gains = list(range(self.gain_min, self.gain_max, self.gain_step)) + [
            self.gain_max
        ]
        self.frequency_min = 1000000000
        self.frequency_max = 3400000000
        self.frequency_step = 200000000
        frequencies = list(
            range(self.frequency_min, self.frequency_max, self.frequency_step)
        ) + [self.frequency_max]
        frequencies = sorted(frequencies)

        # Start with blank cal data dicts
        cal_data = {}

        # Add the simple stuff to new cal format
        cal_data["last_calibration_datetime"] = get_datetime_str_now()
        cal_data["sensor_uid"] = "SAMPLE_CALIBRATION"

        # Add SR/CF lookup table
        cal_data["clock_rate_lookup_by_sample_rate"] = []
        for sr in self.sample_rates:
            cr = sr
            while cr <= 40e6:
                cr *= 2
            cr /= 2
            cal_data["clock_rate_lookup_by_sample_rate"].append(
                {"sample_rate": int(sr), "clock_frequency": int(cr)}
            )

        # Create the JSON architecture for the calibration data
        cal_data["calibration_data"] = {}
        cal_data["calibration_parameters"] = ["sample_rate", "frequency", "gain"]
        for k in range(len(self.sample_rates)):
            cal_data_f = {}
            for i in range(len(frequencies)):
                cal_data_g = {}
                for j in range(len(gains)):
                    # Create the scale factor that ensures easy interpolation
                    gain_sigan = easy_gain(
                        self.sample_rates[k], frequencies[i], gains[j]
                    )

                    # Create the data point
                    cal_data_point = {
                        "gain": gain_sigan,
                        "noise_figure": self.dummy_noise_figure,
                        "1dB_compression_point": self.dummy_compression,
                    }

                    # Add the generated dicts to the parent lists
                    cal_data_g[gains[j]] = deepcopy(cal_data_point)
                cal_data_f[frequencies[i]] = deepcopy(cal_data_g)

            cal_data["calibration_data"][self.sample_rates[k]] = deepcopy(cal_data_f)

        # Write the new json file
        with open(self.calibration_file, "w+") as file:
            json.dump(cal_data, file, indent=4)

        # Load the data back in
        self.sample_cal = SensorCalibration.from_json(self.calibration_file, False)

        # Create a list of previous points to ensure that we don't repeat
        self.pytest_points = []

        # Create sweep lists for test points
        self.srs = self.sample_rates
        self.gi_s = list(range(self.gain_min, self.gain_max, self.gain_step))
        self.fi_s = list(
            range(self.frequency_min, self.frequency_max, self.frequency_step)
        )
        self.g_s = self.gi_s + [self.gain_max]
        self.f_s = self.fi_s + [self.frequency_max]

        # Don't repeat test setup
        self.setup_complete = True

    def test_filter_by_parameter_out_of_range(self):
        calibrations = {200.0: {"some_cal_data"}, 300.0: {"more cal data"}}
        with pytest.raises(CalibrationException) as e_info:
            cal = filter_by_parameter(calibrations, 400.0)
            assert (
                e_info.value.args[0]
                == f"Could not locate calibration data at 400.0"
                + f"\nAttempted lookup using key '400.0'"
                + f"\nUsing calibration data: {calibrations}"
            )

    def test_filter_by_parameter_in_range_requires_match(self):
        calibrations = {
            200.0: {"Gain": "Gain at 200.0"},
            300.0: {"Gain": "Gain at 300.0"},
        }
        with pytest.raises(CalibrationException) as e_info:
            cal = filter_by_parameter(calibrations, 150.0)
            assert e_info.value.args[0] == (
                f"Could not locate calibration data at 150.0"
                + f"\nAttempted lookup using key '150.0'"
                + f"\nUsing calibration data: {calibrations}"
            )

    def test_get_calibration_dict_exact_match_lookup(self):
        calibration_datetime = datetime.datetime.now()
        calibration_params = ["sample_rate", "frequency"]
        calibration_data = {
            100.0: {200.0: {"NF": "NF at 100, 200", "Gain": "Gain at 100, 200"}},
            200.0: {100.0: {"NF": "NF at 200, 100", "Gain": "Gain at 200, 100"}},
        }
        clock_rate_lookup_by_sample_rate = {}
        cal = SensorCalibration(
            calibration_datetime,
            calibration_params,
            calibration_data,
            clock_rate_lookup_by_sample_rate,
            False,
            Path(""),
        )
        cal_data = cal.get_calibration_dict([100.0, 200.0])
        assert cal_data["NF"] == "NF at 100, 200"

    def test_get_calibration_dict_within_range(self):
        calibration_datetime = datetime.datetime.now()
        calibration_params = calibration_params = ["sample_rate", "frequency"]
        calibration_data = {
            100.0: {200: {"NF": "NF at 100, 200"}, 300.0: "Cal data at 100,300"},
            200.0: {100.0: {"NF": "NF at 200, 100"}},
        }
        clock_rate_lookup_by_sample_rate = {}
        test_cal_path = Path("test_calibration.json")
        cal = SensorCalibration(
            calibration_datetime,
            calibration_params,
            calibration_data,
            clock_rate_lookup_by_sample_rate,
            False,
            test_cal_path,
        )
        with pytest.raises(CalibrationException) as e_info:
            cal_data = cal.get_calibration_dict([100.0, 250.0])
            assert e_info.value.args[0] == (
                f"Could not locate calibration data at 250.0"
                + f"\nAttempted lookup using key '250.0'"
                + f"\nUsing calibration data: {cal.calibration_data}"
            )

    def test_sf_bound_points(self):
        """Test SF determination at boundary points"""
        self.run_pytest_point(
            self.srs[0], self.frequency_min, self.gain_min, "Testing boundary points"
        )
        self.run_pytest_point(
            self.srs[0], self.frequency_max, self.gain_max, "Testing boundary points"
        )

    def test_sf_no_interpolation_points(self):
        """Test points without interpolation"""
        for i in range(4 * self.test_repeat_times):
            while True:
                g = self.g_s[self.rand_index(self.g_s)]
                f = self.f_s[self.rand_index(self.f_s)]
                if self.run_pytest_point(
                    self.srs[0], f, g, "Testing no interpolation points"
                ):
                    break

    def test_update(self):
        calibration_datetime = get_datetime_str_now()
        calibration_params = ["sample_rate", "frequency"]
        calibration_data = {100.0: {200.0: {"noise_figure": 0, "gain": 0}}}
        clock_rate_lookup_by_sample_rate = {}
        test_cal_path = Path("test_calibration.json")
        cal = SensorCalibration(
            calibration_datetime,
            calibration_params,
            calibration_data,
            clock_rate_lookup_by_sample_rate,
            False,
            test_cal_path,
        )
        action_params = {"sample_rate": 100.0, "frequency": 200.0}
        update_time = get_datetime_str_now()
        cal.update(action_params, update_time, 30.0, 5.0, 21)
        cal_from_file = SensorCalibration.from_json(test_cal_path, False)
        test_cal_path.unlink()
        file_utc_time = parse_datetime_iso_format_str(cal.last_calibration_datetime)
        cal_time_utc = parse_datetime_iso_format_str(update_time)
        assert file_utc_time.year == cal_time_utc.year
        assert file_utc_time.month == cal_time_utc.month
        assert file_utc_time.day == cal_time_utc.day
        assert file_utc_time.hour == cal_time_utc.hour
        assert file_utc_time.minute == cal_time_utc.minute
        assert cal.calibration_data["100.0"]["200.0"]["gain"] == 30.0
        assert cal.calibration_data["100.0"]["200.0"]["noise_figure"] == 5.0
        assert cal_from_file.calibration_data["100.0"]["200.0"]["gain"] == 30.0
        assert cal_from_file.calibration_data["100.0"]["200.0"]["noise_figure"] == 5.0

    def test_filter_by_paramter_integer(self):
        calibrations = {"200.0": {"some_cal_data"}, 300.0: {"more cal data"}}
        filtered_data = filter_by_parameter(calibrations, 200)
        assert filtered_data is calibrations["200.0"]
