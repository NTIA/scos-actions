"""Test the SensorCalibration dataclass."""

import dataclasses
import datetime
import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import pytest

from scos_actions.calibration.interfaces.calibration import Calibration
from scos_actions.calibration.sensor_calibration import (
    SensorCalibration,
    has_expired_cal_data,
)
from scos_actions.calibration.tests.utils import recursive_check_keys
from scos_actions.calibration.utils import CalibrationException
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

    @pytest.fixture(autouse=True)
    def setup_calibration_file(self, tmp_path: Path):
        """
        Create the dummy calibration file in the pytest temp directory

        The gain values in each calibration data entry are set up as being
        equal to the gain setting minus ``self.dummy_gain_scale_factor``
        """

        # Only setup once
        if self.setup_complete:
            return

        # Create and save the temp directory and file
        self.calibration_file = tmp_path / "dummy_cal_file.json"

        # Setup variables
        self.dummy_gain_scale_factor = 5  # test data gain values are (gain setting - 5)
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
        cal_data["calibration_reference"] = "TESTING"

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
                    # Create the data point
                    cal_data_point = {
                        "gain": gains[j] - self.dummy_gain_scale_factor,
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
        self.sample_cal = SensorCalibration.from_json(self.calibration_file)

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

    def test_calibration_data_key_name_conversion(self):
        """On post-init, all calibration_data key names should be converted to strings."""
        recursive_check_keys(self.sample_cal.calibration_data)

    def test_sensor_calibration_dataclass_fields(self):
        """Check that the dataclass fields are as expected."""
        fields = {
            f.name: f.type
            for f in dataclasses.fields(SensorCalibration)
            if f not in dataclasses.fields(Calibration)
        }
        # Note: does not check field order
        assert fields == {
            "last_calibration_datetime": str,
            "clock_rate_lookup_by_sample_rate": List[Dict[str, float]],
            "sensor_uid": str,
        }

    def test_field_validator(self):
        """Check that the input field type validator works."""
        # only check fields not inherited from Calibration base class
        with pytest.raises(TypeError):
            _ = SensorCalibration([], {}, False, Path(""), "dt_str", [], 10)
        with pytest.raises(TypeError):
            _ = SensorCalibration([], {}, False, Path(""), "dt_str", {}, "uid")
        with pytest.raises(TypeError):
            _ = SensorCalibration(
                [], {}, False, Path(""), datetime.datetime.now(), [], "uid"
            )

    def test_get_clock_rate(self):
        """Test the get_clock_rate method"""
        # Test getting a clock rate by sample rate
        assert self.sample_cal.get_clock_rate(10e6) == 40e6
        # If there isn't an entry, the sample rate should be returned
        assert self.sample_cal.get_clock_rate(-999) == -999

    def test_get_calibration_dict_exact_match_lookup(self):
        calibration_datetime = get_datetime_str_now()
        calibration_params = ["sample_rate", "frequency"]
        calibration_data = {
            100.0: {200.0: {"NF": "NF at 100, 200", "Gain": "Gain at 100, 200"}},
            200.0: {100.0: {"NF": "NF at 200, 100", "Gain": "Gain at 200, 100"}},
        }
        cal = SensorCalibration(
            calibration_parameters=calibration_params,
            calibration_data=calibration_data,
            calibration_reference="testing",
            file_path=Path(""),
            last_calibration_datetime=calibration_datetime,
            clock_rate_lookup_by_sample_rate=[],
            sensor_uid="TESTING",
        )
        cal_data = cal.get_calibration_dict({"sample_rate": 100.0, "frequency": 200.0})
        assert cal_data["NF"] == "NF at 100, 200"

    def test_get_calibration_dict_within_range(self):
        calibration_datetime = get_datetime_str_now()
        calibration_params = calibration_params = ["sample_rate", "frequency"]
        calibration_data = {
            100.0: {200: {"NF": "NF at 100, 200"}, 300.0: "Cal data at 100,300"},
            200.0: {100.0: {"NF": "NF at 200, 100"}},
        }
        cal = SensorCalibration(
            calibration_parameters=calibration_params,
            calibration_data=calibration_data,
            calibration_reference="testing",
            file_path=Path("test_calibration.json"),
            last_calibration_datetime=calibration_datetime,
            clock_rate_lookup_by_sample_rate=[],
            sensor_uid="TESTING",
        )

        lookup_fail_value = 250.0
        with pytest.raises(CalibrationException) as e_info:
            _ = cal.get_calibration_dict(
                {"sample_rate": 100.0, "frequency": lookup_fail_value}
            )
        assert e_info.value.args[0] == (
            f"Could not locate calibration data at {lookup_fail_value}"
            + "\nAttempted lookup using keys: "
            + f"\n\tstr({lookup_fail_value}).lower() = {str(lookup_fail_value).lower()}"
            + f"\n\tstr(int({lookup_fail_value})) = {int(lookup_fail_value)}"
            + f"\nUsing calibration data: {cal.calibration_data['100.0']}"
        )

    def test_update(self):
        calibration_datetime = "2024-03-17T19:16:55.172Z"
        calibration_params = ["sample_rate", "frequency"]
        calibration_data = {100.0: {200.0: {"noise_figure": 0, "gain": 0}}}
        test_cal_path = Path("test_calibration.json")
        cal = SensorCalibration(
            calibration_parameters=calibration_params,
            calibration_data=calibration_data,
            calibration_reference="testing",
            file_path=test_cal_path,
            last_calibration_datetime=calibration_datetime,
            clock_rate_lookup_by_sample_rate=[],
            sensor_uid="TESTING",
        )
        action_params = {"sample_rate": 100.0, "frequency": 200.0}
        update_time = get_datetime_str_now()
        cal.update(action_params, update_time, 30.0, 5.0, 21)
        cal_from_file = SensorCalibration.from_json(test_cal_path)
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

    def test_has_expired_cal_data_not_expired(self):
        cal_date = "2024-03-14T15:48:38.039Z"
        now_date = "2024-03-14T15:49:38.039Z"
        cal_data = {
            "3550": {
                "20.0": {
                    "false": {"datetime": cal_date},
                },
            }
        }
        expired = has_expired_cal_data(
            cal_data, parse_datetime_iso_format_str(now_date), 100
        )
        assert expired == False

    def test_has_expired_cal_data_expired(self):
        cal_date = "2024-03-14T15:48:38.039Z"
        now_date = "2024-03-14T15:49:38.039Z"
        cal_data = {
            "3550": {
                "20.0": {
                    "false": {"datetime": cal_date},
                },
            }
        }
        expired = has_expired_cal_data(
            cal_data, parse_datetime_iso_format_str(now_date), 30
        )
        assert expired == True

    def test_has_expired_cal_data_multipledates_expired(self):
        cal_date_1 = "2024-03-14T15:48:38.039Z"
        cal_date_2 = "2024-03-14T15:40:38.039Z"
        now_date = "2024-03-14T15:49:38.039Z"
        cal_data = {
            "3550": {
                "20.0": {
                    "false": {"datetime": cal_date_1},
                },
                "true": {"datetime": cal_date_2},
            }
        }
        expired = has_expired_cal_data(
            cal_data, parse_datetime_iso_format_str(now_date), 100
        )
        assert expired == True
        cal_data = {
            "3550": {
                "20.0": {
                    "false": {"datetime": cal_date_2},
                },
                "true": {"datetime": cal_date_1},
            }
        }
        expired = has_expired_cal_data(
            cal_data, parse_datetime_iso_format_str(now_date), 100
        )
        assert expired == True
