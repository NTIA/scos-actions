"""Test the Calibration base dataclass."""

import dataclasses
import json
from pathlib import Path
from typing import List

import pytest

from scos_actions.calibration.interfaces.calibration import Calibration
from scos_actions.calibration.sensor_calibration import SensorCalibration
from scos_actions.calibration.tests.utils import recursive_check_keys


class TestBaseCalibration:
    @pytest.fixture(autouse=True)
    def setup_calibration_file(self, tmp_path: Path):
        """Create a dummy calibration file in the pytest temp directory."""
        # Create some dummy calibration data
        self.cal_params = ["frequency", "gain"]
        self.frequencies = [3555e6, 3565e6, 3575e6]
        self.gains = [10.0, 20.0, 30.0]
        cal_data = {}
        for frequency in self.frequencies:
            cal_data[frequency] = {}
            for gain in self.gains:
                cal_data[frequency][gain] = {
                    "gain": gain * 1.1,
                    "noise_figure": gain / 5.0,
                    "1dB_compression_point": -50 + gain,
                }
        self.cal_data = cal_data
        self.dummy_file_path = tmp_path / "dummy_cal.json"
        self.dummy_default_file_path = tmp_path / "dummy_default_cal.json"

        self.sample_cal = Calibration(
            calibration_parameters=self.cal_params,
            calibration_data=self.cal_data,
            calibration_reference="testing",
            file_path=self.dummy_file_path,
        )

        self.sample_default_cal = Calibration(
            calibration_parameters=self.cal_params,
            calibration_data=self.cal_data,
            calibration_reference="testing",
            file_path=self.dummy_default_file_path,
        )

    def test_calibration_data_key_name_conversion(self):
        """On post-init, all calibration_data key names should be converted to strings."""
        recursive_check_keys(self.sample_cal.calibration_data)
        recursive_check_keys(self.sample_default_cal.calibration_data)

    def test_calibration_dataclass_fields(self):
        """Check that the dataclass is set up as expected."""
        fields = {f.name: f.type for f in dataclasses.fields(Calibration)}
        # Note: does not check field order
        assert fields == {
            "calibration_parameters": List[str],
            "calibration_reference": str,
            "calibration_data": dict,
            "file_path": Path,
        }, "Calibration class fields have changed"

    def test_field_validator(self):
        """Check that the input field type validator works."""
        with pytest.raises(TypeError):
            _ = Calibration([], {}, "", False, False)
        with pytest.raises(TypeError):
            _ = Calibration([], {}, "", 100, Path(""))
        with pytest.raises(TypeError):
            _ = Calibration([], {}, 5, False, Path(""))
        with pytest.raises(TypeError):
            _ = Calibration([], [10, 20], "", False, Path(""))
        with pytest.raises(TypeError):
            _ = Calibration({"test": 1}, {}, "", False, Path(""))

    def test_get_calibration_dict(self):
        """Check the get_calibration_dict method with all dummy data."""
        for f in self.frequencies:
            for g in self.gains:
                assert json.loads(
                    json.dumps(self.cal_data[f][g])
                ) == self.sample_cal.get_calibration_dict({"frequency": f, "gain": g})

    def test_to_and_from_json(self, tmp_path: Path):
        """Test the ``from_json`` factory method."""
        # First save the calibration data to temporary files
        self.sample_cal.to_json()
        self.sample_default_cal.to_json()
        # Then load and compare
        assert self.sample_cal == Calibration.from_json(self.dummy_file_path, False)
        assert self.sample_default_cal == Calibration.from_json(
            self.dummy_default_file_path, True
        )

        # from_json should ignore extra keys in the loaded file, but not fail
        # Test this by trying to load a SensorCalibration as a Calibration
        sensor_cal = SensorCalibration(
            self.sample_cal.calibration_parameters,
            self.sample_cal.calibration_data,
            "testing",
            False,
            tmp_path / "testing.json",
            "dt_str",
            [],
            "uid",
        )
        sensor_cal.to_json()
        loaded_cal = Calibration.from_json(tmp_path / "testing.json", False)
        loaded_cal.file_path = self.sample_cal.file_path  # Force these to be the same
        assert loaded_cal == self.sample_cal

        # from_json should fail if required fields are missing
        # Create an incorrect JSON file
        almost_a_cal = {"calibration_parameters": []}
        with open(tmp_path / "almost_a_cal.json", "w") as outfile:
            outfile.write(json.dumps(almost_a_cal))
        with pytest.raises(Exception):
            almost = Calibration.from_json(tmp_path / "almost_a_cal.json", False)

    def test_update_not_implemented(self):
        """Ensure the update abstract method is not implemented in the base class"""
        with pytest.raises(NotImplementedError):
            self.sample_cal.update()
