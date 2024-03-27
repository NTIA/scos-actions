"""Test the DifferentialCalibration dataclass."""

import json
from pathlib import Path

import pytest

from scos_actions.calibration.differential_calibration import DifferentialCalibration


class TestDifferentialCalibration:
    @pytest.fixture(autouse=True)
    def setup_differential_calibration_file(self, tmp_path: Path):
        dict_to_json = {
            "calibration_parameters": ["frequency"],
            "calibration_reference": "antenna input",
            "calibration_data": {3555e6: 11.5},
        }
        self.valid_file_path = tmp_path / "sample_diff_cal.json"
        self.invalid_file_path = tmp_path / "sample_diff_cal_invalid.json"

        self.sample_diff_cal = DifferentialCalibration(
            file_path=self.valid_file_path, **dict_to_json
        )

        with open(self.valid_file_path, "w") as f:
            f.write(json.dumps(dict_to_json))

        dict_to_json.pop("calibration_reference", None)

        with open(self.invalid_file_path, "w") as f:
            f.write(json.dumps(dict_to_json))

    def test_from_json(self):
        """Check from_json functionality with valid and invalid dummy data."""
        diff_cal = DifferentialCalibration.from_json(self.valid_file_path)
        assert diff_cal == self.sample_diff_cal
        with pytest.raises(Exception):
            _ = DifferentialCalibration.from_json(self.invalid_file_path)

    def test_update_not_implemented(self):
        """Check that the update method is not implemented."""
        with pytest.raises(NotImplementedError):
            self.sample_diff_cal.update()
