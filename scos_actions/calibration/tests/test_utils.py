import pytest

from scos_actions.calibration.utils import CalibrationException, filter_by_parameter


class TestCalibrationUtils:
    def test_filter_by_parameter_out_of_range(self):
        calibrations = {200.0: {"some_cal_data"}, 300.0: {"more cal data"}}

        # Also checks error output when missing value is an integer
        test_value = 400
        with pytest.raises(CalibrationException) as e_info:
            _ = filter_by_parameter(calibrations, test_value)
        assert (
            e_info.value.args[0]
            == f"Could not locate calibration data at {test_value}"
            + "\nAttempted lookup using keys: "
            + f"\n\tstr({test_value}).lower() = {str(test_value).lower()}"
            + f"\n\tstr(float({test_value})) = {float(test_value)}"
            + f"\nUsing calibration data: {calibrations}"
        )

    def test_filter_by_parameter_in_range_requires_match(self):
        calibrations = {
            200.0: {"Gain": "Gain at 200.0"},
            300.0: {"Gain": "Gain at 300.0"},
        }

        # Check looking up a missing value with a float
        test_value = 150.0
        with pytest.raises(CalibrationException) as e_info:
            _ = filter_by_parameter(calibrations, test_value)
        assert e_info.value.args[0] == (
            f"Could not locate calibration data at {test_value}"
            + "\nAttempted lookup using keys: "
            + f"\n\tstr({test_value}).lower() = {str(test_value).lower()}"
            + f"\n\tstr(int({test_value})) = {int(test_value)}"
            + f"\nUsing calibration data: {calibrations}"
        )

    def test_filter_by_paramter_integer(self):
        calibrations = {"200.0": {"some_cal_data"}, 300.0: {"more cal data"}}
        filtered_data = filter_by_parameter(calibrations, 200)
        assert filtered_data is calibrations["200.0"]

    def test_filter_by_parameter_type_error(self):
        calibrations = [300.0, 400.0]
        with pytest.raises(CalibrationException) as e_info:
            _ = filter_by_parameter(calibrations, 300.0)
        assert e_info.value.args[0] == (
            f"Provided calibration data is not a dict: {calibrations}"
        )
