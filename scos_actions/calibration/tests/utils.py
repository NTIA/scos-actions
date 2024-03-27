"""Utility functions used for scos_sensor.calibration unit tests."""


def recursive_check_keys(d: dict):
    """Recursively checks a dict to check that all keys are strings"""
    for k, v in d.items():
        if isinstance(v, dict):
            recursive_check_keys(v)
        else:
            assert isinstance(k, str)
