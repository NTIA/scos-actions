import tempfile

import pytest
from ruamel.yaml.scanner import ScannerError


# Indentation makes this invalid
from scos_actions import actions
from scos_actions.discover.yaml import load_from_yaml

INVALID_YAML = b"""\
single_frequency_fft:
    name: acquire_700c_dl
    frequency: 751e6
    gain: 40
        sample_rate: 15.36e6
        fft_size: 1024
        nffts: 300
"""

NONEXISTENT_ACTION_CLASS_NAME = b"""\
this_doesnt_exist:
    name: test_expected_failure
"""

# "frequency" is misspelled
INVALID_PARAMETERS = b"""\
single_frequency_fft:
    name: acquire_700c_dl
    frequnecy: 751e6
    gain: 40
    sample_rate: 15.36e6
    fft_size: 1024
    nffts: 300
"""


def test_load_from_yaml_existing():
    """Any existing action definitions should be valid yaml."""
    load_from_yaml(actions.action_classes)


def _test_load_from_yaml_check_error(yaml_to_write, expected_error):
    # load_from_yaml loads all `.yml` files in the passed directory, so do a
    # bit of setup to create an invalid yaml tempfile in a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        with tempfile.NamedTemporaryFile(suffix=".yml", dir=tmpdir) as tmpfile:
            tmpfile.write(yaml_to_write)
            tmpfile.seek(0)
            # Now try to load the invalid yaml file, expecting an error
            with pytest.raises(expected_error):
                load_from_yaml(actions.action_classes, yaml_dir=tmpdir)


def test_load_from_yaml_parse_error():
    """An invalid yaml file should cause a parse error."""
    _test_load_from_yaml_check_error(INVALID_YAML, ScannerError)


def test_load_from_yaml_invalid_class_name():
    """A nonexistent action class name should raise an error."""
    _test_load_from_yaml_check_error(NONEXISTENT_ACTION_CLASS_NAME, KeyError)


def test_load_from_yaml_invalid_parameters():
    """A nonexistent action class name should raise an error."""
    _test_load_from_yaml_check_error(INVALID_PARAMETERS, TypeError)
