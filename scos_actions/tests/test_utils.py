import pytest
from scos_actions import utils
import datetime
from dateutil import tz
import numpy as np


@pytest.fixture
def valid_params_no_lists():
    return {"name": "valid_params_no_list", "sample_rate": 14e6, "frequency": 770e6}

def test_get_datetime_str_now():
    datetime_str = utils.get_datetime_str_now()
    assert len(datetime_str) == 24
    assert all(datetime_str[i] == "-" for i in [4, 7])
    assert datetime_str[10] == "T"
    assert all(datetime_str[i] == ":" for i in [13, 16])
    assert datetime_str[19] == "."
    assert datetime_str[-1] == "Z"

def test_parse_datetime_iso_format_str():
    tstamp = utils.get_datetime_str_now()
    parsed = utils.parse_datetime_iso_format_str(tstamp)
    assert type(parsed) is datetime.datetime
    # 2022-08-22T16:22:11.833Z
    assert type(parsed.year) is int
    assert parsed.year == int(tstamp[:4])
    assert type(parsed.month) is int
    assert parsed.month == int(tstamp[5:7])
    assert type(parsed.day) is int
    assert parsed.day == int(tstamp[8:10])
    assert type(parsed.hour) is int
    assert parsed.hour == int(tstamp[11:13])
    assert type(parsed.minute) is int
    assert parsed.minute == int(tstamp[14:16])
    assert type(parsed.second) is int
    assert parsed.second == int(tstamp[17:19])
    assert type(parsed.microsecond) is int
    assert parsed.microsecond == int(tstamp[20:23] + "000")
    assert type(parsed.tzinfo) is tz.tz.tzutc

def test_get_iterable_parameters_no_lists(valid_params_no_lists):
    i_params = utils.get_iterable_parameters(valid_params_no_lists)
    assert type(i_params) is list
    assert len(i_params) == 1
    with pytest.raises(KeyError):
        _ = utils.get_iterable_parameters(valid_params_no_lists, "gain")
    assert not any("name" in x.keys() for x in i_params)
    valid_params_no_lists.pop("name")
    assert all(i_params[0][k] == valid_params_no_lists[k] for k in valid_params_no_lists.keys())

def test_get_iterable_parameters_all_lists():
    all_lists = {
        "name": "params_all_lists",
        "sample_rate": [14e6, 28e6, 56e6],
        "frequency": [720e6, 710e6, 700e6]
    }
    i_params = utils.get_iterable_parameters(all_lists)
    assert type(i_params) is list
    assert len(i_params) == 3
    assert not any("name" in x.keys() for x in i_params)
    minf = min(all_lists["frequency"])
    maxf = max(all_lists["frequency"])
    assert i_params[0]["frequency"] == minf
    assert i_params[0]["sample_rate"] == all_lists["sample_rate"][all_lists["frequency"].index(minf)]
    assert i_params[-1]["frequency"] == maxf
    assert i_params[-1]["sample_rate"] == all_lists["sample_rate"][all_lists["frequency"].index(maxf)]

def test_get_iterable_parameters_some_lists():
    some_lists = {
        "name": "some_lists",
        "sample_rate": 14e6,
        "frequency": [720e6, 705e6, 1000e6, 10],
        "gain": [1, 2, 3, 4]
    }
    i_params = utils.get_iterable_parameters(some_lists)
    assert type(i_params) is list
    assert len(i_params) == 4
    assert not any("name" in x.keys() for x in i_params)
    minf = min(some_lists["frequency"])
    maxf = max(some_lists["frequency"])
    aminf = some_lists["frequency"].index(minf)
    amaxf = some_lists["frequency"].index(maxf)
    assert i_params[0]["frequency"] == minf
    assert i_params[0]["sample_rate"] == some_lists["sample_rate"]
    assert i_params[0]["gain"] == some_lists["gain"][aminf]
    assert i_params[-1]["frequency"] == maxf
    assert i_params[-1]["sample_rate"] == some_lists["sample_rate"]
    assert i_params[-1]["gain"] == some_lists["gain"][amaxf]

def test_get_iterable_parameters_incompatible_lists():
    incompat_lists = {
        "name": "incompatible_lists",
        "sample_rate": [14e6, 28e6, 56e6],
        "frequency": 700e6,
        "gain": [1, 2]
    }
    with pytest.raises(utils.ParameterException):
        _ = utils.get_iterable_parameters(incompat_lists)

def test_list_to_string():
    ex_list = [1, 2.0, "testing"]
    test_str = utils.list_to_string(ex_list)
    assert type(test_str) is str
    assert test_str == "1,2.0,testing"

def test_get_parameter(valid_params_no_lists):
    valid_key = "sample_rate"
    invalid_key = "attenuation"
    assert utils.get_parameter(valid_key, valid_params_no_lists) == 14e6
    with pytest.raises(utils.ParameterException):
        utils.get_parameter(invalid_key, valid_params_no_lists)
