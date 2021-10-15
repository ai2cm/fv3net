import contextlib
import datetime
import os
import pytest
import numpy as np
from xarray import DataArray
from typing import Mapping

from fv3gfs.util import Quantity

from emulation._monitor.monitor import (
    _bool_from_str,
    _load_nml,
    _get_timestep,
    _remove_io_suffix,
    _get_attrs,
    _convert_to_quantities,
    _convert_to_xr_dataset,
    _translate_time,
    _create_nc_path,
)


@contextlib.contextmanager
def set_current_directory(path):

    origin = os.getcwd()

    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(origin)


def save_input_nml(path):

    input_nml_content = """
    &coupler_nml
        calendar = 'julian'
        current_date = 2016, 8, 1, 0, 0, 0
        dt_atmos = 900
        days = 0
        hours = 1
        minutes = 0
        months = 0
        seconds = 0
    /

    &fv_core_nml
        layout = 1,1
    /
    """

    with open(path, "w") as f:
        f.write(input_nml_content)


@pytest.fixture
def dummy_rundir(tmp_path):

    rundir = tmp_path
    with set_current_directory(rundir):
        save_input_nml(rundir / "input.nml")

        yield rundir


@pytest.mark.parametrize(
    "value, expected",
    [("y", True), ("yes", True), ("true", True),
     ("n", False), ("no", False), ("false", False)]
)
def test__bool_from_str(value, expected):

    assert _bool_from_str(value) == expected
    assert _bool_from_str(value.capitalize()) == expected
    assert _bool_from_str(value.upper()) == expected


def test__bool_from_str_unrecognized():

    with pytest.raises(ValueError):
        _bool_from_str("not-translatable-to-bool")


def test__load_nml(dummy_rundir):

    namelist = _load_nml()
    assert namelist["coupler_nml"]["hours"] == 1


def test__get_timestep(dummy_rundir):
    namelist = _load_nml()
    timestep = _get_timestep(namelist)

    assert timestep == 900


@pytest.mark.parametrize(
    "value,expected",
    [
        ("a_input", "a"),
        ("a_output", "a"),
        ("a_input_a", "a_input_a"),
        ("a_output_output", "a_output"),
    ],
    ids=["input-remove", "no-change", "output-remove", "double-match-no-change"]
)
def test__remove_io_suffix(value, expected):

    result = _remove_io_suffix(value)
    assert result == expected


def test__get_attrs():

    metadata = dict(
        number={"a_number": 1, "a_list": [1, 2, 3]}
    )

    dat = _get_attrs("number", metadata)
    for k, v in dat.items():
        assert isinstance(v, str)

    dat = _get_attrs("nonexistent", metadata)
    assert not dat
    assert isinstance(dat, Mapping)


def get_state_meta_for_conversion():

    metadata = dict(
        field_A={"units": "beepboop", "real-name": "missingno"},
        field_B={"units": "goobgob", "real-name": "trivial"}
    )
    
    state = {
        "field_A": np.ones(50).reshape(2, 25),
        "field_B": np.ones(50).reshape(2, 25)
    }

    return state, metadata


@pytest.mark.parametrize(
    "convert_func, expected_type",
    [
        (_convert_to_quantities, Quantity),
        (_convert_to_xr_dataset, DataArray),
    ],
    ids=["to-quantity", "to-xr-dataset"]
)
def test__conversions(convert_func, expected_type):

    state, meta = get_state_meta_for_conversion()
    quantities = convert_func(state, meta)

    for data in quantities.values():
        assert isinstance(data, expected_type)
        assert data.values.shape == (25, 2)


def test__translate_time():

    year = 2016
    month = 10
    day = 29

    time = _translate_time([year, month, day, None, 0, 0])

    assert time.year == year
    assert time.month == month
    assert time.day == day


def test__create_nc_path(dummy_rundir):

    _create_nc_path()
    assert os.path.exists(dummy_rundir / "netcdf_output")

