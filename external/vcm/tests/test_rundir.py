import os
import pytest
from datetime import datetime, timedelta
import fsspec

from vcm import open_restarts

from vcm._rundir import (
    _get_tile,
    _get_file_prefix,
    _is_restart_file,
    _parse_category,
    _get_current_date,
    _get_namelist_path,
    _config_from_fs_namelist,
    _get_current_date_from_coupler_res,
    _get_run_duration,
    _get_prefixes,
)

FV_CORE_IN_RESTART = "./RESTART/fv_core.res.tile6.nc"
FV_CORE_IN_RESTART = "./INPUT/fv_core.res.tile6.nc"
FV_CORE_IN_RESTART_WITH_TIMESTEP = "./RESTART/20180605.000000.fv_core.res.tile6.nc"

FINAL = "RESTART"
INIT = "INPUT"

OUTPUT_URL = (
    "gs://vcm-ml-data/2020-01-16-X-SHiELD-2019-12-02-pressure-coarsened-rundirs/"
    "one_step_output/C48/20160801.001500"
)


@pytest.mark.parametrize(
    "path, is_restart",
    [
        ("INPUT/fv_core.res.tile6.nc", True),
        ("atmos_dt_atmos.nc", False),
        ("RESTART/fv_core.res.tile6.nc", True),
        ("INPUT/fv_core.res.txt", False),
    ],
)
def test__is_restart_file(path, is_restart):
    assert _is_restart_file(path) == is_restart


def test__get_tile():
    assert _get_tile(FV_CORE_IN_RESTART) == 5


@pytest.mark.parametrize(
    "dirname, name, expected",
    [
        ("RESTART", "20180605.000000.fv_core.res.tile6.nc", "RESTART/20180605.000000"),
        ("INPUT", "20180605.000000.fv_core.res.tile6.nc", INIT),
        ("INPUT", "fv_core.res.tile6.nc", INIT),
        ("RESTART", "fv_core.res.tile6.nc", FINAL),
    ],
)
def test__get_file_prefix(dirname, name, expected):
    time = _get_file_prefix(dirname, name)
    assert time == expected


def test_restart_files_at_url():
    url = "rundir"
    if not os.path.isdir(url):
        pytest.skip("Data is not available locally.")
    ds = open_restarts(url)
    print(ds)


def test__parse_category_succeeds():
    file = "fv_core.res.tile1.nc"
    assert _parse_category(file) == "fv_core.res"


def test__parse_category_fails_with_ambiguous_category():
    file = "sfc_data.fv_core.res.tile1.nc"
    with pytest.raises(ValueError):
        _parse_category(file)


@pytest.mark.parametrize(
    "url, expected",
    [
        (
            OUTPUT_URL,
            "vcm-ml-data/"
            "2020-01-16-X-SHiELD-2019-12-02-pressure-coarsened-rundirs/"
            "one_step_output/C48/20160801.001500/input.nml",
        )
    ],
)
def test__get_namelist_path(fs, url, expected):
    assert _get_namelist_path(fs, url) == expected


@pytest.fixture()
def fs():
    return fsspec.filesystem("gs")


@pytest.fixture()
def test_config(fs):
    namelist_path = _get_namelist_path(fs, OUTPUT_URL)
    return _config_from_fs_namelist(fs, namelist_path)


def test__get_current_date(fs, test_config):
    expected = datetime(
        **{
            time_unit: value
            for time_unit, value in zip(
                ("year", "month", "day", "hour", "minute", "second"),
                (2016, 8, 1, 0, 15, 0),
            )
        }
    )
    assert _get_current_date(test_config, fs, OUTPUT_URL) == expected


@pytest.mark.parametrize(
    "coupler_res_filename, expected",
    [
        (
            (
                "vcm-ml-data/2019-10-28-X-SHiELD-2019-10-05-multiresolution-extracted/"
                "one_step_output/C48/20160801.003000/INPUT/coupler.res"
            ),
            [2016, 8, 1, 0, 30, 0],
        )
    ],
)
def test__get_current_date_from_coupler_res(fs, coupler_res_filename, expected):
    assert _get_current_date_from_coupler_res(fs, coupler_res_filename) == expected


def test__get_run_duration(test_config):
    expected = timedelta(seconds=900)
    assert _get_run_duration(test_config) == expected


def test_get_prefixes():
    walker = [
        ("blah/RESTART", [], ["20160101.000015.fv_core.res.tile6.nc"]),
        ("blah/RESTART", [], ["20160101.000000.fv_core.res.tile6.nc"]),
        ("blah/INPUT", [], ["fv_core.res.tile6.nc"]),
        ("blah/RESTART", [], ["fv_core.res.tile6.nc"]),
    ]

    expected = [
        "INPUT",
        "RESTART/20160101.000000",
        "RESTART/20160101.000015",
        "RESTART",
    ]
    assert _get_prefixes(walker) == expected
