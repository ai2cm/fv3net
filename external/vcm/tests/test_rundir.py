import os
import pytest
from datetime import datetime, timedelta

from vcm import open_restarts
from vcm.fv3_restarts import (
    _get_tile,
    _get_file_prefix,
    _is_restart_file,
    _parse_category,
    _get_current_date,
    _get_namelist_path,
    _config_from_fs_namelist,
    _get_current_date_from_coupler_res,
    _get_run_duration,
)

FV_CORE_IN_RESTART = "./RESTART/fv_core.res.tile6.nc"
FV_CORE_IN_RESTART = "./INPUT/fv_core.res.tile6.nc"
FV_CORE_IN_RESTART_WITH_TIMESTEP = "./RESTART/20180605.000000.fv_core.res.tile6.nc"

FINAL = "RESTART/"
INIT = "INPUT/"

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


def test__open_restarts_fails_without_input_and_restart_dirs():
    url = (
        "gs://vcm-ml-data/2020-01-16-X-SHiELD-2019-12-02-pressure-coarsened-rundirs/"
        "restarts/C48/20160801.001500"
    )
    with pytest.raises(ValueError):
        open_restarts(url)


@pytest.mark.parametrize(
    "url, expected",
    [
        (
            OUTPUT_URL,
            (
                "gs",
                (
                    "vcm-ml-data/"
                    "2020-01-16-X-SHiELD-2019-12-02-pressure-coarsened-rundirs/"
                    "one_step_output/C48/20160801.001500/input.nml"
                ),
            ),
        )
    ],
)
def test__get_namelist_path(url, expected):
    assert _get_namelist_path(url) == expected


@pytest.fixture()
def test_config():
    proto, namelist_path = _get_namelist_path(OUTPUT_URL)
    return _config_from_fs_namelist(proto, namelist_path)


def test__get_current_date(test_config):
    expected = datetime(
        **{
            time_unit: value
            for time_unit, value in zip(
                ("year", "month", "day", "hour", "minute", "second"),
                (2016, 8, 1, 0, 15, 0),
            )
        }
    )
    assert _get_current_date(test_config, OUTPUT_URL) == expected


@pytest.mark.parametrize(
    "proto, coupler_res_filename, expected",
    [
        (
            "gs",
            (
                "vcm-ml-data/2019-10-28-X-SHiELD-2019-10-05-multiresolution-extracted/"
                "one_step_output/C48/20160801.003000/INPUT/coupler.res"
            ),
            [2016, 8, 1, 0, 30, 0],
        )
    ],
)
def test__get_current_date_from_coupler_res(proto, coupler_res_filename, expected):
    assert _get_current_date_from_coupler_res(proto, coupler_res_filename) == expected


def test__get_run_duration(test_config):
    expected = timedelta(seconds=900)
    assert _get_run_duration(test_config) == expected
