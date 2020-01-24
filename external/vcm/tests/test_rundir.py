import os

import pytest

from vcm import open_restarts
from vcm.fv3_restarts import _get_tile, _get_time, _is_restart_file, _parse_category

FV_CORE_IN_RESTART = "./RESTART/fv_core.res.tile6.nc"
FV_CORE_IN_RESTART = "./INPUT/fv_core.res.tile6.nc"
FV_CORE_IN_RESTART_WITH_TIMESTEP = "./RESTART/20180605.000000.fv_core.res.tile6.nc"

FINAL = "9999_FINAL"
INIT = "0000_INPUT"


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
        ("RESTART", "20180605.000000.fv_core.res.tile6.nc", "20180605.000000"),
        ("INPUT", "20180605.000000.fv_core.res.tile6.nc", INIT),
        ("INPUT", "fv_core.res.tile6.nc", INIT),
        ("RESTART", "fv_core.res.tile6.nc", FINAL),
    ],
)
def test__get_time(dirname, name, expected):
    time = _get_time(dirname, name)
    assert time == expected


def test_restart_files_at_url():
    url = (
        "gs://vcm-ml-data/2019-10-28-X-SHiELD-2019-10-05-multiresolution-extracted/"
        "one-step-run/C48/20160801.003000/rundir"
    )
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
