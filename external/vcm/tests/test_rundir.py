import os

import pytest
import xarray as xr

from vcm import open_restarts
from vcm.fv3_restarts import _get_tile, _get_time, _is_restart_file

FV_CORE_IN_RESTART = "./RESTART/fv_core.res.tile6.nc"
FV_CORE_IN_RESTART = "./INPUT/fv_core.res.tile6.nc"
FV_CORE_IN_RESTART_WITH_TIMESTEP = "./RESTART/20180605.000000.fv_core.res.tile6.nc"

FINAL = "final"
INIT = "init"


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
    assert _get_tile(FV_CORE_IN_RESTART) == 6


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
    time = _get_time(dirname, name, initial_time=INIT, final_time=FINAL)
    assert time == expected


def test_restart_files_at_url():
    url = "gs://vcm-ml-data/2019-10-28-X-SHiELD-2019-10-05-multiresolution-extracted/one-step-run/C48/20160801.003000/rundir"  # noqa
    url = "rundir"
    if not os.path.isdir(url):
        pytest.skip("Data is not available locally.")
    ds = open_restarts(
        url, initial_time="20160801.003000", final_time="20160801.004500"
    )
    print(ds)
    grid = xr.open_mfdataset(
        "rundir/grid_spec.tile?.nc", concat_dim="tile", combine="nested"
    )
    ds = ds.merge(grid)
