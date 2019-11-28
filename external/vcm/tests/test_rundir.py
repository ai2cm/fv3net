import pytest
import xarray as xr

from vcm.misc import rundir

FV_CORE_IN_RESTART = "./RESTART/fv_core.res.tile6.nc"
FV_CORE_IN_RESTART = "./INPUT/fv_core.res.tile6.nc"
FV_CORE_IN_RESTART_WITH_TIMESTEP = "./RESTART/20180605.000000.fv_core.res.tile6.nc"

FINAL = 'final'
INIT = 'init'


def test__get_tile():
    assert rundir._get_tile(FV_CORE_IN_RESTART) == 6


@pytest.mark.parametrize('dirname, name, expected', [
    ( 'RESTART', "20180605.000000.fv_core.res.tile6.nc", '20180605.000000'),
    ( 'INPUT', "20180605.000000.fv_core.res.tile6.nc", INIT),
    ( 'INPUT', "fv_core.res.tile6.nc", INIT),
    ( 'RESTART', "fv_core.res.tile6.nc", FINAL),
])

def test__get_time(dirname, name, expected):
    time = rundir._get_time(dirname, name, initial_time=INIT,final_time=FINAL)
    assert time == expected


def test__nml_to_grid():
    nml = {'fv_core'}


@pytest.mark.skip()
def test_restart_files_at_url():
    url = "gs://vcm-ml-data/2019-10-28-X-SHiELD-2019-10-05-multiresolution-extracted/one-step-run/C48/20160801.003000/rundir"  # noqa
    url = "20160801.003000/rundir"
    ds = rundir.open_restarts(
        url, initial_time="20160801.003000", final_time="20160801.004500"
    )
    grid = xr.open_mfdataset(
        "20160801.003000/rundir/grid_spec.tile?.nc", concat_dim="tile", combine="nested"
    )
    ds = ds.merge(grid)
    print(ds)


def test__concatenate_by_key():
    arr = xr.DataArray([0.0], dims=['x'])
    arrays = {('a', 1): arr, ('a', 2): arr}

    ds = rundir._concatenate_by_key(arrays, ['letter', 'number'])

    assert isinstance(ds, xr.DataArray)
    assert ds.dims == ['letter', 'number', 'x']
