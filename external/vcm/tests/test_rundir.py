# import pytest
import xarray as xr

from vcm.misc import rundir

FV_CORE_IN_RESTART = "./RESTART/fv_core.res.tile6.nc"
FV_CORE_IN_RESTART = "./INPUT/fv_core.res.tile6.nc"
FV_CORE_IN_RESTART_WITH_TIMESTEP = "./RESTART/20180605.000000.fv_core.res.tile6.nc"


def test__get_tile():
    assert rundir._get_tile(FV_CORE_IN_RESTART) == 6


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
