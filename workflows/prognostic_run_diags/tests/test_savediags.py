import fv3net.diagnostics.prognostic_run.compute as savediags
import cftime
import numpy as np
import xarray as xr

import pytest


@pytest.fixture()
def verification():
    pytest.skip()
    # TODO replace these fixtures with synthetic data generation
    return xr.open_dataset("verification.nc").load()


@pytest.fixture()
def resampled():
    pytest.skip()
    # TODO replace these fixtures with synthetic data generation
    return xr.open_dataset("resampled.nc").load()


@pytest.fixture()
def grid():
    pytest.skip()
    # TODO replace these fixtures with synthetic data generation
    return xr.open_dataset("grid.nc").load()


@pytest.mark.parametrize("func", savediags._DIAG_FNS)
def test_compute_diags_succeeds(func, resampled, verification, grid):
    func(resampled, verification, grid)


def test_time_mean():
    ntimes = 5
    time_coord = [cftime.DatetimeJulian(2016, 4, 2, i + 1) for i in range(ntimes)]
    ds = xr.Dataset(
        data_vars={"temperature": (["time", "x"], np.zeros((ntimes, 10)))},
        coords={"time": time_coord},
    )
    diagnostic = savediags.time_mean(ds)
    assert diagnostic.temperature.attrs["diagnostic_start_time"] == str(time_coord[0])
    assert diagnostic.temperature.attrs["diagnostic_end_time"] == str(time_coord[-1])
