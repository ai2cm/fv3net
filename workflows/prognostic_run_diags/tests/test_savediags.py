from itertools import chain
from diagnostics_utils.transform import DiagArg

import fv3net.diagnostics.prognostic_run.compute as savediags
from fv3net.diagnostics.prognostic_run.load_run_data import (
    SegmentedRun,
    CatalogSimulation,
)
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


@pytest.mark.parametrize(
    "func",
    chain.from_iterable(
        registry.funcs.values() for registry in savediags.registries.values()
    ),
)
def test_compute_diags_succeeds(func, resampled, verification, grid):
    diag_arg = DiagArg(resampled, verification, grid)
    func(diag_arg)


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


@pytest.mark.parametrize(
    "url, expected_cls", [("", CatalogSimulation), ("gs://some/run", SegmentedRun)]
)
def test_get_verification_from_catalog(url, expected_cls):
    class Args:
        verification = "hello"
        verification_url = url

    verification = savediags.get_verification(Args, catalog=None)
    assert isinstance(verification, expected_cls), verification
