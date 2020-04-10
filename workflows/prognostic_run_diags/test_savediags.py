import save_prognostic_run_diags as savediags
import xarray as xr
import fsspec
from unittest.mock import Mock

import pytest


@pytest.fixture()
def verification():
    return xr.open_dataset("verification.nc").load()


@pytest.fixture()
def resampled():
    return xr.open_dataset("resampled.nc").load()


@pytest.fixture()
def grid():
    return xr.open_dataset("grid.nc").load()


def test_dump_nc(tmpdir):
    ds = xr.Dataset({"a": (["x"], [1.0])})

    path = str(tmpdir.join("data.nc"))
    with fsspec.open(path, "wb") as f:
        savediags.dump_nc(ds, f)

    ds_compare = xr.open_dataset(path)
    xr.testing.assert_equal(ds, ds_compare)


def test_dump_nc_no_seek():
    """
    GCSFS file objects raise an error when seek is called in write mode::

        if not self.mode == "rb":
            raise ValueError("Seek only available in read mode")
            ValueError: Seek only available in read mode

    """
    ds = xr.Dataset({"a": (["x"], [1.0])})
    m = Mock()

    savediags.dump_nc(ds, m)
    m.seek.assert_not_called()


@pytest.mark.parametrize("func", savediags._DIAG_FNS)
def test_compute_diags_succeeds(func, resampled, verification, grid):
    func(resampled, verification, grid)


@pytest.fixture()
def diags(resampled, verification, grid):
    return savediags.compute_all_diagnostics(resampled, verification, grid)
