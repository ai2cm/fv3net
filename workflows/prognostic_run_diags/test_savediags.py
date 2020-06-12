import os
import save_prognostic_run_diags as savediags
import xarray as xr
import fsspec
from unittest.mock import Mock

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


def test__catalog():
    assert os.path.isfile(savediags._catalog())


@pytest.mark.parametrize(
    "ds, variables, expected",
    [
        (xr.Dataset({"foo": 1}), {"foo"}, True),
        (xr.Dataset({"foo": 1}), {"foo", "bar"}, False),
        (xr.Dataset({"foo": 1}), {"bar"}, False),
        (xr.Dataset({"foo": 1, "bar": 2}), {"foo"}, True),
    ],
)
def test__missing_var(ds, variables, expected):
    assert savediags._ds_contains(ds, variables) == expected
