from vcm import _schema
import numpy as np
import xarray as xr

import pytest


@pytest.fixture()
def da():
    return xr.DataArray(np.ones((4, 5)), dims=["y", "bad"], coords={"y": np.arange(4)})


@pytest.mark.parametrize("container_type", [list, tuple])
def test_Schema_rename_any_dims(da, container_type):
    meta = _schema.Schema(dims=container_type([..., "x"]), dtype=da.dtype, attrs={})
    meta._rename_dimensions(da)


def test_Schema_rename_dimensions_correct(da):
    meta = _schema.Schema(dims=[..., "x"], dtype=da.dtype, attrs={})
    out = meta._rename_dimensions(da)
    xr.testing.assert_allclose(out, da.rename({"bad": "x"}))


def test_Schema_rename_dimensions_fails_without_ellipsis(da):
    meta = _schema.Schema(dims=["x"], dtype=da.dtype, attrs={})
    with pytest.raises(ValueError):
        meta._rename_dimensions(da)


@pytest.mark.parametrize(
    "dims, succeeds", [(["x"], False), (["bad"], False), ([..., "bad"], True)]
)
def test_Schema_validate_fails(da, dims, succeeds):
    meta = _schema.Schema(dims=dims, dtype=da.dtype, attrs={})
    assert meta.validate(da) == succeeds
