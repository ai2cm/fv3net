import xarray as xr
import numpy as np
import pytest

from vcm.safe import _validate_stack_dims, stack_once, warner  # noqa


def test__validate_stack_dims_ok():
    arr_2d = np.ones((10, 10))
    ds = xr.Dataset({"a": (["x", "y"], arr_2d), "b": (["x", "y"], arr_2d)})

    _validate_stack_dims(ds, ["x", "y"])


def test__validate_stack_dims_not_ok():
    arr_2d = np.ones((10, 10))
    arr_3d = np.ones((10, 10, 2))
    ds = xr.Dataset({"a": (["x", "y"], arr_2d), "b": (["x", "y", "z"], arr_3d)})

    with pytest.raises(ValueError):
        # don't allow us to broadcast over the "z" dimension
        _validate_stack_dims(ds, ["x", "y", "z"])

    _validate_stack_dims(ds, ["x", "y", "z"], allowed_broadcast_dims=["z"])


def test_warnings():
    arr_2d = np.ones((10, 10))
    ds = xr.Dataset({"a": (["x", "y"], arr_2d), "b": (["x", "y"], arr_2d)})

    with pytest.warns(UserWarning):
        ds["a"]

    with pytest.warns(UserWarning):
        ds[["a", "b"]]

    with pytest.warns(None) as record:
        with warner.allow():
            ds["a"]
    # ensure no warnings were called
    assert len(record) == 0
