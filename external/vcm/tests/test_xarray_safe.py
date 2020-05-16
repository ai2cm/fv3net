import xarray as xr
import numpy as np
import pytest

from vcm.safe import (
    _validate_stack_dims,
    warner,
    _is_in_module,
)
from vcm import testing


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

    with testing.no_warning(None):
        with warner.allow():
            ds["a"]


@pytest.mark.parametrize(
    "path, module, expected",
    [
        (xr.__file__, xr, True),
        (xr.testing.__file__, xr, True),
        # Test long file name to make sure path inequality works as expected
        ("aadjfoaisdjfa/" * 20 + "localfile.py", xr, False),
        ("localfile.py", xr, False),
    ],
)
def test__is_in_module(path, module, expected):
    assert _is_in_module(path, module)


def test__internal_usage_doesnt_warn():
    """Internal usages of blacklisted functions within xarray should not emit warnings
    """
    ds = xr.Dataset({"a": (["x", "y"], np.ones((10, 5)))})
    a = ds["a"]
    with testing.no_warning(None):
        a.stack(new=["x", "y"])
