import xarray as xr
import numpy as np
import pytest

from vcm.safe import _validate_stack_dims, warn_if_intersecting


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


def test_warn_if_intersecting_duplicates():

    old = {"key"}
    new = ["duplicate", "duplicate"]
    with pytest.warns(UserWarning):
        warn_if_intersecting(old, new)


def test_warn_if_intersecting_overlap():

    old = {"key"}
    new = ["key", "duplicate"]
    with pytest.warns(UserWarning):
        warn_if_intersecting(old, new)


def test_warn_if_intersecting_no_warning():

    old = {"key"}
    new = {"new_key"}

    with pytest.warns(None) as records:
        warn_if_intersecting(old, new)

    # ignore deprecation warnings
    for warning in records:
        assert not isinstance(warning, UserWarning), warning
