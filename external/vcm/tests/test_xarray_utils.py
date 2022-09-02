import dask
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from vcm.xarray_utils import (
    _repeat_dataarray,
    assert_identical_including_dtype,
    isclose,
    repeat,
    weighted_mean_via_groupby_bins,
)


@pytest.mark.parametrize("use_dask", [False, True])
@pytest.mark.parametrize(
    ("a", "b", "expected", "kwargs"),
    [
        ([1.0], [1.0 + 1.0e-4], [False], {}),
        ([1.0], [1.0 + 1.0e-9], [True], {}),
        ([1.0], [1.0 + 1.0e-4], [True], {"rtol": 1.0e-3}),
        ([1.0], [1.0 + 1.0e-4], [True], {"atol": 1.0e-3}),
        ([np.nan], [np.nan], [False], {}),
        ([np.nan], [np.nan], [True], {"equal_nan": True}),
        ([1.0], [1.0], [True], {}),
        ([1.0], [1.0 + 1.0e-9], [False], {"rtol": 1.0e-10, "atol": 1.0e-10}),
    ],
)
def test_isclose(use_dask, a, b, expected, kwargs):
    a = xr.DataArray(a)
    b = xr.DataArray(b)

    if use_dask:
        a = a.chunk()
        b = b.chunk()

    expected = xr.DataArray(expected)
    result = isclose(a, b, **kwargs)
    assert_identical_including_dtype(result, expected)


@pytest.mark.parametrize(
    "dim", ["x", "not_a_dim"], ids=["dim in DataArray", "dim absent from DataArray"]
)
@pytest.mark.parametrize("use_dask", [False, True])
def test__repeat_dataarray(dim, use_dask):
    da = xr.DataArray([1, 2, 3], dims=["x"], coords=[[1, 2, 3]])
    if use_dask:
        da = da.chunk()

    if dim == "x":
        expected = xr.DataArray([1, 1, 2, 2, 3, 3], dims=["x"])
    else:
        expected = da.copy(deep=True)
    result = _repeat_dataarray(da, 2, dim)

    if use_dask:
        assert isinstance(result.data, dask.array.Array)

    assert_identical_including_dtype(result, expected)


@pytest.mark.parametrize("object_type", ["Dataset", "DataArray"])
@pytest.mark.parametrize("dim", ["x", "z"], ids=["dim present", "dim not present"])
@pytest.mark.parametrize(
    ("expected_foo_data", "repeats"),
    [([1, 1, 2, 2, 3, 3], 2), ([1, 1, 2, 3], [2, 1, 1])],
    ids=["integer repeats argument", "array repeats argument"],
)
def test_repeat(object_type, dim, expected_foo_data, repeats):
    foo = xr.DataArray([1, 2, 3], dims=["x"], coords=[[1, 2, 3]], name="foo")
    bar = xr.DataArray([1, 2, 3], dims=["y"], coords=[[1, 2, 3]], name="bar")

    expected_foo = xr.DataArray(expected_foo_data, dims=["x"], name="foo")
    expected_bar = bar.copy(deep=True)

    if object_type == "Dataset":
        obj = xr.merge([foo, bar])
        expected = xr.merge([expected_foo, expected_bar])
    else:
        obj = foo
        expected = expected_foo

    if dim == "x":
        result = repeat(obj, repeats, dim)
        assert_identical_including_dtype(result, expected)
    else:
        with pytest.raises(ValueError, match="Cannot repeat over 'z'"):
            repeat(obj, 2, dim)


@pytest.mark.parametrize("object_type", ["DataArray", "Dataset"])
def test_assert_identical_including_dtype(object_type):
    a = xr.DataArray([1, 2], dims=["x"], coords=[[0, 1]], name="foo")
    b = xr.DataArray([1, 2], dims=["x"], coords=[[0, 1]], name="foo")
    c = a.astype("float64")
    d = a.copy(deep=True)
    d["x"] = d.x.astype("float64")

    if object_type == "Dataset":
        a = a.to_dataset()
        b = b.to_dataset()
        c = c.to_dataset()
        d = d.to_dataset()

    assert_identical_including_dtype(a, b)

    with pytest.raises(AssertionError):
        assert_identical_including_dtype(c, b)

    with pytest.raises(AssertionError):
        assert_identical_including_dtype(d, b)


def test_weighted_mean_via_groupby_bins():
    a = xr.DataArray(np.arange(10), dims=["x"], name="foo")
    group = xr.DataArray(np.arange(10), dims=["x"], name="bar")
    bins = np.array(np.arange(0, 11, 2))
    weights = (group % 2) == 1

    result = weighted_mean_via_groupby_bins(a, group, weights, bins)
    expected_coordinate = pd.interval_range(start=0, end=10, freq=2)
    expected = xr.DataArray(
        np.arange(1, 11, 2, dtype=float),
        dims=["bar_bins"],
        coords=[expected_coordinate],
        name="foo",
    )
    assert_identical_including_dtype(result, expected)


def test_weighted_mean_via_groupby_bins_with_nans():
    a = xr.DataArray(np.arange(10), dims=["x"], name="foo")
    a = a.where((a % 5) > 2)  # Introduce some NaNs
    group = xr.DataArray(np.arange(10), dims=["x"], name="bar")
    bins = np.array(np.arange(0, 11, 5))

    # This is just testing that NaNs are properly zeroed out in the weights,
    # so we can use an array of ones to keep things simple.
    weights = xr.ones_like(group)

    result = weighted_mean_via_groupby_bins(a, group, weights, bins)
    expected_coordinate = pd.interval_range(start=0, end=10, freq=5)
    expected = xr.DataArray(
        [3.5, 8.5], dims=["bar_bins"], coords=[expected_coordinate], name="foo"
    )
    assert_identical_including_dtype(result, expected)


def test_weighted_mean_via_groupby_bins_dataset():
    # This test just ensures the broadcasting behavior matches that
    # of xarray's built-in groupby_bins.  The logic for computing weighted
    # means and ignoring NaNs is tested at the DataArray level above.
    a = xr.DataArray(np.arange(10), dims=["x"], name="foo", attrs={"units": "mm/day"})
    b = xr.DataArray(np.arange(10), dims=["y"], name="bar", attrs={"units": "K"})
    ds = xr.Dataset({a.name: a, b.name: b})

    group = xr.DataArray(np.arange(10), dims=["x"], name="baz")
    bins = np.array(np.arange(0, 11, 2))
    weights = xr.ones_like(a)

    result = weighted_mean_via_groupby_bins(ds, group, weights, bins)
    with xr.set_options(keep_attrs=True):
        expected = ds.groupby_bins(group, bins).mean()
    assert_identical_including_dtype(result, expected)
