import xarray as xr

from vcm.merging import combine_by_key


def test__concatenate_by_key():
    arr = xr.DataArray([0.0], dims=["x"])
    arrays = {("a", 1): arr, ("a", 2): arr, ("b", 1): arr, ("b", 2): arr}

    ds = combine_by_key(arrays, ["letter", "number"])

    assert isinstance(ds, xr.DataArray)
    assert ds.dims == ("letter", "number", "x")
    assert ds.shape == (2, 2, 1)
