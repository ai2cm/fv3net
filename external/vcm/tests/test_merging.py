import xarray as xr

from vcm.merging import combine_by_dims


def test__concatenate_by_key():
    name = "a"
    arr = xr.DataArray([0.0], dims=["x"])
    arrays = [
        (name, ("a", 1), arr),
        (name, ("b", 1), arr),
        (name, ("a", 2), arr),
        (name, ("b", 2), arr),
    ]

    ds = combine_by_dims(arrays, ["letter", "number"])

    assert isinstance(ds, xr.Dataset)
    assert name in ds
    assert ds[name].dims == ("letter", "number", "x")
    assert ds[name].shape == (2, 2, 1)
