import xarray as xr

from vcm.combining import combine_array_sequence


def test_combine_array_sequence():
    name = "a"
    arr = xr.DataArray([0.0], dims=["x"])
    arrays = [
        (name, ("a", 1), arr),
        (name, ("b", 1), arr),
        (name, ("a", 2), arr),
        (name, ("b", 2), arr),
    ]

    ds = combine_array_sequence(arrays, ["letter", "number"])

    assert isinstance(ds, xr.Dataset)
    assert name in ds
    assert ds[name].dims == ("letter", "number", "x")
    assert ds[name].shape == (2, 2, 1)
