import xarray as xr

from vcm.combining import combine_array_sequence, combine_dataset_sequence

def test_combine_dataset_sequence():
    ds = xr.Dataset({
        'a': (['x'], [0.0]),
        'b': ([], 0.0),
    })

    arrays = dict([
        (("a", 1), ds),
        (("b", 1), ds),
        (("a", 2), ds),
        (("b", 2), ds),
    ])

    ds = combine_dataset_sequence(arrays, ["letter", "number"])

    name = 'a'
    assert isinstance(ds, xr.Dataset)
    assert name in ds
    assert ds[name].dims == ("letter", "number", "x")
    assert ds[name].shape == (2, 2, 1)

    assert 'b' in ds
    assert ds['b'].dims == ("letter", "number")
    assert ds['b'].shape == (2, 2)


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
