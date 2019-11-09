import pytest
import numpy as np
import xarray as xr


from vcm.cubedsphere import (
    remove_duplicate_coords,
    weighted_block_average,
    subtile_filenames,
    all_filenames,
    keep_attrs
)


def test_keep_attrs():

    def fun(x):
        # this will destroy the metadata
        return x * 1.0

    x = xr.DataArray([1.0], dims=['x'], attrs={'units': 'm'})
    a = xr.DataArray([1.0], dims=['x'], attrs={'hello': 'world'}, coords={'x': x})

    fun = keep_attrs(fun)
    b = fun(a)

    assert b.hello == 'world'
    assert b.x.units == 'm'


def test_subtile_filenames():
    paths = subtile_filenames(prefix="test", tile=1, num_subtiles=2)
    expected = ["test.tile1.nc.0000", "test.tile1.nc.0001"]
    assert list(paths) == expected


@pytest.mark.parametrize("n", [2, 4, 5, 16])
def test_all_filenames(n):
    num_files_expected = n * 6
    files = all_filenames("test", num_subtiles=n)
    assert len(files) == num_files_expected


@pytest.mark.parametrize(
    ("x", "y", "data", "expected_x", "expected_y", "expected_data"),
    [
        ([1, 1], [3, 4], [[1, 2], [3, 4]], [1], [3, 4], [[1, 2]]),
        ([1, 2], [3, 3], [[1, 2], [3, 4]], [1, 2], [3], [[1], [3]]),
        ([1, 1], [3, 3], [[1, 2], [3, 4]], [1], [3], [[1]]),
        ([1, 2], [3, 4], [[1, 2], [3, 4]], [1, 2], [3, 4], [[1, 2], [3, 4]]),
    ],
    ids=["duplicate x", "duplicate y", "duplicate x and y", "no duplicates"],
)
def test_remove_duplicate_coords(x, y, data, expected_x, expected_y, expected_data):
    x = xr.DataArray(x, coords=[x], dims=["x"])
    y = xr.DataArray(y, coords=[y], dims=["y"])
    data = xr.DataArray(data, coords=[x, y], dims=["x", "y"], name="foo")

    expected_x = xr.DataArray(expected_x, coords=[expected_x], dims=["x"])
    expected_y = xr.DataArray(expected_y, coords=[expected_y], dims=["y"])
    expected = xr.DataArray(
        expected_data, coords=[expected_x, expected_y], dims=["x", "y"], name="foo"
    )

    # Test the DataArray case
    result = remove_duplicate_coords(data)
    xr.testing.assert_identical(result, expected)

    # Test the Dataset case
    data = data.to_dataset()
    expected = expected.to_dataset()
    result = remove_duplicate_coords(data)
    xr.testing.assert_identical(result, expected)


def _test_weights_array(n=10):
    coords = {"x": np.arange(n) + 1.0, "y": np.arange(n) + 1.0}
    arr = np.ones((n, n))
    weights = xr.DataArray(arr, dims=["x", "y"], coords=coords)
    return weights


def test_block_weighted_average():
    expected = 2.0
    weights = _test_weights_array(n=10)
    dataarray = expected * weights

    ans = weighted_block_average(dataarray, weights, 5, x_dim="x", y_dim="y")
    assert ans.shape == (2, 2)
    assert np.all(np.isclose(ans, expected))


@pytest.mark.parametrize(
    "start_coord, expected_start",
    [
        # expected_start = (start - 1) / factor + 1
        (1, 1),
        (11, 3),
    ],
)
def test_block_weighted_average_coords(start_coord, expected_start):
    n = 10
    target_n = 2
    factor = n // target_n

    weights = _test_weights_array(n)
    coords = {dim: np.arange(start_coord, start_coord + n) for dim in weights.dims}
    weights = weights.assign_coords(coords)

    # ensure the coords are correct
    for dim in weights.dims:
        assert weights[dim].values[0] == pytest.approx(start_coord)

    ans = weighted_block_average(weights, weights, factor, x_dim="x", y_dim="y")

    for dim in ans.dims:
        assert weights[dim].values[0] == pytest.approx(start_coord)
        expected = np.arange(expected_start, expected_start + target_n)
        np.testing.assert_allclose(ans[dim].values, expected)
