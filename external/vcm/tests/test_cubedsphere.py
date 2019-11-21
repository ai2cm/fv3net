import pytest
import numpy as np
import xarray as xr

from vcm.cubedsphere import (
    shift_edge_var_to_center,
    remove_duplicate_coords,
    weighted_block_average,
    subtile_filenames,
    all_filenames,
)


@pytest.fixture()
def test_y_component_edge_array():
    y_component_edge_coords = {"tile": [1], "grid_yt": [1], "grid_x": [1, 2]}
    y_component_edge_arr = np.array([[[30, 40]]])
    y_component_edge_da = xr.DataArray(
        y_component_edge_arr,
        dims=["tile", "grid_yt", "grid_x"],
        coords=y_component_edge_coords
    )
    return y_component_edge_da


@pytest.fixture()
def test_x_component_edge_array():
    x_component_edge_coords = {"tile": [1], "grid_y": [1, 2], "grid_xt": [1]}
    x_component_edge_arr = np.array([[[10], [20]]])
    x_component_edge_da = xr.DataArray(
        x_component_edge_arr,
        dims=["tile", "grid_y", "grid_xt"],
        coords=x_component_edge_coords
    )
    return x_component_edge_da

@pytest.fixture()
def test_centered_vector():
    centered_coords = {"tile": [1], "grid_yt": [1], "grid_xt": [1]}
    x_component_da = xr.DataArray(
        [[[15]]],
        dims=["tile", "grid_yt", "grid_xt"],
        coords=centered_coords)
    y_component_da = xr.DataArray(
        [[[35]]],
        dims=["tile", "grid_yt", "grid_xt"],
        coords=centered_coords)
    centered_vector = xr.Dataset({"x_component": x_component_da, "y_component": y_component_da})
    return centered_vector


def test_shift_edge_var_to_center(
        test_y_component_edge_array,
        test_x_component_edge_array,
        test_centered_vector
):
    centered_x_component = shift_edge_var_to_center(test_x_component_edge_array)
    centered_y_component = shift_edge_var_to_center(test_y_component_edge_array)

    xr.testing.assert_equal(centered_x_component, test_centered_vector.x_component)
    xr.testing.assert_equal(centered_y_component, test_centered_vector.y_component)

    with pytest.raises(ValueError):
        shift_edge_var_to_center(_test_weights_array(n=10))


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
