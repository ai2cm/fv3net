import numpy as np
import pytest
import xarray as xr
from skimage.measure import block_reduce as skimage_block_reduce
from vcm.cubedsphere import (
    _xarray_block_reduce_dataarray,
    add_coordinates,
    all_filenames,
    block_coarsen,
    block_edge_sum,
    block_median,
    coarsen_coords,
    edge_weighted_block_average,
    horizontal_block_reduce,
    remove_duplicate_coords,
    shift_edge_var_to_center,
    subtile_filenames,
    weighted_block_average,
)


@pytest.fixture()
def test_y_component_edge_array():
    y_component_edge_coords = {"tile": [1], "grid_yt": [1], "grid_x": [1, 2]}
    y_component_edge_arr = np.array([[[30, 40]]])
    y_component_edge_da = xr.DataArray(
        y_component_edge_arr,
        dims=["tile", "grid_yt", "grid_x"],
        coords=y_component_edge_coords,
    )
    return y_component_edge_da


@pytest.fixture()
def test_x_component_edge_array():
    x_component_edge_coords = {"tile": [1], "grid_y": [1, 2], "grid_xt": [1]}
    x_component_edge_arr = np.array([[[10], [20]]])
    x_component_edge_da = xr.DataArray(
        x_component_edge_arr,
        dims=["tile", "grid_y", "grid_xt"],
        coords=x_component_edge_coords,
    )
    return x_component_edge_da


@pytest.fixture()
def test_centered_vector():
    centered_coords = {"tile": [1], "grid_yt": [1], "grid_xt": [1]}
    x_component_da = xr.DataArray(
        [[[15]]], dims=["tile", "grid_yt", "grid_xt"], coords=centered_coords
    )
    y_component_da = xr.DataArray(
        [[[35]]], dims=["tile", "grid_yt", "grid_xt"], coords=centered_coords
    )
    centered_vector = xr.Dataset(
        {"x_component": x_component_da, "y_component": y_component_da}
    )
    return centered_vector


def test_shift_edge_var_to_center(
    test_y_component_edge_array, test_x_component_edge_array, test_centered_vector
):
    centered_x_component = shift_edge_var_to_center(test_x_component_edge_array)
    centered_y_component = shift_edge_var_to_center(test_y_component_edge_array)

    xr.testing.assert_equal(centered_x_component, test_centered_vector.x_component)
    xr.testing.assert_equal(centered_y_component, test_centered_vector.y_component)

    with pytest.raises(ValueError):
        shift_edge_var_to_center(test_centered_vector)


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


@pytest.mark.parametrize(
    ("coarsening_factor", "input_coordinate", "expected_coordinate"),
    [
        (2, [1.0, 2.0, 3.0, 4.0], [1.0, 2.0]),
        (2, [5.0, 6.0, 7.0, 8.0], [3.0, 4.0]),
        (2, [1.0, 2.0, 3.0], [1.0, 2.0]),
        (2, [3.0, 4.0, 5.0], [2.0, 3.0]),
    ],
    ids=[
        "cell centers; first subtile",
        "cell centers; second subtile",
        "cell interfaces; first subtile",
        "cell interfaces; second subtile",
    ],
)
def test_coarsen_coords(coarsening_factor, input_coordinate, expected_coordinate):
    input_coordinate = xr.DataArray(
        input_coordinate, dims=["x"], coords=[input_coordinate], name="x"
    )
    expected_coordinate = np.array(expected_coordinate).astype(np.float32)
    expected_coordinate = xr.DataArray(
        expected_coordinate, dims=["x"], coords=[expected_coordinate], name="x"
    )
    expected = {"x": expected_coordinate}

    result = coarsen_coords(coarsening_factor, input_coordinate, ["x"])
    xr.testing.assert_identical(result["x"], expected["x"])


@pytest.mark.parametrize("coarsened_object_type", ["DataArray", "Dataset"])
def test_add_coordinates(coarsened_object_type):
    coarsening_factor = 2

    x = np.array([1.0, 2.0]).astype(np.float32)
    y = np.array([3.0, 4.0]).astype(np.float32)
    reference_obj = xr.DataArray(
        [[1, 1], [1, 1]], dims=["x", "y"], coords=[x, y], name="foo"
    )

    coarsened_obj = xr.DataArray([[1]], dims=["x", "y"], coords=None, name="foo")
    if coarsened_object_type == "Dataset":
        coarsened_obj = coarsened_obj.to_dataset()

    coarse_x = np.array([1.0]).astype(np.float32)
    coarse_y = np.array([2.0]).astype(np.float32)
    expected = xr.DataArray(
        [[1]], dims=["x", "y"], coords=[coarse_x, coarse_y], name="foo"
    )
    if coarsened_object_type == "Dataset":
        expected = expected.to_dataset()

    result = add_coordinates(
        reference_obj, coarsened_obj, coarsening_factor, ["x", "y"]
    )

    xr.testing.assert_identical(result, expected)


@pytest.mark.parametrize("object_type", ["DataArray", "Dataset"])
def test_weighted_block_average(object_type):
    coarsening_factor = 2
    dims = ["x", "y"]
    data = xr.DataArray(
        np.array([[2.0, 6.0], [6.0, 2.0]]), dims=dims, coords=None, name="foo"
    )

    if object_type == "Dataset":
        data = data.to_dataset()

    weights = xr.DataArray(np.array([[6.0, 2.0], [2.0, 6.0]]), dims=dims, coords=None)

    expected = xr.DataArray(np.array([[3.0]]), dims=dims, coords=None, name="foo")
    if object_type == "Dataset":
        expected = expected.to_dataset()

    result = weighted_block_average(
        data, weights, coarsening_factor, x_dim="x", y_dim="y"
    )
    xr.testing.assert_identical(result, expected)


@pytest.mark.parametrize(
    ("data", "spacing", "factor", "edge", "expected_data"),
    [
        ([[2, 6, 2], [6, 2, 6]], [[6, 2, 6], [2, 6, 2]], 2, "x", [[3, 3]]),
        ([[2, 6], [6, 2], [2, 6]], [[6, 2], [2, 6], [6, 2]], 2, "y", [[3], [3]]),
    ],
    ids=["edge='x'", "edge='y'"],
)
def test_edge_weighted_block_average(data, spacing, factor, edge, expected_data):
    dims = ["x_dim", "y_dim"]
    da = xr.DataArray(data, dims=dims, coords=None)
    weights = xr.DataArray(spacing, dims=dims, coords=None)

    expected = xr.DataArray(expected_data, dims=dims, coords=None)

    result = edge_weighted_block_average(
        da, weights, factor, x_dim="x_dim", y_dim="y_dim", edge=edge
    )
    xr.testing.assert_identical(result, expected)


@pytest.fixture()
def input_dataarray():
    shape = (4, 4, 2)
    data = np.arange(np.product(shape)).reshape(shape).astype(np.float32)
    dims = ["x", "y", "z"]
    return xr.DataArray(data, dims=dims, coords=None, name="foo")


@pytest.fixture()
def input_dataset(input_dataarray):
    bar = xr.DataArray([1, 2, 3], dims=["t"], name="bar")
    return xr.merge([input_dataarray, bar])


@pytest.mark.parametrize("reduction_function", [np.mean, np.median])
@pytest.mark.parametrize("use_dask", [False, True])
def test_xarray_block_reduce_dataarray(reduction_function, use_dask, input_dataarray):
    block_size = (2, 2, 1)
    expected_data = skimage_block_reduce(
        input_dataarray.values, block_size=block_size, func=reduction_function
    )
    expected = xr.DataArray(
        expected_data, dims=input_dataarray.dims, coords=None, name="foo"
    )

    if use_dask:
        input_dataarray = input_dataarray.chunk({"x": 2, "y": 2, "z": -1})

    block_sizes = {"x": 2, "y": 2}
    result = _xarray_block_reduce_dataarray(
        input_dataarray, block_sizes, reduction_function
    )
    xr.testing.assert_identical(result, expected)


def test_xarray_block_reduce_dataarray_bad_chunk_size(input_dataarray):
    input_dataarray = input_dataarray.chunk({"x": -1, "y": 3, "z": -1})
    block_sizes = {"x": 1, "y": 2, "z": 1}
    with pytest.raises(ValueError, match="All chunks along dimension"):
        _xarray_block_reduce_dataarray(input_dataarray, block_sizes, np.median)


@pytest.mark.parametrize(
    "coord_func",
    ["mean", np.mean, {"x": np.max, "y": "median"}, {"x": np.min}],
    ids=[
        "single str",
        "single function",
        "dict mapping coord name to str or function",
        "dict with a dimension missing",
    ],
)
def test_block_reduce_dataarray_coordinates(input_dataarray, coord_func):
    # Add coordinates to the input_dataarray; make sure coordinate behavior
    # matches xarray's default for coarsen.  This ensures that the
    # coordinate transformation behavior for any function that depends on
    # _block_reduce_dataarray matches that for xarray's coarsen.
    for dim, size in input_dataarray.sizes.items():
        input_dataarray[dim] = np.arange(size)

    block_sizes = {"x": 2, "y": 2}
    result = _xarray_block_reduce_dataarray(
        input_dataarray, block_sizes, np.median, coord_func=coord_func
    )
    expected = (
        input_dataarray.coarsen(x=2, y=2, coord_func=coord_func)
        .median()
        .rename(input_dataarray.name)
    )
    xr.testing.assert_identical(result, expected)


def test_horizontal_block_reduce_dataarray(input_dataarray):
    coarsening_factor = 2
    block_sizes = {"x": coarsening_factor, "y": coarsening_factor, "z": 1}
    expected = _xarray_block_reduce_dataarray(input_dataarray, block_sizes, np.median)
    result = horizontal_block_reduce(
        input_dataarray, coarsening_factor, np.median, "x", "y"
    )
    xr.testing.assert_identical(result, expected)


def test_horizontal_block_reduce_dataset(input_dataset):
    coarsening_factor = 2
    block_sizes = {"x": coarsening_factor, "y": coarsening_factor, "z": 1}

    expected_foo = _xarray_block_reduce_dataarray(
        input_dataset.foo, block_sizes, np.median
    )

    # No change expected to bar, because it contains no horizontal dimensions.
    expected_bar = input_dataset.bar
    expected = xr.merge([expected_foo, expected_bar])

    result = horizontal_block_reduce(
        input_dataset, coarsening_factor, np.median, "x", "y"
    )

    xr.testing.assert_identical(result, expected)


def test_block_median(input_dataarray):
    coarsening_factor = 2
    block_sizes = {"x": coarsening_factor, "y": coarsening_factor, "z": 1}
    expected = _xarray_block_reduce_dataarray(input_dataarray, block_sizes, np.median)
    result = block_median(input_dataarray, coarsening_factor, "x", "y")
    xr.testing.assert_identical(result, expected)


def test_block_coarsen(input_dataarray):
    coarsening_factor = 2
    method = "min"
    expected = input_dataarray.coarsen(x=coarsening_factor, y=coarsening_factor).min()
    result = block_coarsen(input_dataarray, coarsening_factor, "x", "y", method)
    xr.testing.assert_identical(result, expected)


@pytest.mark.parametrize(
    ("data", "factor", "edge", "expected_data"),
    [
        ([[2, 6, 2], [6, 2, 6]], 2, "x", [[8, 8]]),
        ([[2, 6], [6, 2], [2, 6]], 2, "y", [[8], [8]]),
    ],
    ids=["edge='x'", "edge='y'"],
)
def test_block_edge_sum(data, factor, edge, expected_data):
    dims = ["x_dim", "y_dim"]
    da = xr.DataArray(data, dims=dims, coords=None)
    expected = xr.DataArray(expected_data, dims=dims, coords=None)
    result = block_edge_sum(da, factor, x_dim="x_dim", y_dim="y_dim", edge=edge)
    xr.testing.assert_identical(result, expected)


@pytest.fixture(params=[0, 1, 2])
def subtile_x(request):
    return request.param


@pytest.fixture(params=[0, 1, 2])
def subtile_y(request):
    return request.param


@pytest.fixture()
def input_subtile_x_coordinates(subtile_x):
    return np.array([1.0, 2.0, 3.0, 4.0]) + 4.0 * subtile_x


@pytest.fixture()
def input_subtile_y_coordinates(subtile_y):
    return np.array([1.0, 2.0, 3.0, 4.0]) + 4.0 * subtile_y


@pytest.fixture()
def expected_subtile_x_coordinates(subtile_x):
    data = np.array([1.0, 2.0]) + 2.0 * subtile_x
    return xr.DataArray(data, dims=["x"], coords=[data], name="x")


@pytest.fixture()
def expected_subtile_y_coordinates(subtile_y):
    data = np.array([1.0, 2.0]) + 2.0 * subtile_y
    return xr.DataArray(data, dims=["y"], coords=[data], name="y")


@pytest.fixture()
def input_dataarray_with_subtile_coordinates(
    input_subtile_x_coordinates, input_subtile_y_coordinates
):
    shape = (4, 4, 2)
    data = np.arange(np.product(shape)).reshape(shape).astype(np.float32)
    dims = ["x", "y", "z"]
    coords = {"x": input_subtile_x_coordinates, "y": input_subtile_y_coordinates}
    return xr.DataArray(data, dims=dims, coords=coords, name="foo")


def test_weighted_block_average_with_coordinates(
    input_dataarray_with_subtile_coordinates,
    expected_subtile_x_coordinates,
    expected_subtile_y_coordinates,
):
    coarsening_factor = 2
    weights = input_dataarray_with_subtile_coordinates
    result = weighted_block_average(
        input_dataarray_with_subtile_coordinates, weights, coarsening_factor, "x", "y"
    )

    xr.testing.assert_identical(result["x"], expected_subtile_x_coordinates)
    xr.testing.assert_identical(result["y"], expected_subtile_y_coordinates)
    assert "z" not in result.coords


def test_horizontal_block_reduce_with_coordinates(
    input_dataarray_with_subtile_coordinates,
    expected_subtile_x_coordinates,
    expected_subtile_y_coordinates,
):
    coarsening_factor = 2
    result = horizontal_block_reduce(
        input_dataarray_with_subtile_coordinates, coarsening_factor, np.mean, "x", "y"
    )

    xr.testing.assert_identical(result["x"], expected_subtile_x_coordinates)
    xr.testing.assert_identical(result["y"], expected_subtile_y_coordinates)
    assert "z" not in result.coords


def test_block_median_with_coordinates(
    input_dataarray_with_subtile_coordinates,
    expected_subtile_x_coordinates,
    expected_subtile_y_coordinates,
):
    coarsening_factor = 2
    result = block_median(
        input_dataarray_with_subtile_coordinates, coarsening_factor, "x", "y"
    )

    xr.testing.assert_identical(result["x"], expected_subtile_x_coordinates)
    xr.testing.assert_identical(result["y"], expected_subtile_y_coordinates)
    assert "z" not in result.coords


def test_block_coarsen_with_coordinates(
    input_dataarray_with_subtile_coordinates,
    expected_subtile_x_coordinates,
    expected_subtile_y_coordinates,
):
    coarsening_factor = 2
    result = block_coarsen(
        input_dataarray_with_subtile_coordinates, coarsening_factor, "x", "y", "sum"
    )

    xr.testing.assert_identical(result["x"], expected_subtile_x_coordinates)
    xr.testing.assert_identical(result["y"], expected_subtile_y_coordinates)
    assert "z" not in result.coords


@pytest.fixture()
def input_subtile_staggered_x_coordinates(subtile_x):
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0]) + 4.0 * subtile_x


@pytest.fixture()
def input_subtile_staggered_y_coordinates(subtile_y):
    return np.array([1.0, 2.0, 3.0, 4.0]) + 4.0 * subtile_y


@pytest.fixture()
def expected_subtile_staggered_x_coordinates(subtile_x):
    data = np.array([1.0, 2.0, 3.0]) + 2.0 * subtile_x
    return xr.DataArray(data, dims=["x"], coords=[data], name="x")


@pytest.fixture()
def expected_subtile_staggered_y_coordinates(subtile_y):
    data = np.array([1.0, 2.0]) + 2.0 * subtile_y
    return xr.DataArray(data, dims=["y"], coords=[data], name="y")


@pytest.fixture()
def input_dataarray_with_staggered_subtile_coordinates(
    input_subtile_staggered_x_coordinates, input_subtile_staggered_y_coordinates
):
    shape = (5, 4, 2)
    data = np.arange(np.product(shape)).reshape(shape).astype(np.float32)
    dims = ["x", "y", "z"]
    coords = {
        "x": input_subtile_staggered_x_coordinates,
        "y": input_subtile_staggered_y_coordinates,
    }
    return xr.DataArray(data, dims=dims, coords=coords, name="foo")


def test_edge_weighted_block_average_with_coordinates(
    input_dataarray_with_staggered_subtile_coordinates,
    expected_subtile_staggered_x_coordinates,
    expected_subtile_staggered_y_coordinates,
):
    coarsening_factor = 2
    spacing = input_dataarray_with_staggered_subtile_coordinates
    result = edge_weighted_block_average(
        input_dataarray_with_staggered_subtile_coordinates,
        spacing,
        coarsening_factor,
        "x",
        "y",
        edge="y",
    )

    xr.testing.assert_identical(result["x"], expected_subtile_staggered_x_coordinates)
    xr.testing.assert_identical(result["y"], expected_subtile_staggered_y_coordinates)
    assert "z" not in result.coords


def test_block_edge_sum_with_coordinates(
    input_dataarray_with_staggered_subtile_coordinates,
    expected_subtile_staggered_x_coordinates,
    expected_subtile_staggered_y_coordinates,
):
    coarsening_factor = 2
    result = block_edge_sum(
        input_dataarray_with_staggered_subtile_coordinates,
        coarsening_factor,
        "x",
        "y",
        edge="y",
    )

    xr.testing.assert_identical(result["x"], expected_subtile_staggered_x_coordinates)
    xr.testing.assert_identical(result["y"], expected_subtile_staggered_y_coordinates)
    assert "z" not in result.coords
