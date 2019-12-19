import numpy as np
import pytest
import xarray as xr
from skimage.measure import block_reduce as skimage_block_reduce
import xgcm

from vcm.cubedsphere.coarsen import (
    _block_mode,
    _mode,
    _mode_reduce,
    _ureduce,
    _xarray_block_reduce_dataarray,
    add_coordinates,
    block_coarsen,
    block_edge_sum,
    block_median,
    block_upsample,
    coarsen_coords,
    edge_weighted_block_average,
    horizontal_block_reduce,
    shift_edge_var_to_center,
    weighted_block_average,
)
from vcm.cubedsphere.io import all_filenames, remove_duplicate_coords, subtile_filenames
from vcm.cubedsphere import create_fv3_grid
from vcm.xarray_utils import assert_identical_including_dtype


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
    assert_identical_including_dtype(result, expected)

    # Test the Dataset case
    data = data.to_dataset()
    expected = expected.to_dataset()
    result = remove_duplicate_coords(data)
    assert_identical_including_dtype(result, expected)


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
    assert_identical_including_dtype(result["x"], expected["x"])


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

    assert_identical_including_dtype(result, expected)


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
    assert_identical_including_dtype(result, expected)


@pytest.mark.parametrize(
    ("data", "spacing", "factor", "edge", "expected_data"),
    [
        ([[2, 6, 2], [6, 2, 6]], [[6, 2, 6], [2, 6, 2]], 2, "x", [[3.0, 3.0]]),
        ([[2, 6], [6, 2], [2, 6]], [[6, 2], [2, 6], [6, 2]], 2, "y", [[3.0], [3.0]]),
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
    assert_identical_including_dtype(result, expected)


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
    assert_identical_including_dtype(result, expected)


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
    assert_identical_including_dtype(result, expected)


def test_horizontal_block_reduce_dataarray(input_dataarray):
    coarsening_factor = 2
    block_sizes = {"x": coarsening_factor, "y": coarsening_factor, "z": 1}
    expected = _xarray_block_reduce_dataarray(input_dataarray, block_sizes, np.median)
    result = horizontal_block_reduce(
        input_dataarray, coarsening_factor, np.median, "x", "y"
    )
    assert_identical_including_dtype(result, expected)


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

    assert_identical_including_dtype(result, expected)


def test_block_median(input_dataarray):
    coarsening_factor = 2
    block_sizes = {"x": coarsening_factor, "y": coarsening_factor, "z": 1}
    expected = _xarray_block_reduce_dataarray(input_dataarray, block_sizes, np.median)
    result = block_median(input_dataarray, coarsening_factor, "x", "y")
    assert_identical_including_dtype(result, expected)


def test_block_median_via_block_coarsen(input_dataarray):
    coarsening_factor = 2
    block_sizes = {"x": coarsening_factor, "y": coarsening_factor, "z": 1}
    expected = _xarray_block_reduce_dataarray(input_dataarray, block_sizes, np.median)
    result = block_coarsen(
        input_dataarray, coarsening_factor, "x", "y", method="median"
    )
    assert_identical_including_dtype(result, expected)


def test_block_coarsen(input_dataarray):
    coarsening_factor = 2
    method = "min"
    expected = input_dataarray.coarsen(x=coarsening_factor, y=coarsening_factor).min()
    result = block_coarsen(input_dataarray, coarsening_factor, "x", "y", method)
    assert_identical_including_dtype(result, expected)


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
    assert_identical_including_dtype(result, expected)


@pytest.fixture(params=[np.float32, np.float64])
def dtype(request):
    return request.param


@pytest.fixture(params=[0, 1, 2])
def subtile_x(request):
    return request.param


@pytest.fixture(params=[0, 1, 2])
def subtile_y(request):
    return request.param


@pytest.fixture()
def input_subtile_x_coordinates(subtile_x):
    return (np.array([1.0, 2.0, 3.0, 4.0]) + 4.0 * subtile_x).astype(np.float32)


@pytest.fixture()
def input_subtile_y_coordinates(subtile_y):
    return (np.array([1.0, 2.0, 3.0, 4.0]) + 4.0 * subtile_y).astype(np.float32)


@pytest.fixture()
def expected_subtile_x_coordinates(subtile_x):
    data = (np.array([1.0, 2.0]) + 2.0 * subtile_x).astype(np.float32)
    return xr.DataArray(data, dims=["x"], coords=[data], name="x")


@pytest.fixture()
def expected_subtile_y_coordinates(subtile_y):
    data = (np.array([1.0, 2.0]) + 2.0 * subtile_y).astype(np.float32)
    return xr.DataArray(data, dims=["y"], coords=[data], name="y")


@pytest.fixture()
def input_dataarray_with_subtile_coordinates(
    input_subtile_x_coordinates, input_subtile_y_coordinates, dtype
):
    shape = (4, 4, 2)
    data = np.arange(np.product(shape)).reshape(shape).astype(dtype)
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

    assert_identical_including_dtype(result["x"], expected_subtile_x_coordinates)
    assert_identical_including_dtype(result["y"], expected_subtile_y_coordinates)
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

    assert_identical_including_dtype(result["x"], expected_subtile_x_coordinates)
    assert_identical_including_dtype(result["y"], expected_subtile_y_coordinates)
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

    assert_identical_including_dtype(result["x"], expected_subtile_x_coordinates)
    assert_identical_including_dtype(result["y"], expected_subtile_y_coordinates)
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

    assert_identical_including_dtype(result["x"], expected_subtile_x_coordinates)
    assert_identical_including_dtype(result["y"], expected_subtile_y_coordinates)
    assert "z" not in result.coords


@pytest.fixture()
def input_subtile_staggered_x_coordinates(subtile_x):
    return (np.array([1.0, 2.0, 3.0, 4.0, 5.0]) + 4.0 * subtile_x).astype(np.float32)


@pytest.fixture()
def input_subtile_staggered_y_coordinates(subtile_y):
    return (np.array([1.0, 2.0, 3.0, 4.0]) + 4.0 * subtile_y).astype(np.float32)


@pytest.fixture()
def expected_subtile_staggered_x_coordinates(subtile_x):
    data = (np.array([1.0, 2.0, 3.0]) + 2.0 * subtile_x).astype(np.float32)
    return xr.DataArray(data, dims=["x"], coords=[data], name="x")


@pytest.fixture()
def expected_subtile_staggered_y_coordinates(subtile_y):
    data = (np.array([1.0, 2.0]) + 2.0 * subtile_y).astype(np.float32)
    return xr.DataArray(data, dims=["y"], coords=[data], name="y")


@pytest.fixture()
def input_dataarray_with_staggered_subtile_coordinates(
    input_subtile_staggered_x_coordinates, input_subtile_staggered_y_coordinates, dtype
):
    shape = (5, 4, 2)
    data = np.arange(np.product(shape)).reshape(shape).astype(dtype)
    dims = ["x", "y", "z"]
    coords = {
        "x": input_subtile_staggered_x_coordinates,
        "y": input_subtile_staggered_y_coordinates,
    }
    print(data.dtype)
    print(coords["x"].dtype)
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

    assert_identical_including_dtype(
        result["x"], expected_subtile_staggered_x_coordinates
    )
    assert_identical_including_dtype(
        result["y"], expected_subtile_staggered_y_coordinates
    )
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

    assert_identical_including_dtype(
        result["x"], expected_subtile_staggered_x_coordinates
    )
    assert_identical_including_dtype(
        result["y"], expected_subtile_staggered_y_coordinates
    )
    assert "z" not in result.coords


@pytest.mark.parametrize(
    "axes",
    [
        (0,),
        (1,),
        (2,),
        (0, 1),
        (0, 2),
        (1, 2),
        (0, 1, 2),
        (2, 0),
        (2, 1),
        (1, 0),
        (2, 0, 1),
        (2, 1, 0),
        (),
        None,
    ],
    ids=lambda x: f"axis={x}",
)
def test__ureduce(axes):
    def _median(arr, axis=-1):
        """A median function that works strictly on a single axis.

        We will demonstrate in this test that our implementation of _ureduce
        properly transforms this function into one that can be applied to
        multiple axes.
        """
        if axis is not None:
            arr = np.moveaxis(arr, axis, -1)
            return np.median(arr, axis=-1)
        else:
            return np.median(arr)

    arr = np.random.random((5, 3, 5))

    result = _ureduce(arr, _median, axis=axes)
    expected = np.median(arr, axis=axes)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("axes", [1, "a", (1, -1), (-1,)], ids=lambda x: f"axis={x}")
def test__ureduce_error(axes):
    def _median(arr, axis=-1):
        """A median function that works strictly on a single axis.

        We will demonstrate in this test that our implementation of _ureduce
        properly transforms this function into one that can be applied to
        multiple axes.
        """
        arr = np.moveaxis(arr, axis, -1)
        return np.median(arr, axis=-1)

    arr = np.random.random((5,))

    with pytest.raises(ValueError, match="in-house version of _ureduce"):
        _ureduce(arr, _median, axis=axes)


@pytest.mark.parametrize(
    ("array", "axis", "kwargs", "expected"),
    [
        (np.array([[1, 1, 0], [0, 0, 1], [0, 1, 1]]), None, {}, 1),
        (np.array([[1, 1, 0], [0, 0, 1], [0, 1, 1]]), 0, {}, np.array([0, 1, 1])),
        (np.array([[1, 1, 0], [0, 0, 1], [0, 1, 1]]), 1, {}, np.array([1, 0, 1])),
        (np.array([np.nan, np.nan, 3.0]), None, {"nan_policy": "omit"}, 3.0),
    ],
    ids=[
        "axis=None, kwargs={}",
        "axis=0, kwargs={}",
        "axis=1, kwargs={}",
        "axis=None, kwargs={'nan_policy': 'omit'}",
    ],
)
def test__mode(array, axis, expected, kwargs):
    result = _mode(array, axis=axis, **kwargs)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    ("array", "axis", "kwargs", "expected"),
    [
        (np.array([[[1, 1, 0], [0, 0, 1], [0, 1, 1]]]), None, {}, 1),
        (
            np.array([[[1, 1, 0], [0, 0, 1], [0, 1, 1]]]),
            (0, 1),
            {},
            np.array([0, 1, 1]),
        ),
        (
            np.array([[[1, 1, 0], [0, 0, 1], [0, 1, 1]]]),
            (0, 2),
            {},
            np.array([1, 0, 1]),
        ),
        (
            np.array([[[1, 1, 0], [0, 0, 1], [np.nan, 1, 1]]]),
            (0, 2),
            {"nan_policy": "omit"},
            np.array([1, 0, 1]),
        ),
    ],
    ids=[
        "axis=None, kwargs={}",
        "axis=(0, 1), kwargs={}",
        "axis=(0, 2), kwargs={}",
        "axis=(0, 2), kwargs={'nan_policy': 'omit'}",
    ],
)
def test__mode_reduce(array, axis, expected, kwargs):
    result = _mode_reduce(array, axis=axis, **kwargs)
    np.testing.assert_array_equal(result, expected)


def test__block_mode():
    data = np.array(
        [
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, np.nan],
        ]
    )
    da = xr.DataArray(data, dims=["x", "y"])

    expected_data = np.array([[0.0, 1.0], [1.0, 0.0]])
    expected = xr.DataArray(expected_data, dims=["x", "y"])

    result = _block_mode(da, 2, x_dim="x", y_dim="y", nan_policy="omit")
    assert_identical_including_dtype(result, expected)


def test_block_mode_via_block_coarsen():
    data = np.array(
        [
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, np.nan],
        ]
    )
    da = xr.DataArray(data, dims=["x", "y"])

    expected_data = np.array([[0.0, 1.0], [1.0, 0.0]])
    expected = xr.DataArray(expected_data, dims=["x", "y"])

    result = block_coarsen(
        da, 2, x_dim="x", y_dim="y", method="mode", func_kwargs={"nan_policy": "omit"}
    )
    assert_identical_including_dtype(result, expected)


@pytest.mark.parametrize("use_dask", [False, True])
def test_block_upsample_dataset(use_dask):
    foo = xr.DataArray([[1, 2], [3, 4]], dims=["xt", "yt"], name="foo")
    u = xr.DataArray([[1, 2, 3], [4, 5, 6]], dims=["xt", "y"], name="u")

    ds = xr.merge([foo, u])

    expected_foo = xr.DataArray(
        [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]],
        dims=["xt", "yt"],
        name="foo",
    )
    expected_u = xr.DataArray(
        [[1, 1, 2, 2, 3], [1, 1, 2, 2, 3], [4, 4, 5, 5, 6], [4, 4, 5, 5, 6]],
        dims=["xt", "y"],
        name="u",
    )

    expected = xr.merge([expected_foo, expected_u])

    if use_dask:
        ds = ds.chunk({"xt": 1})

    result = block_upsample(ds, 2, dims=["xt", "y", "yt"])
    assert_identical_including_dtype(result, expected)


@pytest.mark.parametrize(
    ("data", "expected_data"),
    [
        ([[1, 2], [3, 4]], [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]),
        (
            [[1, 2, 3], [4, 5, 6]],
            [[1, 1, 2, 2, 3], [1, 1, 2, 2, 3], [4, 4, 5, 5, 6], [4, 4, 5, 5, 6]],
        ),
    ],
)
@pytest.mark.parametrize("use_dask", [False, True])
def test_block_upsample_dataarray(data, expected_data, use_dask):
    foo = xr.DataArray(data, dims=["x", "y"], name="foo")
    expected = xr.DataArray(expected_data, dims=["x", "y"], name="foo")
    if use_dask:
        foo = foo.chunk({"x": 1})

    result = block_upsample(foo, 2, dims=["x", "y"])
    assert_identical_including_dtype(result, expected)


def test_create_fv3_grid_fails_without_tile_coord():
    ds = xr.DataArray([1.0], dims=["tile"]).to_dataset(name="a")

    with pytest.raises(ValueError):
        create_fv3_grid(ds)


def test_create_fv3_grid_fails_on_incomplete_tile_coord():
    ds = xr.Dataset(
        {"a": (["tile", "grid_yt", "grid_xt"], np.ones((5, 1, 1)))},
        coords={"tile": [1, 2, 3, 4, 5]},
    )

    with pytest.raises(ValueError):
        create_fv3_grid(ds)


@pytest.fixture()
def grid_dataset():
    return xr.Dataset(
        {"a": (["tile", "grid_yt", "grid_xt"], np.ones((6, 2, 2)))},
        coords={"tile": [0, 1, 2, 3, 4, 5]},
    )


def test_create_fv3_grid_succeeds(grid_dataset):
    grid = create_fv3_grid(grid_dataset)
    assert isinstance(grid, xgcm.Grid)


def test_xgcm_grid_interp(grid_dataset):

    grid = create_fv3_grid(grid_dataset)
    grid.interp(grid_dataset.a, "x")
