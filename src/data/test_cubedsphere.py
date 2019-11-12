import pytest
import numpy as np
import xarray as xr

from skimage.measure import block_reduce as skimage_block_reduce

from .cubedsphere import (
    add_coarsened_subtile_coordinates,
    coarsen_subtile_coordinates,
    _block_reduce_dataarray,
    horizontal_block_reduce,
    block_median,
    remove_duplicate_coords,
    weighted_block_average,
    edge_weighted_block_average,
    block_coarsen,
    block_edge_sum
)


@pytest.mark.parametrize(
    ('x', 'y', 'data', 'expected_x', 'expected_y', 'expected_data'),
    [
        ([1, 1], [3, 4], [[1, 2], [3, 4]], [1], [3, 4], [[1, 2]]),
        ([1, 2], [3, 3], [[1, 2], [3, 4]], [1, 2], [3], [[1], [3]]),
        ([1, 1], [3, 3], [[1, 2], [3, 4]], [1], [3], [[1]]),
        ([1, 2], [3, 4], [[1, 2], [3, 4]], [1, 2], [3, 4], [[1, 2], [3, 4]])
    ],
    ids=['duplicate x', 'duplicate y', 'duplicate x and y', 'no duplicates']
)
def test_remove_duplicate_coords(
    x, y, data, expected_x, expected_y, expected_data
):
    x = xr.DataArray(x, coords=[x], dims=['x'])
    y = xr.DataArray(y, coords=[y], dims=['y'])
    data = xr.DataArray(data, coords=[x, y], dims=['x', 'y'], name='foo')

    expected_x = xr.DataArray(expected_x, coords=[expected_x], dims=['x'])
    expected_y = xr.DataArray(expected_y, coords=[expected_y], dims=['y'])
    expected = xr.DataArray(
        expected_data,
        coords=[expected_x, expected_y],
        dims=['x', 'y'],
        name='foo'
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
    ('coarsening_factor', 'input_coordinate', 'expected_coordinate'),
    [
        (2, [1., 2., 3., 4.], [1., 2.]),
        (2, [5., 6., 7., 8.], [3., 4.]),
        (2, [1., 2., 3.], [1., 2.]),
        (2, [3., 4., 5.], [2., 3.])
    ],
    ids=[
        'cell centers; first subtile',
        'cell centers; second subtile',
        'cell interfaces; first subtile',
        'cell interfaces; second subtile'
    ]
)
def test_coarsen_subtile_coordinates(
    coarsening_factor,
    input_coordinate,
    expected_coordinate
):
    input_coordinate = xr.DataArray(
        input_coordinate,
        dims=['x'],
        coords=[input_coordinate],
        name='x'
    )
    expected_coordinate = np.array(expected_coordinate).astype(np.float32)
    expected_coordinate = xr.DataArray(
        expected_coordinate,
        dims=['x'],
        coords=[expected_coordinate],
        name='x'
    )
    expected = {'x': expected_coordinate}

    result = coarsen_subtile_coordinates(
        coarsening_factor,
        input_coordinate,
        ['x']
    )
    xr.testing.assert_identical(result['x'], expected['x'])


@pytest.mark.parametrize('coarsened_object_type', ['DataArray', 'Dataset'])
def test_add_coarsened_subtile_coordinates(coarsened_object_type):
    coarsening_factor = 2

    x = np.array([1., 2.]).astype(np.float32)
    y = np.array([3., 4.]).astype(np.float32)
    reference_obj = xr.DataArray(
        [[1, 1], [1, 1]],
        dims=['x', 'y'],
        coords=[x, y],
        name='foo'
    )

    coarsened_obj = xr.DataArray(
        [[1]],
        dims=['x', 'y'],
        coords=None,
        name='foo'
    )
    if coarsened_object_type == 'Dataset':
        coarsened_obj = coarsened_obj.to_dataset()

    coarse_x = np.array([1.]).astype(np.float32)
    coarse_y = np.array([2.]).astype(np.float32)
    expected = xr.DataArray(
        [[1]],
        dims=['x', 'y'],
        coords=[coarse_x, coarse_y],
        name='foo'
    )
    if coarsened_object_type == 'Dataset':
        expected = expected.to_dataset()

    result = add_coarsened_subtile_coordinates(
        reference_obj,
        coarsened_obj,
        coarsening_factor,
        ['x', 'y']
    )

    xr.testing.assert_identical(result, expected)


@pytest.mark.parametrize('object_type', ['DataArray', 'Dataset'])
def test_weighted_block_average(object_type):
    coarsening_factor = 2
    dims = ['x', 'y']
    data = xr.DataArray(
        np.array([[2., 6.], [6., 2.]]),
        dims=dims,
        coords=None,
        name='foo'
    )

    if object_type == 'Dataset':
        data = data.to_dataset()

    weights = xr.DataArray(
        np.array([[6., 2.], [2., 6.]]),
        dims=dims,
        coords=None
    )

    expected = xr.DataArray(
        np.array([[3.]]),
        dims=dims,
        coords=None,
        name='foo'
    )

    if object_type == 'Dataset':
        expected = expected.to_dataset()

    result = weighted_block_average(
        data,
        weights,
        coarsening_factor,
        x_dim='x',
        y_dim='y'
    )
    xr.testing.assert_identical(result, expected)


@pytest.mark.parametrize(
    ('data', 'spacing', 'factor', 'edge', 'expected_data'),
    [
        ([[2, 6, 2], [6, 2, 6]], [[6, 2, 6], [2, 6, 2]], 2, 'x', [[3, 3]]),
        ([[2, 6], [6, 2], [2, 6]], [[6, 2], [2, 6], [6, 2]], 2, 'y', [[3], [3]])
    ],
    ids=["edge='x'", "edge='y'"]
)
def test_edge_weighted_block_average(
    data,
    spacing,
    factor,
    edge,
    expected_data
):
    nx, ny = np.array(data).shape
    dims = ['x_dim', 'y_dim']
    da = xr.DataArray(data, dims=dims, coords=None)
    weights = xr.DataArray(spacing, dims=dims, coords=None)

    nx_expected, ny_expected = np.array(expected_data).shape
    expected = xr.DataArray(
        expected_data,
        dims=dims,
        coords=None
    )
    result = edge_weighted_block_average(
        da,
        weights,
        factor,
        x_dim='x_dim',
        y_dim='y_dim',
        edge=edge
    )
    xr.testing.assert_identical(result, expected)


@pytest.fixture()
def input_dataarray():
    shape = (4, 4, 2)
    nx, ny, nz = shape
    data = np.arange(np.product(shape)).reshape(shape).astype(np.float32)
    dims = ['x', 'y', 'z']
    return xr.DataArray(data, dims=dims, coords=None, name='foo')


@pytest.fixture()
def input_dataset(input_dataarray):
    bar = xr.DataArray([1, 2, 3], dims=['t'], name='bar')
    return xr.merge([input_dataarray, bar])


@pytest.mark.parametrize('reduction_function', [np.mean, np.median])
@pytest.mark.parametrize('use_dask', [False, True])
def test_block_reduce_dataarray(reduction_function, use_dask, input_dataarray):
    block_size = (2, 2, 1)
    expected_data = skimage_block_reduce(
        input_dataarray.values,
        block_size=block_size,
        func=reduction_function
    )
    expected = xr.DataArray(
        expected_data,
        dims=input_dataarray.dims,
        coords=None,
        name='foo'
    )

    if use_dask:
        input_dataarray = input_dataarray.chunk({'x': 2, 'y': 2, 'z': -1})

    block_sizes = dict(zip(input_dataarray.dims, block_size))
    result = _block_reduce_dataarray(
        input_dataarray,
        block_sizes,
        reduction_function
    )
    xr.testing.assert_identical(result, expected)


def test_block_reduce_dataarray_bad_chunk_size(input_dataarray):
    input_dataarray = input_dataarray.chunk({'x': -1, 'y': 3, 'z': -1})
    block_sizes = {'x': 1, 'y': 2, 'z': 1}
    with pytest.raises(ValueError, match='All chunks along dimension'):
        _block_reduce_dataarray(input_dataarray, block_sizes, np.median)


def test_block_reduce_dataarray_coordinate_behavior(input_dataarray):
    # Add coordinates to the input_dataarray; make sure coordinate behavior
    # matches xarray's default for coarsen.  This ensures that the default
    # coordinate transformation behavior for any function that depends on
    # _block_reduce_dataarray matches that for xarray's coarsen.
    for dim, size in input_dataarray.sizes.items():
        input_dataarray[dim] = np.arange(size)

    block_sizes = {'x': 2, 'y': 2, 'z': 1}
    result = _block_reduce_dataarray(
        input_dataarray,
        block_sizes,
        np.median
    )
    expected = input_dataarray.coarsen(x=2, y=2).median().rename(
        input_dataarray.name
    )
    xr.testing.assert_identical(result, expected)


def test_horizontal_block_reduce_dataarray(input_dataarray):
    coarsening_factor = 2
    block_sizes = {'x': coarsening_factor, 'y': coarsening_factor, 'z': 1}

    expected = _block_reduce_dataarray(input_dataarray, block_sizes, np.median)
    result = horizontal_block_reduce(
        input_dataarray,
        coarsening_factor,
        np.median,
        'x',
        'y'
    )
    xr.testing.assert_identical(result, expected)


def test_horizontal_block_reduce_dataset(input_dataset):
    coarsening_factor = 2
    block_sizes = {'x': coarsening_factor, 'y': coarsening_factor, 'z': 1}

    expected_foo = _block_reduce_dataarray(
        input_dataset.foo,
        block_sizes,
        np.median
    )

    # No change expected to bar, because it contains no horizontal dimensions.
    expected_bar = input_dataset.bar
    expected = xr.merge([expected_foo, expected_bar])

    result = horizontal_block_reduce(
        input_dataset,
        coarsening_factor,
        np.median,
        'x',
        'y'
    )

    xr.testing.assert_identical(result, expected)


def test_block_median(input_dataarray):
    coarsening_factor = 2
    block_sizes = {'x': coarsening_factor, 'y': coarsening_factor, 'z': 1}

    expected = _block_reduce_dataarray(
        input_dataarray,
        block_sizes,
        np.median
    )

    result = block_median(
        input_dataarray,
        coarsening_factor,
        'x',
        'y'
    )

    xr.testing.assert_identical(result, expected)


def test_block_coarsen(input_dataarray):
    coarsening_factor = 2
    method = 'min'

    expected = input_dataarray.coarsen(
        x=coarsening_factor,
        y=coarsening_factor
    ).min()

    result = block_coarsen(
        input_dataarray,
        coarsening_factor,
        'x',
        'y',
        method
    )

    xr.testing.assert_identical(result, expected)


@pytest.mark.parametrize(
    ('data', 'factor', 'edge', 'expected_data'),
    [
        ([[2, 6, 2], [6, 2, 6]], 2, 'x', [[8, 8]]),
        ([[2, 6], [6, 2], [2, 6]], 2, 'y', [[8], [8]])
    ],
    ids=["edge='x'", "edge='y'"]
)
def test_block_edge_sum(
    data,
    factor,
    edge,
    expected_data
):
    nx, ny = np.array(data).shape
    dims = ['x_dim', 'y_dim']
    da = xr.DataArray(data, dims=dims, coords=None)

    nx_expected, ny_expected = np.array(expected_data).shape
    expected = xr.DataArray(expected_data, dims=dims, coords=None)

    result = block_edge_sum(
        da,
        factor,
        x_dim='x_dim',
        y_dim='y_dim',
        edge=edge
    )
    xr.testing.assert_identical(result, expected)
