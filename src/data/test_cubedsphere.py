import pytest
import numpy as np
import xarray as xr

from skimage.measure import block_reduce as skimage_block_reduce

from .cubedsphere import (
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


def _test_weights_array(n=10):
    coords = {'x': np.arange(n)+1.0, 'y': np.arange(n) + 1.0}
    arr = np.ones((n, n))
    weights = xr.DataArray(arr, dims=['x', 'y'], coords=coords)
    return weights


def test_block_weighted_average():
    expected = 2.0
    weights = _test_weights_array(n=10)
    dataarray = expected * weights

    ans = weighted_block_average(dataarray, weights, 5, x_dim='x', y_dim='y')
    assert ans.shape == (2, 2)
    assert np.all(np.isclose(ans, expected))


@pytest.mark.parametrize('start_coord, expected_start', [
    # expected_start = (start - 1) / factor + 1
    (1, 1),
    (11, 3)
])
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

    ans = weighted_block_average(weights, weights, factor, x_dim='x', y_dim='y')

    for dim in ans.dims:
        assert weights[dim].values[0] == pytest.approx(start_coord)
        expected = np.arange(expected_start, expected_start + target_n)
        np.testing.assert_allclose(ans[dim].values, expected)


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
    x_coord = np.arange(1., nx + 1.)
    y_coord = np.arange(1., ny + 1.)
    da = xr.DataArray(data, dims=['x_dim', 'y_dim'], coords=[x_coord, y_coord])
    weights = xr.DataArray(
        spacing,
        dims=['x_dim', 'y_dim'],
        coords=[x_coord, y_coord]
    )

    nx_expected, ny_expected = np.array(expected_data).shape
    x_coord_expected = np.arange(1., nx_expected + 1.)
    y_coord_expected = np.arange(1., ny_expected + 1.)
    expected = xr.DataArray(
        expected_data,
        dims=['x_dim', 'y_dim'],
        coords=[x_coord_expected, y_coord_expected])
    result = edge_weighted_block_average(
        da,
        weights,
        factor,
        x_dim='x_dim',
        y_dim='y_dim',
        edge=edge
    )
    xr.testing.assert_identical(result, expected)


@pytest.mark.parametrize('reduction_function', [np.mean, np.median])
@pytest.mark.parametrize('use_dask', [False, True])
def test_block_reduce_dataarray(reduction_function, use_dask):
    shape = (2, 4, 8)
    nx, ny, nz = shape
    data = np.arange(np.product(shape)).reshape(shape).astype(np.float32)
    x_coord = np.arange(nx)
    y_coord = np.arange(ny)
    z_coord = np.arange(nz)

    dims = ['x', 'y', 'z']
    coords = [x_coord, y_coord, z_coord]
    da = xr.DataArray(data, dims=dims, coords=coords)

    if use_dask:
        da = da.chunk({'x': -1, 'y': -1, 'z': 4})

    expected_data = skimage_block_reduce(
        data,
        block_size=(1, 2, 4),
        func=reduction_function
    )
    expected_dims = ['x', 'y', 'z']
    expected_coords = {'x': x_coord}
    expected = xr.DataArray(
        expected_data,
        dims=expected_dims,
        coords=expected_coords
    )

    block_sizes = {'x': 1, 'y': 2, 'z': 4}
    result = _block_reduce_dataarray(da, block_sizes, reduction_function)
    xr.testing.assert_identical(result, expected)


def test_block_reduce_dataarray_bad_chunk_size():
    shape = (2, 8)
    nx, ny = shape
    data = np.arange(np.product(shape)).reshape(shape).astype(np.float32)
    x_coord = np.arange(nx)
    y_coord = np.arange(ny)

    dims = ['x', 'y']
    coords = [x_coord, y_coord]
    da = xr.DataArray(data, dims=dims, coords=coords)
    da = da.chunk({'x': -1, 'y': 3})

    block_sizes = {'x': 1, 'y': 2}
    with pytest.raises(ValueError, match='All chunks along dimension'):
        _block_reduce_dataarray(da, block_sizes, np.median)


def test_horizontal_block_reduce_dataarray():
    shape = (4, 4, 2)
    nx, ny, nz = shape
    data = np.arange(np.product(shape)).reshape(shape).astype(np.float32)
    x_coord = np.arange(nx)
    y_coord = np.arange(ny)
    z_coord = np.arange(nz)

    dims = ['x', 'y', 'z']
    coords = [x_coord, y_coord, z_coord]
    da = xr.DataArray(data, dims=dims, coords=coords)

    coarsening_factor = 2
    block_sizes = {'x': coarsening_factor, 'y': coarsening_factor, 'z': 1}

    expected = _block_reduce_dataarray(da, block_sizes, np.median)
    expected['x'] = np.arange(nx / coarsening_factor)
    expected['y'] = np.arange(ny / coarsening_factor)

    result = horizontal_block_reduce(
        da,
        coarsening_factor,
        np.median,
        'x',
        'y'
    )

    xr.testing.assert_identical(result, expected)


def test_horizontal_block_reduce_dataset():
    shape = (4, 4, 2)
    nx, ny, nz = shape
    data = np.arange(np.product(shape)).reshape(shape).astype(np.float32)
    x_coord = np.arange(nx)
    y_coord = np.arange(ny)
    z_coord = np.arange(nz)

    dims = ['x', 'y', 'z']
    coords = [x_coord, y_coord, z_coord]
    foo = xr.DataArray(data, dims=dims, coords=coords, name='foo')
    bar = xr.DataArray([1, 2, 3], dims=['t'], name='bar')
    ds = xr.merge([foo, bar])

    coarsening_factor = 2
    block_sizes = {'x': coarsening_factor, 'y': coarsening_factor, 'z': 1}

    expected_foo = _block_reduce_dataarray(foo, block_sizes, np.median)
    expected_foo['x'] = np.arange(nx / coarsening_factor)
    expected_foo['y'] = np.arange(ny / coarsening_factor)
    expected_bar = bar
    expected = xr.merge([expected_foo, expected_bar])

    result = horizontal_block_reduce(
        ds,
        coarsening_factor,
        np.median,
        'x',
        'y'
    )

    xr.testing.assert_identical(result, expected)


def test_block_median():
    shape = (4, 4, 2)
    nx, ny, nz = shape
    data = np.arange(np.product(shape)).reshape(shape).astype(np.float32)
    x_coord = np.arange(nx)
    y_coord = np.arange(ny)
    z_coord = np.arange(nz)

    dims = ['x', 'y', 'z']
    coords = [x_coord, y_coord, z_coord]
    da = xr.DataArray(data, dims=dims, coords=coords)

    coarsening_factor = 2
    block_sizes = {'x': coarsening_factor, 'y': coarsening_factor, 'z': 1}

    expected = _block_reduce_dataarray(da, block_sizes, np.median)
    expected['x'] = np.arange(nx / coarsening_factor)
    expected['y'] = np.arange(ny / coarsening_factor)

    result = block_median(
        da,
        coarsening_factor,
        'x',
        'y'
    )

    xr.testing.assert_identical(result, expected)


def test_block_coarsen():
    shape = (4, 4, 2)
    nx, ny, nz = shape
    data = np.arange(np.product(shape)).reshape(shape).astype(np.float32)
    x_coord = np.arange(nx)
    y_coord = np.arange(ny)
    z_coord = np.arange(nz)

    dims = ['x', 'y', 'z']
    coords = [x_coord, y_coord, z_coord]
    da = xr.DataArray(data, dims=dims, coords=coords)

    coarsening_factor = 2
    method = 'min'

    expected = da.coarsen(x=coarsening_factor, y=coarsening_factor).min()
    expected['x'] = np.arange(nx / coarsening_factor)
    expected['y'] = np.arange(ny / coarsening_factor)

    result = block_coarsen(
        da,
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
    x_coord = np.arange(1., nx + 1.)
    y_coord = np.arange(1., ny + 1.)
    da = xr.DataArray(data, dims=['x_dim', 'y_dim'], coords=[x_coord, y_coord])

    nx_expected, ny_expected = np.array(expected_data).shape
    x_coord_expected = np.arange(1., nx_expected + 1.)
    y_coord_expected = np.arange(1., ny_expected + 1.)
    expected = xr.DataArray(
        expected_data,
        dims=['x_dim', 'y_dim'],
        coords=[x_coord_expected, y_coord_expected])
    result = block_edge_sum(
        da,
        factor,
        x_dim='x_dim',
        y_dim='y_dim',
        edge=edge
    )
    xr.testing.assert_identical(result, expected)
