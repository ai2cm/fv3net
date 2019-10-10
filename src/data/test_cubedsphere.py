import pytest
import xarray as xr


from .cubedsphere import remove_duplicate_coords


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
