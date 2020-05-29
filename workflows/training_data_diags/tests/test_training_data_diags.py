import pytest
import xarray as xr
import numpy as np
from training_data_diags import __version__
from training_data_diags import utils


def test_version():
    assert __version__ == "0.1.0"

da = xr.DataArray(np.arange(1., 5.), dims=['z'])
ds = xr.Dataset({'a': da})
weights = xr.DataArray([0.5, 0.5, 1, 1], dims=['z'])
    
@pytest.mark.parametrize(
    'da,weights,dims,expected',
    [
        (da, weights, "z", xr.DataArray(17.0/6.0)),
        (ds, weights, "z", xr.Dataset({'a': xr.DataArray(17.0/6.0)}))
    ]
)
def test_weighted_average(da, weights, dims, expected):
    xr.testing.assert_allclose(utils.weighted_average(da, weights, dims), expected)
    

def test_weighted_averaged_no_dims():
    
    da = xr.DataArray([[[np.arange(1., 5.)]]], dims=['tile', 'y', 'x', 'z'])
    weights = xr.DataArray([[[[0.5, 0.5, 1, 1]]]], dims=['tile', 'y', 'x', 'z'])
    expected = xr.DataArray(np.arange(1., 5.), dims=['z'])
    
    xr.testing.assert_allclose(utils.weighted_average(da, weights), expected)
    
enumeration = {'land': 1, 'sea': 0}

@pytest.mark.parametrize(
    "float_mask,enumeration,atol,expected", [
        (
            xr.DataArray([1.0, 0.0], dims=['x']),
            enumeration,
            1e-7,
            xr.DataArray(['land', 'sea'], dims=['x'])
        ),
        (
            xr.DataArray([1.0000001, 0.0], dims=['x']),
            enumeration,
            1e-7,
            xr.DataArray(['land', 'sea'], dims=['x'])
        ),
        (
            xr.DataArray([1.0001, 0.0], dims=['x']),
            enumeration,
            1e-7,
            xr.DataArray([np.nan, 'sea'], dims=['x'])
        )
    ]
)
def test_snap_mask_to_type(float_mask, enumeration, atol, expected):
    xr.testing.assert_equal(
        utils.snap_mask_to_type(float_mask, enumeration, atol),
        expected
    )
    
    
ds = xr.Dataset(
    {'a': xr.DataArray([[[np.arange(1., 5.)]]], dims=['z', 'tile', 'y', 'x'])}
)
surface_type_da = xr.DataArray(
    [[[['sea', 'land', 'land', 'land']]]],
    dims=['z', 'tile', 'y', 'x',]
)
area = xr.DataArray([1., 1., 1., 1.,], dims=['x'])

@pytest.mark.parametrize(
    "ds,surface_type_da,surface_type,area,expected",
    [
        (ds, surface_type_da, 'sea', area, xr.Dataset({'a': xr.DataArray([1.0], dims=['z'])})),
        (ds, surface_type_da, 'land', area, xr.Dataset({'a': xr.DataArray([3.0], dims=['z'])}))
    ]
)
def test__conditional_average(ds, surface_type_da, surface_type, area, expected):
    
    average = utils._conditional_average(ds, surface_type_da, surface_type, area)
    xr.testing.assert_allclose(average, expected)
    
    