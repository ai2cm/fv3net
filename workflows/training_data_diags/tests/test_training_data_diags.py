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
    