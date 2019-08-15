from .base import _flatten
import numpy as np
import xarray as xr

def test__flatten():
    x = np.ones((3,4,5))
    shape = (3, 4, 5)
    dims = 'x y z'.split()
    sample_dim = 'z'
    
    nx, ny, nz = shape

    a = xr.DataArray(x, dims=dims)
    ds = xr.Dataset({'a': a, 'b': a})

    ans = _flatten(ds, sample_dim)
    assert ans.shape == (nz, 2*nx*ny)
    assert isinstance(ans, np.ndarray)
    

def test__flatten_1d_input():
    x = np.ones((3,4,5))
    shape = (3, 4, 5)
    dims = 'x y z'.split()
    sample_dim = 'z'
    
    nx, ny, nz = shape

    a = xr.DataArray(x, dims=dims)
    ds = xr.Dataset({'a': a, 'b': a.isel(x=0, y=0)})

    ans = _flatten(ds, sample_dim)
    assert ans.shape == (nz, nx*ny + 1)
    assert isinstance(ans, np.ndarray)


def test__flatten_same_order():
    nx, ny = 10, 4
    x = xr.DataArray(
            np.arange(nx*ny).reshape((nx, ny)), dims=['sample', 'feature'])

    ds = xr.Dataset({'a': x ,'b': x.T})
    sample_dim = 'sample'
    a = _flatten(ds[['a']], sample_dim)
    b = _flatten(ds[['b']], sample_dim)

    np.testing.assert_allclose(a, b)

