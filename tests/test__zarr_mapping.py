import unittest
import fv3net
import xarray as xr
import zarr
import numpy as np

import pytest


@pytest.mark.parametrize('dtype, fill_value',[
    (int, -1),
    (float, np.nan)
])
def test_zarr_mapping_init_coord_fill_value(dtype, fill_value):
    keys = list('abc')
    arr = np.array([2.0], dtype=dtype)
    schema = xr.Dataset({'x': (['x'], arr)})

    store = {}
    group = zarr.open_group(store)
    m = fv3net.ZarrMapping(group, schema, keys, dim='time')

    # check that both are NaN since NaN != Nan
    if np.isnan(fill_value) and np.isnan(group['x'].fill_value):
        return
    assert group['x'].fill_value == fill_value


def test_zarr_mapping_set(dtype=int):
    keys = list('abc')
    arr = np.array([2.0], dtype=dtype)
    schema = xr.Dataset({'a': (['x'], arr)}).chunk()
    store = {}
    group = zarr.open_group(store)
    m = fv3net.ZarrMapping(group, schema, keys, dim='time')
    m['a'] = schema
    m['b'] = schema
    m['c'] = schema

    ds = xr.open_zarr(store)
    for time in m.keys():
        a = ds.sel(time=time).drop('time').load()
        b = schema.load()
        xr.testing.assert_equal(a, b)



if __name__ == '__main__':
    unittest.main()
