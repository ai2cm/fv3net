import unittest
import fv3net
import xarray as xr
import zarr

def test_zarr_mapping_set():
    keys = list('abc')
    schema = xr.Dataset({'a': (['x'], [2.0])}).chunk()
    store = {}
    group = zarr.open_group(store)
    m = fv3net.ZarrMapping(group, keys, schema, dim='time')
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
