import numpy as np
import xarray as xr
from vcm.convenience import open_delayed
from dask.delayed import delayed
from dask.array import Array

import pytest


@pytest.fixture()
def dataset():
    arr = np.random.rand(100, 10)
    coords = dict(time=np.arange(100), x=np.arange(10),)
    return xr.Dataset({"a": (["time", "x"], arr), 'b': (['time', 'x'], arr)}, coords=coords)

def test_open_delayed(dataset):
    a_delayed = delayed(lambda x: x)(dataset)
    ds = open_delayed(a_delayed, schema=dataset)

    xr.testing.assert_equal(dataset, ds.compute())

def test_open_delayed_fills_nans(dataset):
    ds_no_b = dataset[['a']]
    # wrap idenity with delated object
    a_delayed = delayed(lambda x: x)(ds_no_b)
    ds = open_delayed(a_delayed, schema=dataset)

    # test that b is filled with anans
    b = ds['b'].compute()
    assert np.all(np.isnan(b))
    assert b.dims == dataset['b'].dims
    assert b.dtype == dataset['b'].dtype
