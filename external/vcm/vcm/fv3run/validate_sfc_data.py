import xarray as xr
import numpy as np
import sys


def binarize(x):
    return (x > .5).astype(x.dtype)


ds = xr.open_mfdataset(sys.argv[1:], concat_dim='tile')

np.testing.assert_array_equal(binarize(ds.slmsk), ds.slmsk)

print("Data is Valid")
