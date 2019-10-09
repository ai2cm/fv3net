import xarray as xr
import numpy as np

floating_point_types = tuple(np.dtype(t) for t in ['float32', 'float64', 'float128'])


def is_float(x):
    if x.dtype in floating_point_types:
        return True
    else:
        return False


def cast_doubles_to_floats(ds: xr.Dataset):
    coords = {}
    data_vars = {}

    for key in ds.coords:
        coord = ds[key]
        if is_float(coord):
            coords[key] = ds[key].astype(np.float32)

    for key in ds.data_vars:
        var = ds[key]
        if is_float(var):
            data_vars[key] = ds[key].astype(np.float32).drop(var.coords)

    return xr.Dataset(data_vars, coords=coords)
