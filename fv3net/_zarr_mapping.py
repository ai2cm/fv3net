from typing import Sequence, Hashable
from collections import MutableMapping
import zarr
import xarray as xr
import numpy as np


def _set_dims(array: zarr.Array, dims: Sequence[Hashable]):
    ARRAY_DIMENSIONS = "_ARRAY_DIMENSIONS"
    array.attrs[ARRAY_DIMENSIONS] = list(dims)


def _create_zarr(keys, dim, group: zarr.Group, template: xr.Dataset):
    ds = template
    group.attrs.update(ds.attrs)
    nt = len(keys)
    for name in ds:
        _init_data_var(group, ds[name], nt, dim)

    for name in ds.coords:
        _init_coord(group, ds[name], dim)

    dim_var = group.array(dim, data=keys)
    _set_dims(dim_var, [dim])


def _init_data_var(group: zarr.Group, array: xr.DataArray, nt: int, dim):
    shape = (nt,) + array.data.shape
    chunks = (1,) + tuple(size[0] for size in array.data.chunks)
    out_array = group.empty(
        name=array.name, shape=shape, chunks=chunks, dtype=array.dtype
    )
    out_array.attrs.update(array.attrs)
    dims = [dim] + list(array.dims)
    _set_dims(out_array, dims)


def _init_coord(group: zarr.Group, coord, dim):
    # fill_value=NaN is needed below for xr.open_zarr to succesfully load this
    # coordinate if decode_cf=True. Otherwise, time=0 gets filled in as nan. very
    # confusing...
    out_array = group.array(name=coord.name, data=np.asarray(coord), fill_value="NaN")
    out_array.attrs.update(coord.attrs)
    _set_dims(out_array, coord.dims)


class ZarrMapping:
    """Store xarray data by key

    """
    def __init__(self, group: zarr.Group, schema: xr.Dataset, keys: Sequence[Hashable], dim: str):
        self.group = group
        self._keys = list(keys)
        _create_zarr(keys, dim, self.group, schema)

    def __setitem__(self, key: str, value: xr.Dataset):
        index = self.keys().index(key)
        for variable in value:
            self.group[variable][index] = np.asarray(value[variable])

    def keys(self):
        return self._keys
