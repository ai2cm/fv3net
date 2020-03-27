from typing import Sequence, Hashable, Tuple, Mapping
from collections import MutableMapping
import zarr
import xarray as xr
import numpy as np


def _set_dims(array: zarr.Array, dims: Sequence[Hashable]):
    ARRAY_DIMENSIONS = "_ARRAY_DIMENSIONS"
    array.attrs[ARRAY_DIMENSIONS] = list(dims)


def _create_zarr(dims, coords, group: zarr.Group, template: xr.Dataset):
    ds = template
    group.attrs.update(ds.attrs)

    start_shape = [len(coords[dim]) for dim in dims]
    start_chunks = (1,)*len(start_shape)

    for name in ds:
        _init_data_var(group, ds[name], start_shape, start_chunks, dims)

    for name in ds.coords:
        _init_coord(group, ds[name])

    for name in coords:
        _init_coord(group, coords[name])


def _init_data_var(group: zarr.Group, array: xr.DataArray, start_shape: Tuple[int], start_chunks, dims):
    shape = tuple(start_shape) + array.data.shape
    chunks = tuple(start_chunks) + array.data.shape
    out_array = group.empty(
        name=array.name, shape=shape, chunks=chunks, dtype=array.dtype
    )
    out_array.attrs.update(array.attrs)
    dims = list(dims) + list(array.dims)
    _set_dims(out_array, dims)


def _fill_value(dtype):
    if np.issubdtype(dtype, np.integer):
        return -1
    else:
        return "NaN"


def _init_coord(group: zarr.Group, coord):
    # fill_value=NaN is needed below for xr.open_zarr to succesfully load this
    # coordinate if decode_cf=True. Otherwise, time=0 gets filled in as nan. very
    # confusing...
    arr = np.asarray(coord)
    out_array = group.array(name=coord.name, data=arr, fill_value=_fill_value(arr.dtype))
    out_array.attrs.update(coord.attrs)
    _set_dims(out_array, coord.dims)


def _index(coord, val) -> int:
    return np.asarray(coord).tolist().index(val)


class ZarrMapping:
    """Store xarray data by key

    """
    def __init__(self, group: zarr.Group, schema: xr.Dataset, dims: Sequence[Hashable], coords: Mapping):
        self.group = group
        self.dims = dims
        self.coords = {name: xr.DataArray(coords[name], name=name, dims=[name])
                       for name in coords}

        _create_zarr(dims, self.coords, self.group, schema)

    def _get_index(self, keys):
        return tuple(_index(self.coords[dim], key) for dim, key in zip(self.dims, keys))

    def __setitem__(self, keys, value: xr.Dataset):
        index = self._get_index(keys)
        for variable in value:
            self.group[variable][index] = np.asarray(value[variable])

