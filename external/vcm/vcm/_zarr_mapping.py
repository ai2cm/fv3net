from typing import Sequence, Hashable, Tuple, Mapping
import zarr
import xarray as xr
import numpy as np
import logging

logger = logging.getLogger("ZarrMapping")


def _set_dims(array: zarr.Array, dims: Sequence[Hashable]):
    array.attrs["_ARRAY_DIMENSIONS"] = list(dims)


def _create_zarr(dims: Sequence[str], coords: Mapping[str, Sequence], group: zarr.Group, template: xr.Dataset):
    ds = template
    group.attrs.update(ds.attrs)

    start_shape = [len(coords[dim]) for dim in dims]
    start_chunks = (1,) * len(start_shape)

    for name in ds:
        _init_data_var(group, ds[name], start_shape, start_chunks, dims)

    for name in ds.coords:
        _init_coord(group, ds[name])

    for name in coords:
        _init_coord(group, coords[name])

    group.attrs["DIMS"] = dims


def _init_data_var(
    group: zarr.Group, array: xr.DataArray, start_shape: Tuple[int], start_chunks, dims
):
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


def _init_coord(group: zarr.Group, coord: xr.DataArray):
    # fill_value=NaN is needed below for xr.open_zarr to succesfully load this
    # coordinate if decode_cf=True. Otherwise, time=0 gets filled in as nan. very
    # confusing...
    arr = np.asarray(coord)
    out_array = group.array(
        name=coord.name, data=arr, fill_value=_fill_value(arr.dtype)
    )
    out_array.attrs.update(coord.attrs)
    _set_dims(out_array, coord.dims)


def _index(coord, val) -> int:
    return np.asarray(coord).tolist().index(val)


class ZarrMapping:
    """A database like front end to zarr

    This object can be initialized once at the start of job, and then data inserted
    into it by multiple workers such as in a large Map-reduce like job.

    This object should be initialized from a template xarray object (see :ref:`from_schema`),
    and a list of dimensions and corresponding coordinate labels that will be managed by the
    ZarrMapping.

    Once initialized, xarray datasets with the exact same dimensions and coordinates
    as the template array, can be inserted into it using index like notation::

        mapping[(key1, key2)] = data

    A chunk size of 1 is used along the dimensions managed by ZarrMapping. This allows
    independent workers to write concurrently to the ZarrMapping without race
    conditions, provided they each have unique key.

    Example:

        >>> import xarray as xr
        >>> import vcm
        >>> import numpy as  np
        >>> template = xr.Dataset({'a': (['x'], np.ones(10))})
        >>> store = {}
        >>> mapping = vcm.ZarrMapping.from_schema(store, template, dims=['time'], coords={'time': [0, 1, 2, 3]})
        >>> mapping[(0,)] = template
        >>> mapping[(1,)] = template
        >>> mapping[(2,)] = template
        >>> mapping[(3,)] = template
        >>> template
        <xarray.Dataset>
        Dimensions:  (x: 10)
        Dimensions without coordinates: x
        Data variables:
            a        (x) float64 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
        >>> xr.open_zarr(store)
        <xarray.Dataset>
        Dimensions:  (time: 4, x: 10)
        Coordinates:
        * time     (time) float64 0.0 1.0 2.0 3.0
        Dimensions without coordinates: x
        Data variables:
            a        (time, x) float64 dask.array<chunksize=(1, 10), meta=np.ndarray>
        Attributes:
            DIMS:     ['time']

    """

    def __init__(self, store: zarr.ABSStore):
        self.store = store

    @property
    def group(self):
        return zarr.open_group(self.store, mode="a")

    @property
    def dims(self):
        return self.group.attrs["DIMS"]

    @property
    def coords(self):
        g = self.group
        return {dim: g[dim][:] for dim in self.dims}

    @staticmethod
    def from_schema(store: zarr.ABSStore, schema: xr.Dataset, dims: Sequence[str], coords: Mapping[str, Sequence]) -> "ZarrMapping":
        """Initialize a ZarrMapping using an xarray dataset as a template

        Args:
            store: A object implementing the mutable mapping interface required by zarr.open_group
            schema: A template for the datasets that will be inserted into the ZarrMapping.
            dims: The list of dimensions that will be managed by the zarr mapping. The zarr dataset
                produced by ZarrMapping will have these dimensions pre-pendended to the list of
                dimensions of each variable in the schema object. 
            coords: the coordinate labels corresponding to the dimensions in dims

        Returns:
            an initialized ZarrMapping object

        """
        group = zarr.open_group(store, mode="w")
        coords = {
            name: xr.DataArray(coords[name], name=name, dims=[name]) for name in coords
        }
        _create_zarr(dims, coords, group, schema)
        return ZarrMapping(store)

    def _get_index(self, keys):
        return tuple(_index(self.coords[dim], key) for dim, key in zip(self.dims, keys))

    def __setitem__(self, keys, value: xr.Dataset):
        index = self._get_index(keys)
        for variable in value:
            logger.debug(f"Setting {variable}")
            self.group[variable][index] = np.asarray(value[variable])

    def flush(self):
        return self.group.store.flush()
