import logging

import dask.array as da
import numpy as np
import xarray as xr
from dask.delayed import delayed

from vcm.cloud.fsspec import get_fs


def _read_metadata_remote(fs, url):
    logging.info("Reading metadata")
    with fs.open(url, "rb") as f:
        return xr.open_dataset(f)


def _open_remote_nc(fs, url):
    with fs.open(url, "rb") as f:
        return xr.open_dataset(f).load()


def open_tiles(url_prefix: str) -> xr.Dataset:
    """Lazily open a set of FV3 netCDF tile files

    Args:
        url_prefix: the prefix of the set of files before ".tile?.nc". The metadata
            and dimensions are harvested from the first tile only.
    Returns:
        dataset of merged tiles.

    """
    fs = get_fs(url_prefix)
    files = sorted(fs.glob(url_prefix + ".tile?.nc"))
    if len(files) != 6:
        raise ValueError(
            f"Invalid set of input files. {len(files)} detected, but 6 expected."
        )
    schema = _read_metadata_remote(fs, files[0])
    delayeds = [delayed(_open_remote_nc)(fs, url) for url in files]
    datasets = [open_delayed(d, schema) for d in delayeds]
    return xr.concat(datasets, dim="tile").assign_coords(tile=list(range(6)))


def _delayed_to_array(delayed_dataset, key, shape, dtype):
    null = da.full(shape, np.nan, dtype=dtype)
    array_delayed = delayed_dataset.get(key, null)
    return da.from_delayed(array_delayed, shape, dtype)


def open_delayed(delayed_dataset, schema: xr.Dataset) -> xr.Dataset:
    """Open dask delayed object with the same metadata as template

    Mostly useful for lazily loading remote resources. For example, this greatly
    accelerates opening a list of remote netCDF resources conforming to the same
    "schema".

    Args:
        delayed_dataset: a dask delayed object which resolves to an xarray Dataset
        schema: an xarray Dataset with the same coords and dims as the
            Dataset wrapped with the delayed object.

    Returns:
        dataset: a dask-array backed dataset

    Example:

        >>> import xarray as xr
        >>> from dask.delayed import delayed
        >>> @delayed
        ... def identity(x):
        ...     return x
        ...
        >>> ds = xr.Dataset({'a': (['x'], np.ones(10))})
        >>> delayed = identity(ds)
        >>> delayed
        Delayed('identity-6539bc06-097a-4864-8cbf-699ebe3c4130')
        >>> wrapped_delayed_obj = open_delayed(delayed, schema=ds)
        >>> wrapped_delayed_obj
        <xarray.Dataset>
        Dimensions:  (x: 10)
        Dimensions without coordinates: x
        Data variables:
            a        (x) float64 dask.array<chunksize=(10,), meta=np.ndarray>


    """
    data_vars = {}
    for key in schema:
        template_var = schema[key]
        array = _delayed_to_array(
            delayed_dataset, key, shape=template_var.shape, dtype=template_var.dtype
        )
        data_vars[key] = (template_var.dims, array, template_var.attrs)
    return xr.Dataset(data_vars, coords=schema.coords, attrs=schema.attrs)
