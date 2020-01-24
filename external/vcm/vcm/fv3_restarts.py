import os
import re
from datetime import datetime
from typing import Any, Generator, Tuple

import cftime
import fsspec
import xarray as xr
from dask.delayed import delayed

from vcm.schema_registry import impose_dataset_to_schema
from vcm.combining import combine_array_sequence
from vcm.convenience import open_delayed

TIME_FMT = "%Y%m%d.%H%M%S"
SCHEMA_CACHE = {}
RESTART_CATEGORIES = ["fv_core.res", "sfc_data", "fv_tracer", "fv_srf_wnd.res"]


def open_restarts(url: str) -> xr.Dataset:
    """Opens all the restart file within a certain path

    The dimension names are the same as the diagnostic output

    Args:
        url: a URL to the root directory of a run directory. Can be any type of protocol
            used by fsspec, such as google cloud storage 'gs://path-to-rundir'. If no
            protocol prefix is used, then it will be assumed to be a path to a local
            file.
            
    Returns:
        a combined dataset of all the restart files. All except the first file of
        each restart-file type (e.g. fv_core.res) will only be lazily loaded. This
        allows opening large datasets out-of-core.

    """
    restart_files = _restart_files_at_url(url)
    arrays = _load_arrays(restart_files)
    return xr.Dataset(
        combine_array_sequence(arrays, labels=["file_prefix", "tile"])
    ).pipe(_sort_file_prefixes)


def standardize_metadata(ds: xr.Dataset) -> xr.Dataset:
    """Update the meta-data of an individual restart file

    This drops the singleton time dimension and applies the known dimensions
    listed in `vcm.schema` and `vcm._schema_registry`.
    """
    try:
        ds_no_time = ds.isel(Time=0).drop("Time")
    except ValueError:
        ds_no_time = ds
    return impose_dataset_to_schema(ds_no_time)


def _parse_time_string(time):
    t = datetime.strptime(time, TIME_FMT)
    return cftime.DatetimeJulian(t.year, t.month, t.day, t.hour, t.minute, t.second)


def _split_url(url):

    try:
        protocol, path = url.split("://")
    except ValueError:
        protocol = "file"
        path = url

    return protocol, path


def _parse_time(path):
    return re.search(r"(\d\d\d\d\d\d\d\d\.\d\d\d\d\d\d)", path).group(1)


def _get_file_prefix(dirname, path):
    if dirname.endswith("INPUT"):
        return "INPUT/"
    elif dirname.endswith("RESTART"):
        try:
            return os.path.join("RESTART", _parse_time(path))
        except AttributeError:
            return "RESTART/"


def _sort_file_prefixes(ds):

    if "INPUT/" not in ds.file_prefix:
        raise ValueError("Open restarts() did not find the input set of restart files.")
    if "RESTART/" not in ds.file_prefix:
        raise ValueError("Open_restarts() did not find the final set of restart files.")

    intermediate_prefixes = sorted(
        [
            prefix.item()
            for prefix in ds.file_prefix
            if prefix.item() not in ["INPUT/", "RESTART/"]
        ]
    )

    return xr.concat(
        [
            ds.sel(file_prefix="INPUT/"),
            ds.sel(file_prefix=intermediate_prefixes),
            ds.sel(file_prefix="RESTART/"),
        ],
        dim="file_prefix",
    )


def _parse_category(path):
    cats_in_path = {category for category in RESTART_CATEGORIES if category in path}
    if len(cats_in_path) == 1:
        return cats_in_path.pop()
    else:
        # Check that the file only matches one restart category for safety
        # it not clear if this is completely necessary, but it ensures the output of
        # this routine is more predictable
        raise ValueError("Multiple categories present in filename.")


def _get_tile(path):
    """Get tile number

    Following python, but unlike FV3, the first tile number is 0. In other words, the
    tile number of `.tile1.nc` is 0.

    This avoids confusion when using the outputs of :ref:`open_restarts`.
    """
    tile = re.search(r"tile(\d)\.nc", path).group(1)
    return int(tile) - 1


def _is_restart_file(path):
    return any(category in path for category in RESTART_CATEGORIES) and "tile" in path


def _restart_files_at_url(url):
    """List restart files with a given initial and end time within a particular URL

    Yields:
        (time, restart_category, tile, protocol, path)

    """
    proto, path = _split_url(url)
    fs = fsspec.filesystem(proto)

    for root, dirs, files in fs.walk(path):
        for file in files:
            path = os.path.join(root, file)
            if _is_restart_file(file):
                file_prefix = _get_file_prefix(root, file)
                tile = _get_tile(file)
                category = _parse_category(file)
                yield file_prefix, category, tile, proto, path


def _load_restart(protocol, path):
    fs = fsspec.filesystem(protocol)
    with fs.open(path) as f:
        return xr.open_dataset(f).compute()


def _load_restart_with_schema(protocol, path, schema):
    promise = delayed(_load_restart)(protocol, path)
    return open_delayed(promise, schema)


def _load_restart_lazily(protocol, path, restart_category):
    # only actively load the initial data
    if restart_category in SCHEMA_CACHE:
        schema = SCHEMA_CACHE[restart_category]
    else:
        schema = _load_restart(protocol, path)
        SCHEMA_CACHE[restart_category] = schema

    return _load_restart_with_schema(protocol, path, schema)


def _load_arrays(
    restart_files,
) -> Generator[Tuple[Any, Tuple, xr.DataArray], None, None]:
    # use the same schema for all coupler_res
    for (file_prefix, restart_category, tile, protocol, path) in restart_files:
        ds = _load_restart_lazily(protocol, path, restart_category)
        ds_standard_metadata = standardize_metadata(ds)
        #         time_obj = _parse_time_string(time)
        for var in ds_standard_metadata:
            yield var, (file_prefix, tile), ds_standard_metadata[var]
