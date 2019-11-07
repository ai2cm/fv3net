import xarray as xr
import numpy as np
import subprocess
import tempfile
from dask.delayed import delayed
import dask.array as da
from collections import defaultdict

import logging


floating_point_types = tuple(np.dtype(t) for t in ["float32", "float64", "float128"])


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


@delayed
def _open_remote_nc(url):
    with tempfile.NamedTemporaryFile() as fp:
        logging.info("downloading %s to disk" % url)
        subprocess.check_call((["gsutil", "-q", "cp", url, fp.name]))
        return xr.open_dataset(fp.name).load()


def file_names_for_time_step(timestep, category, resolution=3072):
    # TODO remove this hardcode
    bucket = f"gs://vcm-ml-data/2019-10-28-X-SHiELD-2019-10-05-multiresolution-extracted/C{resolution}/{timestep}/{timestep}.{category}*"
    return gslist(bucket)


def tile_num(name):
    return name[-len("1.nc.0000")]


def group_file_names(files):
    out = defaultdict(list)
    for file in files:
        out[tile_num(file)].append(file)

    return out


def map_ops(fun, grouped_files, *args):
    out = {}
    for key, files in grouped_files.items():
        seq = []
        for file in files:
            ds = fun(file, *args)
            seq.append(ds)
        out[key] = seq
    return out


def open_remote_nc(path, meta=None):

    computation = delayed(_open_remote_nc)(path)

    data_vars = {}
    for key in meta:
        template_var = meta[key]
        array = da.from_delayed(
            computation[key], shape=template_var.shape, dtype=template_var.dtype
        )
        data_vars[key] = (template_var.dims, array)

    return xr.Dataset(data_vars)
