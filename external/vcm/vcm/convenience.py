from datetime import timedelta
import logging
import os
import subprocess
import tempfile
from collections import defaultdict
import pathlib
from typing import List

import numpy as np
import dask.array as da
import intake
import xarray as xr
import yaml
import re
from dask import delayed

from vcm.cloud import gsutil
from vcm.cloud.remote_data import open_gfdl_data_with_2d
from vcm.cloud.fsspec import get_fs

TOP_LEVEL_DIR = pathlib.Path(__file__).parent.parent.absolute()


def round_time(t):
    """ cftime will introduces noise when decoding values into date objects.
    This rounds time in the date object to the nearest second, assuming the init time
    is at most 1 sec away from a round minute. This is used when merging datasets so
    their time dims match up.

    Args:
        t: datetime or cftime object

    Returns:
        datetime or cftime object rounded to nearest minute
    """
    if t.second == 0:
        return t.replace(microsecond=0)
    elif t.second == 59:
        return t.replace(microsecond=0) + timedelta(seconds=1)
    else:
        raise ValueError(
            f"Time value > 1 second from 1 minute timesteps for "
            "C48 initialization time {t}. Are you sure you're joining "
            "the correct high res data?"
        )


def get_root():
    """Returns the absolute path to the root directory for any machine"""
    return str(TOP_LEVEL_DIR)


def get_shortened_dataset_tags():
    """"Return a dictionary mapping short dataset definitions to the full names"""
    short_dset_yaml = (
        pathlib.Path(TOP_LEVEL_DIR, "configurations") / "short_datatag_defs.yml"
    )
    return yaml.load(short_dset_yaml.open(), Loader=yaml.SafeLoader)


def parse_timestep_from_path(path: str):
    """Get the model timestep timestamp from a given path"""

    extracted_time = re.search(r"(\d\d\d\d\d\d\d\d\.\d\d\d\d\d\d)", path)

    if extracted_time is not None:
        return extracted_time.group(1)
    else:
        raise ValueError(f"No matching time pattern found in path: {path}")


def list_timesteps(path: str) -> List[str]:
    """Returns the unique timesteps at a path

    Args:
        path: local or remote path to directory containing timesteps

    Returns:
        sorted list of all timesteps within path
    """
    file_list = get_fs(path).ls(path)
    timesteps = map(parse_timestep_from_path, file_list)
    return sorted(timesteps)


root = get_root()


# TODO: I believe fv3_data_root and paths are legacy
fv3_data_root = "/home/noahb/data/2019-07-17-GFDL_FV3_DYAMOND_0.25deg_15minute/"

paths = {
    "1deg_src": f"{root}/data/interim/advection/"
    "2019-07-17-FV3_DYAMOND_0.25deg_15minute_regrid_1degree.zarr"
}


def open_catalog():
    return intake.Catalog(f"{root}/catalog.yml")


def open_dataset(tag) -> xr.Dataset:
    short_dset_tags = get_shortened_dataset_tags()
    if tag == "gfdl":
        return open_gfdl_data_with_2d(open_catalog())
    elif tag == "1degTrain":
        return (
            open_dataset("1deg")
            .merge(xr.open_zarr(paths["1deg_src"]))
            .pipe(_insert_apparent_heating)
        )
    else:
        if tag in short_dset_tags:
            # TODO: Debug logging about tag transformation
            tag = short_dset_tags[tag]

        curr_catalog = open_catalog()
        source = curr_catalog[tag]
        dataset = source.to_dask()

        for transform in source.metadata.get("data_transforms", ()):
            print(f"Applying data transform: {transform}")
            transform_function = globals()[transform]
            dataset = transform_function(dataset)

        return dataset


def open_data(sources=False, two_dimensional=True):
    path = os.path.join(
        root, "data/interim/2019-07-17-FV3_DYAMOND_0.25deg_15minute_256x256_blocks.zarr"
    )
    ds = xr.open_zarr(path)

    if sources:
        src = xr.open_zarr(root + "data/interim/apparent_sources.zarr")
        ds = ds.merge(src)

    if two_dimensional:
        data_2d = xr.open_zarr(fv3_data_root + "2d.zarr")
        ds = xr.merge([ds, data_2d], join="left")

    return ds


# Data Adjustments ##
def _rename_SHiELD_varnames_to_orig(ds: xr.Dataset) -> xr.Dataset:
    """
    Replace varnames from new dataset to match original style
    from initial DYAMOND data.
    """

    rename_list = {
        "ucomp": "u",
        "vcomp": "v",
        "sphum": "qv",
        "HGTsfc": "zs",
        "delz": "dz",
        "delp": "dp",
    }
    return ds.rename(rename_list)


def replace_esmf_coords_reg_latlon(ds: xr.Dataset) -> xr.Dataset:
    """
    Replace ESMF coordinates from regridding to a regular lat/lon grid
    """

    lat_1d = ds.lat.isel(x=0).values
    lon_1d = ds.lon.isel(y=0).values
    ds = ds.assign_coords({"x": lon_1d, "y": lat_1d})
    ds = ds.rename({"lat": "lat_grid", "lon": "lon_grid"})

    return ds


def _replace_esmf_coords(ds):
    #  Older version using DYAMOND data -AP Oct 2019
    return (
        # DYAMOND data
        ds.assign_coords(x=ds.lon.isel(y=0), y=ds.lat.isel(x=0))
        .drop(["lat", "lon"])
        .rename({"x": "lon", "y": "lat"})
    )


def _insert_apparent_heating(ds: xr.Dataset) -> xr.Dataset:
    from vcm.calc.calc import apparent_heating

    dqt_dt = ds.advection_qv + ds.storage_qv
    dtemp_dt = ds.advection_temp + ds.storage_temp
    return ds.assign(q1=apparent_heating(dtemp_dt, ds.w), q2=dqt_dt)


@delayed
def _open_remote_nc(url):
    with tempfile.NamedTemporaryFile() as fp:
        logging.info("downloading %s to disk" % url)
        subprocess.check_call((["gsutil", "-q", "cp", url, fp.name]))
        return xr.open_dataset(fp.name).load()


def file_names_for_time_step(timestep, category, resolution=3072):
    # TODO remove this hardcode
    bucket = f"gs://vcm-ml-data"
    f"2019-10-28-X-SHiELD-2019-10-05-multiresolution-extracted/"
    f"C{resolution}/{timestep}/{timestep}.{category}*"
    return gsutil.list_matches(bucket)


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
    return open_delayed(computation, schema=meta)


def _delayed_to_array(delayed_dataset, key, shape, dtype):
    null = da.full(shape, np.nan, dtype=dtype)
    array_delayed = delayed_dataset.get(key, null)
    return da.from_delayed(array_delayed, shape, dtype)


def open_delayed(delayed_dataset, schema=None):
    """Open dask delayed object with as an xarray with given metadata"""
    data_vars = {}
    for key in schema:
        template_var = schema[key]
        array = _delayed_to_array(
            delayed_dataset, key, shape=template_var.shape, dtype=template_var.dtype
        )
        data_vars[key] = (template_var.dims, array)

    return xr.Dataset(data_vars, coords=schema.coords)
