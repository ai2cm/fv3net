from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import shutil
from typing import Sequence, List
import xarray as xr
import vcm
from vcm.cloud import gsutil
from vcm.convenience import round_time

TIME_NAME = "time"
TIME_FMT = "%Y%m%d.%H%M%S"


def copy_outputs(temp_dir, output_dir):
    if output_dir.startswith("gs://"):
        gsutil.copy(temp_dir, output_dir)
    else:
        shutil.copytree(temp_dir, output_dir)
        

def time_series(
    da: xr.DataArray,
    grid: xr.Dataset,
    title=None,
    units=None,
    vmin=None,
    vmax=None,
    cmap=None
):  
    
    units = units or da.units or ""
    title = title or f"{da.name} [{units}]"
    mean_dims = [dim for dim in ["tile", "x", "y"] if dim in da.dims]
    da = da.mean(mean_dims)
    fig = plt.figure()
    if "z" in da.dims:
        da.plot(x="time", vmin=vmin or None, vmax=vmax or None, cmap=cmap or "RdBu_r")
        plt.ylabel("model level")
        plt.gca().invert_yaxis()
    else:
        da.plot()
    plt.title(title)
    return fig


def standardize_zarr_time_coord(ds: xr.Dataset) -> xr.Dataset:
    """ Casts a datetime coord to to python datetime and rounds to
    nearest even second (because cftime coords have small rounding
    errors that makes it hard to other datasets join on time)

    Args:
        ds (xr.Dataset): time coordinate is datetime-like object

    Returns:
        xr.Dataset with standardized time coordinates
    """
    # Vectorize doesn't work on type-dispatched function overloading
    times = np.array(list(map(vcm.cast_to_datetime, ds[TIME_NAME].values)))
    times = np.vectorize(round_time)(times)
    ds = ds.assign_coords({TIME_NAME: times})
    return ds


def dataset_from_timesteps(mapper, keys: Sequence[str], vars: List[str]):
    time_coords = [datetime.strptime(key, TIME_FMT) for key in keys]
    ds = xr.concat([mapper[key][vars] for key in keys], pd.Index(time_coords, name="time"))
    return ds