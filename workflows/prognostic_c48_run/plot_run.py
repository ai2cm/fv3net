# coding: utf-8
import os
from functools import partial
from multiprocessing import Pool

import matplotlib
import matplotlib.pyplot as plt
import vcm
import xarray as xr

matplotlib.use("Agg")


GRID_VARS = ["lon", "lat", "lonb", "latb", "time"]
rename_dims = {"x": "grid_xt", "y": "grid_yt", "rank": "tile"}


def open_ds():
    grid = xr.open_mfdataset(
        "rundir/atmos_dt_atmos.tile?.nc", concat_dim="tile", combine="nested"
    )[GRID_VARS]
    ds = xr.open_zarr("test.zarr").rename(rename_dims).merge(grid)
    return ds


def plot_axes(ds, key, **kwargs):
    mv = vcm.mappable_var(ds, key)
    return vcm.plot_cube(mv, **kwargs)


def combine_frames(prefix, pattern="%04d.png"):
    os.system(
        f"ffmpeg -y -r 15 -i {prefix}/{pattern} -vf fps=15 "
        "-pix_fmt yuv420p -s:v 1920x1080 {prefix}.mp4"
    )


def plot_frame_i(i, key, **kwargs):
    print("Saving frame %d" % i)
    ds = open_ds()
    FMT = "%y-%m-%d %H:%M:%S"
    ax = plot_axes(ds.isel(time=i), key, **kwargs)[1]
    ax.item().set_title(ds.time[i].item().strftime(FMT))
    plt.savefig(f"{key}/{i:04d}")
    plt.close()


def plot_field(pool, key, **kwargs):
    os.makedirs(key, exist_ok=False)
    ds = open_ds()
    pool.map(partial(plot_frame_i, key=key, **kwargs), range(len(ds.time)))
    combine_frames(key)


pool = Pool(16)
plot_field(pool, "net_precip", vmin=-100 / 86400, vmax=100 / 86400)
plot_field(pool, "net_heating", vmin=-20000, vmax=20000)
plot_field(pool, "PW", vmin=0, vmax=90)
