# flake8: noqa
# %%
import matplotlib.pyplot as plt

plt.style.use("matplotlibrc")
from matplotlib import ticker
import numpy as np
from matplotlib import colors
import functools
import xarray
import vcm.catalog
import vcm
import intake
from fv3net.diagnostics.prognostic_run import load_run_data
from fv3net.diagnostics.prognostic_run.emulation import query
import fv3viz
from cartopy import crs
import wandb
from toolz import memoize
import joblib
import os

WIDTH = 5.1

# increment to manually invalidate cache
VERSION = "1"

os.makedirs("figs", exist_ok=True)


def savefig(filename):
    plt.savefig("figs/" + filename + ".png", bbox_inches="tight")
    plt.savefig("figs/" + filename + ".pdf", bbox_inches="tight")


def memoize_xarray_out(func):
    @functools.wraps(func)
    def myfunc(*args, **kwargs):
        hash = joblib.hash((args, kwargs))
        file_name = func.__name__ + "-" + VERSION + "-" + hash + ".nc"
        try:
            return xarray.open_dataset(file_name)
        except FileNotFoundError:
            ds = func(*args, **kwargs)
            ds.to_netcdf(file_name)
            return ds

    return myfunc


def open_group(group):
    # open data
    client = query.PrognosticRunClient(
        group, project="microphysics-emulation", entity="ai2cm", api=wandb.Api()
    )
    url = client.get_rundir_url()
    catalog_path = vcm.catalog.catalog_path
    catalog = intake.open_catalog(catalog_path)
    grid = load_run_data.load_grid(catalog)
    prognostic = load_run_data.SegmentedRun(url, catalog)
    data_3d = prognostic.data_3d.merge(grid)
    data_2d = grid.merge(prognostic.data_2d, compat="override")
    return data_2d, data_3d


@memoize_xarray_out
def get_zonal_avg(group, field):
    _, data_3d = open_group(group)
    # data is already on pressure levels
    return vcm.zonal_average_approximate(data_3d.lat, data_3d[field]).load()


def figure_zonal():
    def label(ax):
        ax.set_xlabel("latitude")
        ax.set_ylabel("pressure (mb)")

    def avg(x):
        return x.sel(time=slice("2016-07-20", "2016-07-31")).mean("time")

    M = 50
    n = 5

    group = "precpd-diff-only-rnn-combined-cloudlimit-v5-online"
    cloud_avg = get_zonal_avg(group, "cloud_water_mixing_ratio")

    group = "precpd-diff-only-rnn-combined-cloudlimit-v5-offline"
    cloud_avg_offline = get_zonal_avg(group, "cloud_water_mixing_ratio")

    bias = cloud_avg - cloud_avg_offline

    plt.figure(figsize=(WIDTH / 2, WIDTH / 2 / 1.61))
    z = avg(cloud_avg_offline.cloud_water_mixing_ratio)
    z["pressure"] = z["pressure"] / 100
    z *= 1e6
    z.plot(
        cmap=plt.get_cmap("Blues", 10),
        vmax=M,
        vmin=0,
        yincrease=False,
        add_labels=False,
        rasterized=True,
    )
    plt.title("a) Truth", loc="left")
    label(plt.gca())
    savefig("zonal-a")

    plt.figure(figsize=(WIDTH / 2, WIDTH / 2 / 1.61))
    z = avg(bias.cloud_water_mixing_ratio)
    z["pressure"] = z["pressure"] / 100
    z = z * 1e6
    z.plot(
        yincrease=False,
        vmax=M,
        vmin=-M,
        cmap=plt.get_cmap("RdBu", 10),
        add_labels=False,
        rasterized=True,
    )
    plt.title("b) Bias", loc="left")
    label(plt.gca())
    savefig("zonal-b")

    print(
        "Zonal average of cloud water mixing ratio (mg/kg) from the truth (a) and the bias of the emulation run (b). The average is computed from July 20 to July 31, inclusive. Pressure is commonly used vertical coordinate in atmospheric science.",
    )


figure_zonal()
