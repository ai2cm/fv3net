import matplotlib.pyplot as plt
import functools
import os
import xarray
import vcm.catalog
import vcm
import intake
from fv3net.diagnostics.prognostic_run import load_run_data
from fv3net.diagnostics.prognostic_run.emulation import query
import wandb
import joblib

WIDTH = 5.1
VERSION = "1"
plt.style.use("seaborn-colorblind")
MEMOIZE_DIR = "./cache"


def memoize_xarray_out(func):
    @functools.wraps(func)
    def myfunc(*args, **kwargs):
        hash = joblib.hash((args, kwargs))
        file_name = func.__name__ + "-" + VERSION + "-" + hash + ".nc"
        path = os.path.join(MEMOIZE_DIR, file_name)
        try:
            return xarray.open_dataset(path)
        except FileNotFoundError:
            ds = func(*args, **kwargs)
            ds.to_netcdf(path)
            return ds

    return myfunc


def open_prognostic_data(url, catalog):

    files = ["state_after_timestep.zarr", "piggy.zarr"]

    ds = xarray.Dataset()
    for f in files:
        full_path = os.path.join(url, f)
        ds = ds.merge(
            load_run_data.load_coarse_data(full_path, catalog), compat="override"
        )

    return ds


def open_group(group):
    # open data
    client = query.PrognosticRunClient(
        group, project="microphysics-emulation", entity="ai2cm", api=wandb.Api()
    )
    url = client.get_rundir_url()
    catalog_path = vcm.catalog.catalog_path
    catalog = intake.open_catalog(catalog_path)
    grid = load_run_data.load_grid(catalog)
    prognostic = open_prognostic_data(url, catalog)
    data = prognostic.merge(grid, compat="override")
    return data


def savefig(filename):
    plt.savefig("figs/" + filename + ".png", bbox_inches="tight")
    plt.savefig("figs/" + filename + ".pdf", bbox_inches="tight")


def meridional_transect(ds: xarray.Dataset, lon):
    transect_coords = vcm.select.meridional_ring(lon)
    ds = vcm.interpolate_unstructured(ds, transect_coords)
    return ds.swap_dims({"sample": "lat"})
