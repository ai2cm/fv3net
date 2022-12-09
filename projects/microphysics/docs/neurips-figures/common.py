import matplotlib.pyplot as plt
import functools
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
plt.style.use("matplotlibrc")


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


def savefig(filename):
    plt.savefig("figs/" + filename + ".png", bbox_inches="tight")
    plt.savefig("figs/" + filename + ".pdf", bbox_inches="tight")
