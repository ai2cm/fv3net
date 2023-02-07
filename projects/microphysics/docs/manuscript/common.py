import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import functools
import numpy as np
import os
import xarray
import vcm.catalog
import vcm
import intake
import xarray as xr
from fv3net.diagnostics.prognostic_run import load_run_data
from fv3net.diagnostics.prognostic_run.emulation import query
import wandb
import joblib

WIDTH = 5.1
VERSION = "1"
plt.style.use("seaborn-colorblind")
MEMOIZE_DIR = "./cache"


land_color = "#bcb3a2"
RdBu_LAND = plt.get_cmap("RdBu_r")
RdBu_LAND.set_bad(land_color)

Viridis_LAND = plt.get_cmap("viridis")
Viridis_LAND.set_bad(land_color)


def _insert_white(key, new_key, position, bad_color=land_color):
    tmp = plt.get_cmap(key)(np.linspace(0, 1, 8))
    tmp[position] = [1.0, 1.0, 1.0, 1.0]
    cmap = LinearSegmentedColormap.from_list(new_key, tmp)
    cmap.set_bad(bad_color)

    return cmap


NuReds_LAND = _insert_white("Reds", "NuReds", 0)
NuBlues_LAND = _insert_white("Blues_r", "NuBlues", -1)

dt = 900  # s
seconds_per_day = 60 * 60 * 24  # seconds/min * min/hr * hr/day
m_to_mm = 1000

# from physcons.f
cp = 1.0046e3  # J / (kg K)
gravity = 9.80665  # m / s^2
lv = 2.5e6  # J / kg water
rho_water = 1000.0  # kg / m^3

levels = vcm.interpolate.PRESSURE_GRID
x = range(0, 70, 2)
x2 = range(0, 70)
yp = np.interp(x2, x, levels)
LEVELS = xr.DataArray(yp[:-1], dims=["pressure"])


def kg_m2_s_to_mm_day(da):
    return da * seconds_per_day / rho_water * m_to_mm


def kg_m2_to_mm(da):
    return da / rho_water * m_to_mm


def m_to_mm_day(da):
    return da / dt * m_to_mm * seconds_per_day


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


def interp_vertical(ds, levels=vcm.interpolate.PRESSURE_GRID):
    ds_interp = xarray.Dataset()
    pressure_vars = [var for var in ds.data_vars if "z" in ds[var].dims]
    for var in pressure_vars:
        ds_interp[var] = vcm.interpolate_to_pressure_levels(
            field=ds[var],
            delp=ds["pressure_thickness_of_atmospheric_layer"],
            dim="z",
            levels=levels,
        )

    return ds_interp


def open_prognostic_data(url, catalog):

    files = [
        "state_after_timestep.zarr",
        "piggy.zarr",
        "sfc_dt_atmos.zarr",
        "atmos_dt_atmos.zarr",
    ]

    ds = xarray.Dataset()
    for f in files:
        full_path = os.path.join(url, f)
        ds = ds.merge(
            load_run_data.load_coarse_data(full_path, catalog), compat="override"
        )

    return ds


def open_group(group):
    # open data
    url = get_group_url(group)
    catalog_path = vcm.catalog.catalog_path
    catalog = intake.open_catalog(catalog_path)
    grid = load_run_data.load_grid(catalog)
    prognostic = open_prognostic_data(url, catalog)
    data = prognostic.merge(grid, compat="override")
    return data


def get_group_url(group):
    client = query.PrognosticRunClient(
        group, project="microphysics-emulation", entity="ai2cm", api=wandb.Api()
    )
    return client.get_rundir_url()


def savefig(filename):
    plt.savefig("figs/" + filename + ".png", bbox_inches="tight")
    plt.savefig("figs/" + filename + ".pdf", bbox_inches="tight")


def meridional_transect(ds: xarray.Dataset, lon):
    transect_coords = vcm.select.meridional_ring(lon)
    ds = vcm.interpolate_unstructured(ds, transect_coords)
    return ds.swap_dims({"sample": "lat"})
