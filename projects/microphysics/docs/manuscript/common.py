import json
import re
from typing import Union
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import functools
import numpy as np
import os
import pandas as pd
import xarray
import vcm.catalog
import vcm
import intake
import xarray as xr
from joblib import delayed, Parallel
from fv3net.diagnostics.prognostic_run import load_run_data
from fv3net.diagnostics.prognostic_run.emulation import query
import wandb
import joblib

API = wandb.Api()

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
        except (FileNotFoundError, ValueError):
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


def get_rh_from_ds(ds):
    delp_key = "pressure_thickness_of_atmospheric_layer"
    pressure = ds[delp_key].cumsum(dim="z") + 300.0
    humidity = ds["specific_humidity"]
    temperature = ds["air_temperature"]
    return vcm.relative_humidity_from_pressure(temperature, humidity, pressure)


def get_upper_and_lower_bounds(data, upper_percentile=97.5, lower_percentile=2.5):
    ub = np.percentile(data, upper_percentile, axis=0)
    lb = np.percentile(data, lower_percentile, axis=0)

    return ub, lb


def bootstrap_metric_and_signif(metric, n_iters=10_000):
    metric = metric.flatten()

    def _resample_func():
        return np.random.choice(metric, size=metric.shape[0], replace=True).mean()

    jobs = [delayed(_resample_func)() for i in range(n_iters)]
    resampled_metrics = Parallel(n_jobs=8)(jobs)

    ub, lb = get_upper_and_lower_bounds(resampled_metrics)
    signif = np.logical_or(
        np.logical_and(lb < 0, ub < 0), np.logical_and(lb > 0, ub > 0)
    )
    return resampled_metrics, signif


def bootstrap_signif_two_metrics(metric1, metric2, n_iters=10_000):
    resampled_metric1, _ = bootstrap_metric_and_signif(metric1, n_iters=n_iters)
    resampled_metric2, _ = bootstrap_metric_and_signif(metric2, n_iters=n_iters)

    metric1_ub, metric1_lb = get_upper_and_lower_bounds(resampled_metric1)
    metric2_ub, metric2_lb = get_upper_and_lower_bounds(resampled_metric2)

    signif = np.logical_or(metric1_ub < metric2_lb, metric2_ub < metric1_lb)
    return (resampled_metric1, resampled_metric2), signif


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


def open_12init_from_group_prefix(prefix):

    months = [f"{i:02d}" for i in range(1, 13)]
    groups = [prefix.format(init=m) for m in months]

    def _open_group(group):
        offline = open_group(f"{group}-offline")
        online = open_group(f"{group}-online")
        return offline, online

    jobs = [delayed(_open_group)(group) for group in groups]
    results = Parallel(n_jobs=12)(jobs)

    opened_map = {group: result for group, result in zip(groups, results)}
    return opened_map


def get_group_url(group):
    client = query.PrognosticRunClient(
        group, project="microphysics-emulation", entity="ai2cm", api=API
    )
    return client.get_rundir_url()


def _get_runs(group, job_types=None):
    runs = API.runs(filters={"group": group}, path="ai2cm/microphysics-emulation")
    if job_types is not None:
        runs = [r for r in runs if r.job_type in job_types]
    return runs


def _get_air_temp_plotly_file(run):
    match_str = ".*time_vs_lev.*gscond.*air_temperature"
    for f in run.files():
        if re.search(match_str, f.name) is not None:
            return f.download(replace=True)


def _plotly_2d_data_to_array(f):
    json_payload = json.loads(str(f.read()))
    data = json_payload["data"][0]
    data = {k: data[k] for k in ["x", "y", "z"]}
    data = {k: np.array(v) for k, v in data.items()}

    return data


def get_skill_arr_from_group(group):
    run = _get_runs(group, {"piggy-back"})
    plotly_file = _get_air_temp_plotly_file(run)
    return _plotly_2d_data_to_array(plotly_file)


def limit_sigfigs(df: Union[pd.DataFrame, pd.Series], num_sigfigs=3):
    power_offset = -np.floor(np.log10(np.abs(df))).astype(int)
    to_round = df * 10.0 ** power_offset
    return to_round.round(num_sigfigs - 1) * 10.0 ** -power_offset


def savefig(filename):
    plt.savefig("figs/" + filename + ".png", bbox_inches="tight")
    plt.savefig("figs/" + filename + ".pdf", bbox_inches="tight")


def meridional_transect(ds: xarray.Dataset, lon):
    transect_coords = vcm.select.meridional_ring(lon)
    ds = vcm.interpolate_unstructured(ds, transect_coords)
    return ds.swap_dims({"sample": "lat"})
