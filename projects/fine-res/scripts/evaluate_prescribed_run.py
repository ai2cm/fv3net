"""Assuming prognostic run has diagnostics output according to something like
https://github.com/ai2cm/vcm-workflow-control/blob/2021-11-05-redo-n2f-prescribe
-and-baseline-runs-with-more-diags/examples/nudge-to-fine-run/nudging-config.yaml"""
import os
import click
import xarray as xr
import fsspec
from dask.distributed import Client
import intake
import matplotlib.pyplot as plt
from matplotlib import cycler

import fv3viz
import vcm
from vcm.catalog import catalog_path

DEFAULT_RUN_URL = (
    "gs://vcm-ml-experiments/default/2021-11-08/"
    "n2f-prescribe-q1-q2-more-diags/fv3gfs_run"
)
DIAGS_2D_NAME = "diags.zarr"
PHYSICS_TENDENCIES_NAME = "physics_tendencies.zarr"
NUDGING_TENDENCIES_NAME = "nudging_tendencies.zarr"
STATE_NAME = "state_after_timestep.zarr"
VARIABLES = ["air_temperature", "specific_humidity"]
Z_INDICES = [25, 78]
TIME_INDEX = 95
COLORS = [
    "#000000",
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
]
TENDENCY_NAMES = {
    "air_temperature": {
        "tendency": {
            "fine-res": "tendency_of_air_temperature_due_to_applied_physics",
            "dynamics": "tendency_of_air_temperature_due_to_dynamics",
            "nudging": "air_temperature_tendency_due_to_nudging",
        },
        "path_storage": {
            "fine-res": "storage_of_internal_energy_path_due_to_applied_physics",
            "dynamics": "storage_of_internal_energy_path_due_to_dynamics",
            "nudging": "column_heating_due_to_nudging",
        },
    },
    "specific_humidity": {
        "tendency": {
            "fine-res": "tendency_of_specific_humidity_due_to_applied_physics",
            "dynamics": "tendency_of_specific_humidity_due_to_dynamics",
            "nudging": "specific_humidity_tendency_due_to_nudging",
        },
        "path_storage": {
            "fine-res": "storage_of_specific_humidity_path_due_to_applied_physics",
            "dynamics": "storage_of_specific_humidity_path_due_to_dynamics",
            "nudging": "net_moistening_due_to_nudging",
        },
    },
}

default_cycler = cycler(color=COLORS)
plt.rc("axes", prop_cycle=default_cycler)


def organize_by_process(ds):
    data_by_process = []
    for process in ["fine-res", "dynamics", "nudging"]:
        tmp = xr.Dataset()
        for variable in ["air_temperature", "specific_humidity"]:
            for type_ in ["tendency", "path_storage"]:
                name = TENDENCY_NAMES[variable][type_][process]
                tmp[f"{variable}_{type_}"] = ds[name].assign_coords(process=process)
        data_by_process.append(tmp)
    return xr.concat(data_by_process, dim="process")


def add_suffix(ds, suffix):
    names = list(ds.data_vars)
    return ds.rename({k: f"{k}{suffix}" for k in names})


def adjust_units(ds):
    for variable in ds:
        if "air_temperature_tendency" in variable:
            ds[variable] = (86400 * ds[variable]).assign_attrs(
                units="K/day", long_name="air temperature tendency"
            )
        elif "specific_humidity_tendency" in variable:
            ds[variable] = (1000 * 86400 * ds[variable]).assign_attrs(
                units="g/kg/day", long_name="specific humidity tendency"
            )
        elif "air_temperature_path_storage" in variable:
            ds[variable] = ds[variable].assign_attrs(
                units="W/m**2", long_name="column integrated air temperature tendency"
            )
        elif "specific_humidity_path_storage" in variable:
            ds[variable] = (86400 * ds[variable]).assign_attrs(
                units="mm/day", long_name="column integrated specific humidity tendency"
            )


def plot_profile(ds, variable, output):
    da = ds[variable]
    path_variable = variable.replace("tendency", "path_storage")
    fg = da.plot(col="region", y="z", yincrease=False, hue="process", col_wrap=3)
    axes = fg.axes.flat
    for i in range(ds.sizes["region"]):
        integrals = ds[path_variable].isel(region=i).values
        units = ds[path_variable].units
        text_kwargs = dict(fontsize=8, transform=axes[i].transAxes)
        axes[i].text(0.05, 0.9, f"Integral ({units})", **text_kwargs)
        for j, val in enumerate(integrals):
            axes[i].text(
                0.05, 0.84 - 0.06 * j, f"{val:.2f}", **text_kwargs, color=COLORS[j]
            )
    fg.fig.savefig(output, dpi=150)
    plt.close(fg.fig)


def plot_map(ds, name, output):
    fg = fv3viz.plot_cube(ds, name, col="process", col_wrap=2)[-1]
    fg.fig.set_size_inches((7, 3.8))
    if "snapshot" in name:
        time = ds[name].time.values.item()
    elif "time_mean" in name:
        time = "time-mean"
    else:
        time = ""
    fg.fig.suptitle(time)
    fg.fig.savefig(output, dpi=200)
    plt.close(fg.fig)


def time_mean(ds):
    start = ds.time.values[0]
    end = ds.time.values[-1]
    return ds.mean("time").assign_coords(time=f"{start} to {end} mean")


def compute(run_url, output):
    catalog = intake.open_catalog(catalog_path)
    data = []
    for name in [
        DIAGS_2D_NAME,
        PHYSICS_TENDENCIES_NAME,
        NUDGING_TENDENCIES_NAME,
        STATE_NAME,
    ]:
        data.append(xr.open_zarr(fsspec.get_mapper(os.path.join(run_url, name))))
    grid = catalog["grid/c48"].to_dask().load()
    data.append(grid)
    data = xr.merge(data).assign_coords(z=range(data.sizes["z"]))

    # create 'process' dimension in data
    data_by_process = organize_by_process(data)
    sum_ = data_by_process.sum("process").assign_coords(process="sum")
    data_by_process = xr.concat([sum_, data_by_process], dim="process")

    # generate some masks
    column_q2_name = "storage_of_specific_humidity_path_due_to_applied_physics"
    masks = {
        "global": grid.area,
        "ocean": grid.area.where(data.land_sea_mask.astype("int") != 1),
        "land": grid.area.where(data.land_sea_mask.astype("int") == 1),
        "P-E>0": grid.area.where(data[column_q2_name] < 0),
        "P-E<0": grid.area.where(data[column_q2_name] > 0),
    }

    # compute means over regions
    region_mean = []
    for name, area in masks.items():
        reduced = vcm.weighted_average(data_by_process, area)
        region_mean.append(reduced.assign_coords(region=name))
    region_mean = xr.concat(region_mean, dim="region")
    region_mean = add_suffix(region_mean, "_region_mean")
    region_mean_time_mean = region_mean.mean("time")
    region_mean_time_mean = add_suffix(region_mean_time_mean, "_time_mean")

    # generate dataset of 2d fields (column integral and specific levels)
    storage_path_names = [f"{v}_path_storage" for v in VARIABLES]
    tendency_path_names = [f"{v}_tendency" for v in VARIABLES]
    data_2d = data_by_process[storage_path_names]
    for z in Z_INDICES:
        data_level = data_by_process[tendency_path_names].isel(z=z).drop("z")
        data_level = add_suffix(data_level, f"_z{z}")
        data_2d = xr.merge([data_2d, data_level])

    # compute time mean and select a snapshot from 2D data
    data_snapshot = add_suffix(data_2d.isel(time=TIME_INDEX), "_snapshot")
    data_time_mean = add_suffix(time_mean(data_2d), "_time_mean")

    # merge and compute
    diagnostics = xr.merge([region_mean_time_mean, data_snapshot, data_time_mean, grid])
    diagnostics = diagnostics.compute()
    adjust_units(diagnostics)

    with fsspec.open(output, "wb") as f:
        vcm.dump_nc(diagnostics, f)


def plot(ds: xr.Dataset, output: str):
    horizontal = []
    profile = []
    for variable in ds:
        dims = set(ds[variable].dims)
        if {"x", "y", "tile", "process"}.issubset(dims):
            horizontal.append(variable)
        if {"z", "process"}.issubset(dims) and "x" not in dims:
            profile.append(variable)

    for variable in profile:
        plot_profile(ds, variable, os.path.join(output, f"profile_{variable}.png"))

    for variable in horizontal:
        plot_map(ds, variable, os.path.join(output, f"map_{variable}.png"))


@click.command()
@click.argument("run_url", type=str, default=DEFAULT_RUN_URL)
@click.argument("output", type=str, default=".")
def evaluate(run_url, output):
    Client()
    diag_output = os.path.normpath(os.path.join(output, "diags.nc"))
    compute(run_url, diag_output)
    with fsspec.open(diag_output) as f:
        ds = xr.open_dataset(f, engine="h5netcdf").load()
    plot(ds, output)


if __name__ == "__main__":
    evaluate()
