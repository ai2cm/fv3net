import os
from typing import Sequence, Mapping, cast, Hashable
from fv3net import runtime
import logging
import time

# avoid out of memory errors
# dask.config.set(scheduler='single-threaded')

import fsspec
import zarr
import xarray as xr
import numpy as np


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DELP = "pressure_thickness_of_atmospheric_layer"
TIME = "time"

TRACERS = (
    "specific_humidity",
    "cloud_water_mixing_ratio",
    "rain_mixing_ratio",
    "cloud_ice_mixing_ratio",
    "snow_mixing_ratio",
    "graupel_mixing_ratio",
    "ozone_mixing_ratio",
    "cloud_amount",
)

VARIABLES = (
    "x_wind",
    "y_wind",
    "air_temperature",
    "specific_humidity",
    "pressure_thickness_of_atmospheric_layer",
    "vertical_wind",
    "vertical_thickness_of_atmospheric_layer",
    "surface_geopotential",
    "eastward_wind_at_surface",
    "mean_cos_zenith_angle",
    "sensible_heat_flux",
    "latent_heat_flux",
    "convective_cloud_fraction",
    "convective_cloud_top_pressure",
    "convective_cloud_bottom_pressure",
    "land_sea_mask",
    "surface_temperature",
    "water_equivalent_of_accumulated_snow_depth",
    "deep_soil_temperature",
    "surface_roughness",
    "mean_visible_albedo_with_strong_cosz_dependency",
    "mean_visible_albedo_with_weak_cosz_dependency",
    "mean_near_infrared_albedo_with_strong_cosz_dependency",
    "mean_near_infrared_albedo_with_weak_cosz_dependency",
    "fractional_coverage_with_strong_cosz_dependency",
    "fractional_coverage_with_weak_cosz_dependency",
    "vegetation_fraction",
    "canopy_water",
    "fm_at_10m",
    "air_temperature_at_2m",
    "specific_humidity_at_2m",
    "vegetation_type",
    "soil_type",
    "friction_velocity",
    "fm_parameter",
    "fh_parameter",
    "sea_ice_thickness",
    "ice_fraction_over_open_water",
    "surface_temperature_over_ice_fraction",
    "total_precipitation",
    "snow_rain_flag",
    "snow_depth_water_equivalent",
    "minimum_fractional_coverage_of_green_vegetation",
    "maximum_fractional_coverage_of_green_vegetation",
    "surface_slope_type",
    "maximum_snow_albedo_in_fraction",
    "snow_cover_in_fraction",
    "soil_temperature",
    "total_soil_moisture",
    "liquid_soil_moisture",
) + TRACERS

SFC_VARIABLES = (
    "DSWRFtoa",
    "DSWRFsfc",
    "USWRFtoa",
    "USWRFsfc",
    "DLWRFsfc",
    "ULWRFtoa",
    "ULWRFsfc",
)

GRID_VARIABLES = ("lat", "lon", "latb", "lonb", "area")


def rename_sfc_dt_atmos(sfc: xr.Dataset) -> xr.Dataset:

    DIMS = {
        "grid_xt": "x",
        "grid_yt": "y",
        "grid_x": "x_interface",
        "grid_y": "y_interface",
        "time": "forecast_time",
    }

    return (
        _safe_get_variables(sfc, SFC_VARIABLES + GRID_VARIABLES)
        .rename(DIMS)
        .transpose("forecast_time", "tile", "y", "x", "y_interface", "x_interface")
        .drop(["forecast_time", "y", "x", "y_interface", "x_interface"])
    )


def align_sfc_step_ds(sfc: xr.Dataset, step_names: Sequence) -> xr.Dataset:

    realigned_sfc_vars = {
        varname: _align_sfc_step_da(sfc[varname], step_names)
        for varname in SFC_VARIABLES
    }
    sfc = sfc.drop(SFC_VARIABLES)
    sfc = sfc.assign(realigned_sfc_vars)

    return sfc


def _align_sfc_step_da(da: xr.DataArray, step_names: Sequence) -> xr.DataArray:

    da_shift = da.shift(shifts={"forecast_time": 1})
    da_list = []
    for step in step_names:
        if step != "after_physics":
            new_da = da_shift.expand_dims({"step": [step]})
        else:
            new_da = da.expand_dims({"step": [step]})
        da_list.append(new_da)

    return xr.concat(da_list, dim="step")


def init_data_var(group: zarr.Group, array: xr.DataArray, nt: int):
    logger.info(f"Initializing variable: {array.name}")
    shape = (nt,) + array.data.shape
    chunks = (1,) + tuple(size[0] for size in array.data.chunks)
    out_array = group.empty(
        name=array.name, shape=shape, chunks=chunks, dtype=array.dtype
    )
    out_array.attrs.update(array.attrs)
    out_array.attrs["_ARRAY_DIMENSIONS"] = ["initial_time"] + list(array.dims)


def init_coord(group: zarr.Group, coord):
    logger.info(f"Initializing coordinate: {coord.name}")
    # fill_value=NaN is needed below for xr.open_zarr to succesfully load this
    # coordinate if decode_cf=True. Otherwise, time=0 gets filled in as nan. very
    # confusing...
    out_array = group.array(name=coord.name, data=np.asarray(coord), fill_value="NaN")
    out_array.attrs.update(coord.attrs)
    out_array.attrs["_ARRAY_DIMENSIONS"] = list(coord.dims)


def create_zarr_store(
    timesteps: Sequence[str], group: zarr.Group, template: xr.Dataset
):
    logger.info("Creating group")
    ds = template
    group.attrs.update(ds.attrs)
    nt = len(timesteps)
    for name in ds:
        init_data_var(group, ds[name], nt)

    for name in ds.coords:
        init_coord(group, ds[name])
    dim = group.array("initial_time", data=timesteps)
    dim.attrs["_ARRAY_DIMENSIONS"] = ["initial_time"]


def _get_forecast_time(time) -> xr.DataArray:
    dt = np.asarray(time - time[0])
    return xr.DataArray(
        _convert_time_delta_to_float_seconds(dt),
        name="time",
        dims=["time"],
        attrs={"units": "s"},
    )


def _convert_time_delta_to_float_seconds(a):
    ns_per_s = 1e9
    return a.astype("timedelta64[ns]").astype(float) / ns_per_s


def _merge_monitor_data(paths: Mapping[str, str]) -> xr.Dataset:
    datasets = {key: xr.open_zarr(val) for key, val in paths.items()}
    time = _get_forecast_time(datasets["begin"].time)
    datasets_no_time = [val.drop("time") for val in datasets.values()]
    steps = list(datasets.keys())
    return xr.concat(datasets_no_time, dim="step").assign_coords(step=steps, time=time)


def _write_to_store(group: zarr.ABSStore, index: int, ds: xr.Dataset):
    for variable in ds:
        logger.info(f"Writing {variable} to {group}")
        dims = group[variable].attrs["_ARRAY_DIMENSIONS"][1:]
        dask_arr = ds[variable].transpose(*dims).data
        dask_arr.store(group[variable], regions=(index,))


def _safe_get_variables(ds: xr.Dataset, variables: Sequence[Hashable]) -> xr.Dataset:
    """ds[...] is very confusing function from a typing perspective and should be
    avoided in long-running pipeline codes. This function introduces a type-stable
    alternative that works better with mypy.

    In particular, ds[('a' , 'b' ,'c')] looks for a variable named ('a', 'b', 'c') which
    usually doesn't exist, so it causes a key error. but ds[['a', 'b', 'c']] makes a
    dataset only consisting of the variables 'a', 'b', and 'c'. This causes tons of
    hard to find errors.
    """
    variables = list(variables)
    return cast(xr.Dataset, ds[variables])


def _zarr_safe_string_coord(ds: xr.Dataset, coord_name: str) -> xr.Dataset:
    """Ensures an xr.Dataset coordinate is of dtype "<U14" instead of object,
    which is necessary to do for string coordinates so they they may be written
    to zarr arrays without an object codec.
    """

    return ds.assign_coords({coord_name: ds[coord_name].values.astype("<U14")})


def post_process(
    monitor_paths: Mapping[str, str],
    sfc_pattern: str,
    store_url: str,
    index: int,
    init: bool = False,
    timesteps: Sequence = (),
):

    if init and len(timesteps) > 0 and index:
        raise ValueError(
            f"To initialize the zarr store, {timesteps} must not be empty."
        )
    logger.info("Post processing model outputs")

    sfc = (
        xr.open_mfdataset(sfc_pattern, concat_dim="tile", combine="nested")
        .pipe(rename_sfc_dt_atmos)
        .pipe(align_sfc_step_ds, monitor_paths.keys())
    )

    ds = (
        _merge_monitor_data(monitor_paths)
        .rename({"time": "forecast_time"})
        .chunk({"forecast_time": 1, "tile": 6, "step": 3})
    )

    merged = xr.merge([sfc, ds])
    merged = _zarr_safe_string_coord(merged, coord_name="step")
    mapper = fsspec.get_mapper(store_url)

    if init:
        logging.info("initializing zarr store")
        group = zarr.open_group(mapper, mode="w")
        create_zarr_store(timesteps, group, merged)

    group = zarr.open_group(mapper, mode="a")
    _write_to_store(group, index, merged)


if __name__ == "__main__":
    import fv3gfs
    from mpi4py import MPI

    RUN_DIR = os.path.dirname(os.path.realpath(__file__))

    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    current_dir = os.getcwd()
    config = runtime.get_config()
    MPI.COMM_WORLD.barrier()  # wait for master rank to write run directory
    logger = logging.getLogger(f"one_step:{rank}/{size}:{config['one_step']['index']}")

    partitioner = fv3gfs.CubedSpherePartitioner.from_namelist(config["namelist"])

    sfc_pattern = f"{RUN_DIR}/sfc_dt_atmos.tile?.nc"
    paths = dict(
        begin=os.path.join(RUN_DIR, "before_physics.zarr"),
        after_physics=os.path.join(RUN_DIR, "after_physics.zarr"),
        after_dynamics=os.path.join(RUN_DIR, "after_dynamics.zarr"),
    )

    monitors = {
        key: fv3gfs.ZarrMonitor(path, partitioner, mode="w", mpi_comm=MPI.COMM_WORLD)
        for key, path in paths.items()
    }

    fv3gfs.initialize()
    state = fv3gfs.get_state(names=VARIABLES + (TIME,))
    if rank == 0:
        logger.info("Beginning steps")
    for i in range(fv3gfs.get_step_count()):
        if rank == 0:
            logger.info(f"step {i}")
        monitors["begin"].store(state)
        fv3gfs.step_dynamics()
        state = fv3gfs.get_state(names=VARIABLES + (TIME,))
        monitors["after_dynamics"].store(state)
        fv3gfs.step_physics()
        state = fv3gfs.get_state(names=VARIABLES + (TIME,))
        monitors["after_physics"].store(state)

    # parallelize across variables
    fv3gfs.cleanup()
    del monitors

    if rank == 0:
        # TODO it would be much cleaner to call this is a separate script, but that
        # would be incompatible with the run_k8s api
        # sleep a little while to allow all process to finish finalizing the netCDFs
        time.sleep(2)
        c = config["one_step"]
        url = c.pop("url")
        index = c.pop("index")
        post_process(paths, sfc_pattern, url, index, **c)
else:
    logger = logging.getLogger(__name__)
