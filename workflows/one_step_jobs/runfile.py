import os
from fv3net import runtime
import fsspec
import zarr
import xarray as xr
import numpy as np
import logging


logging.basicConfig(level=logging.INFO)

DELP = "pressure_thickness_of_atmospheric_layer"
TIME = "time"

VARIABLES = ( 
    "x_wind",
    "y_wind",
    "air_temperature",
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
    "liquid_soil_moisture"
 )



def init_data_var(group, array, nt):
    logger.info(f"Initializing variable: {array.name}")
    shape = (nt,) + array.data.shape
    chunks = (1,) + tuple(size[0] for size in array.data.chunks)
    out_array = group.empty(
        name=array.name, shape=shape, chunks=chunks, dtype=array.dtype
    )
    out_array.attrs.update(array.attrs)
    out_array.attrs["_ARRAY_DIMENSIONS"] = ["initial_time"] + list(array.dims)


def init_coord(group, coord):
    logger.info(f"Initializing coordinate: {coord.name}")
    out_array = group.array(name=coord.name, data=np.asarray(coord))
    out_array.attrs.update(coord.attrs)
    out_array.attrs["_ARRAY_DIMENSIONS"] = list(coord.dims)


def create_zarr_store(timesteps, group, template):
    logger.info("Creating group")
    ds = template
    group.attrs.update(ds.attrs)
    nt = len(timesteps)
    for name in ds:
        init_data_var(group, ds[name], nt)

    for name in ds.coords:
        init_coord(group, ds[name])
    dim = group.array('initial_time', data=timesteps)
    dim.attrs['_ARRAY_DIMENSIONS'] = ['initial_time']


def post_process(out_dir, url, index, init=False, timesteps=(), comm=None):
    store_url = url
    logger.info("Post processing model outputs")
    begin = xr.open_zarr(f"{out_dir}/begin_physics.zarr")
    before = xr.open_zarr(f"{out_dir}/before_physics.zarr")
    after = xr.open_zarr(f"{out_dir}/after_physics.zarr")

    # make the time dims consistent
    time = begin.time
    before = before.drop("time")
    after = after.drop("time")
    begin = begin.drop("time")

    # concat data
    dt = np.timedelta64(15, "m")
    time = np.arange(len(time)) * dt
    ds = xr.concat([begin, before, after], dim="step").assign_coords(
        step=["begin", "after_dynamics", "after_physics"], time=time
    )
    ds = ds.rename({"time": "forecast_time"}).chunk({"forecast_time": 1, "tile": 6})

    if comm is not None:
        rank = comm.Get_rank()
    else:
        rank = 0

    mapper = fsspec.get_mapper(store_url)
    if init and rank == 0:
        group = zarr.open_group(mapper, mode="w")
        create_zarr_store(timesteps, group, ds)
            
    if comm is None:
        variables = VARIABLES
    else:
        comm.barrier()
        variables = list(VARIABLES)[comm.rank::comm.size]

    # all processes open group
    group = zarr.open_group(mapper, mode="a")
    logger.info(f"Variables to process: {variables}")
    for variable in ds[list(variables)]:
        logger.info(f"Writing {variable} to {group}")
        dims = group[variable].attrs["_ARRAY_DIMENSIONS"][1:]
        dask_arr = ds[variable].transpose(*dims).data
        dask_arr.store(group[variable][index])


if __name__ == "__main__":
    import fv3gfs
    from mpi4py import MPI

    RUN_DIR = os.path.dirname(os.path.realpath(__file__))

    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    logger = logging.getLogger(__file__ + f"({rank}/{size})")
    current_dir = os.getcwd()
    config = runtime.get_config()
    MPI.COMM_WORLD.barrier()  # wait for master rank to write run directory

    partitioner = fv3gfs.CubedSpherePartitioner.from_namelist(config["namelist"])

    before_monitor = fv3gfs.ZarrMonitor(
        os.path.join(RUN_DIR, "before_physics.zarr"),
        partitioner,
        mode="w",
        mpi_comm=MPI.COMM_WORLD,
    )

    after_monitor = fv3gfs.ZarrMonitor(
        os.path.join(RUN_DIR, "after_physics.zarr"),
        partitioner,
        mode="w",
        mpi_comm=MPI.COMM_WORLD,
    )

    begin_monitor = fv3gfs.ZarrMonitor(
        os.path.join(RUN_DIR, "begin_physics.zarr"),
        partitioner,
        mode="w",
        mpi_comm=MPI.COMM_WORLD,
    )

    fv3gfs.initialize()
    state = fv3gfs.get_state(names=VARIABLES + (TIME,))
    if rank == 0:
        logger.info("Beginning steps")
    for i in range(fv3gfs.get_step_count()):
        if rank == 0:
            logger.info(f"step {i}")
        begin_monitor.store(state)
        fv3gfs.step_dynamics()
        state = fv3gfs.get_state(names=VARIABLES + (TIME,))
        before_monitor.store(state)
        fv3gfs.step_physics()
        state = fv3gfs.get_state(names=VARIABLES + (TIME,))
        after_monitor.store(state)

    MPI.COMM_WORLD.barrier()
    # parallelize across variables
    if rank == 0:
        post_process(RUN_DIR, **config["one_step"], comm=None)
    fv3gfs.cleanup()
