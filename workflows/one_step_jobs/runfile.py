import os
from fv3net import runtime
import fsspec
import zarr
import xarray as xr
import numpy as np
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)


def _compute_chunks(shape, chunks):
    return tuple(size if chunk == -1 else chunk for size, chunk in zip(shape, chunks))


def _get_schema(shape=(3, 15, 6, 79, 48, 48)):
    variables = [
        "air_temperature",
        "specific_humidity",
        "pressure_thickness_of_atmospheric_layer",
    ]
    dims_scalar = ["step", "forecast_time", "tile", "z", "y", "x"]
    chunks_scalar = _compute_chunks(shape, [-1, 1, -1, -1, -1, -1])
    DTYPE = np.float32
    scalar_schema = {
        "dims": dims_scalar,
        "chunks": chunks_scalar,
        "dtype": DTYPE,
        "shape": shape,
    }
    return {key: scalar_schema for key in variables}


def _init_group_with_schema(group, schemas, timesteps):
    for name, schema in schemas.items():
        shape = (len(timesteps),) + schema["shape"]
        chunks = (1,) + schema["chunks"]
        array = group.empty(name, shape=shape, chunks=chunks, dtype=schema["dtype"])
        array.attrs.update({"_ARRAY_DIMENSIONS": ["initial_time"] + schema["dims"]})


def init_data_var(group, array):
    logger.info(f"Initializing coordinate: {array.name}")
    shape = (1,) + array.data.shape
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
    for name in ds:
        init_data_var(group, ds[name])

    for name in ds.coords:
        init_coord(group, ds[name])


def post_process(out_dir, url, index, init=False, timesteps=()):

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

    mapper = fsspec.get_mapper(store_url)
    group = zarr.open_group(mapper, mode="a")

    if init:
        group = zarr.open_group(mapper, mode="w")
        create_zarr_store(timesteps, group, ds)

    for variable in ds:
        logger.info(f"Writing {variable} to {group}")
        dims = group[variable].attrs["_ARRAY_DIMENSIONS"][1:]
        group[variable][index] = np.asarray(ds[variable].transpose(*dims))


if __name__ == "__main__":
    import fv3gfs
    from mpi4py import MPI

    RUN_DIR = os.path.dirname(os.path.realpath(__file__))

    DELP = "pressure_thickness_of_atmospheric_layer"
    TIME = "time"
    VARIABLES = list(runtime.CF_TO_RESTART_MAP) + [DELP, TIME]

    rank = MPI.COMM_WORLD.Get_rank()
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
    state = fv3gfs.get_state(names=VARIABLES)
    if rank == 0:
        logger.info("Beginning steps")
    for i in range(fv3gfs.get_step_count()):
        if rank == 0:
            logger.info(f"step {i}")
        begin_monitor.store(state)
        fv3gfs.step_dynamics()
        state = fv3gfs.get_state(names=VARIABLES)
        before_monitor.store(state)
        fv3gfs.step_physics()
        state = fv3gfs.get_state(names=VARIABLES)
        after_monitor.store(state)

    if rank == 0:
        post_process(RUN_DIR, **config["one_step"])
    fv3gfs.cleanup()
