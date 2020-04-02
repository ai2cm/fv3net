import logging

import fv3gfs
import fv3util
from mpi4py import MPI
from fv3net import runtime
import fsspec
import xarray as xr
import cftime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CF_TO_NUDGE = {
    "air_temperature": "t_dt_nudge",
    "specific_humidity": "q_dt_nudge",
    "pressure_thickness_of_atmospheric_layer": "delp_dt_nudge",
    "eastward_wind_after_physics": "u_dt_nudge",
    "northward_wind_after_physics": "v_dt_nudge",
}


def _ensure_Julian(date):
    return cftime.DatetimeJulian(
        date.year, date.month, date.day, date.hour, date.minute, date.second
    )


def get_current_nudging_tendency(variables, time, ds_nudging):
    nudging_tendency = {}
    for variable in variables:
        nudging_variable_name = CF_TO_NUDGE[variable]
        nudging_tendency[variable] = ds_nudging[nudging_variable_name].sel(
            time=_ensure_Julian(time), method="nearest"
        )
    return nudging_tendency


def apply_nudging_tendency(variables, ds_nudging, dt, communicator):
    state = fv3gfs.get_state(names=["time"] + variables)
    tile = communicator.partitioner.tile_index(communicator.rank)
    if communicator.tile.rank == 0:
        logger.info(f"Getting nudging tendency for current timestep on tile {tile}")
        tendency = get_current_nudging_tendency(variables, state["time"], ds_nudging)
        logger.info(f"Converting nudging tendency to fv3gfs state on tile {tile}")
        tendency_as_state = {
            variable: fv3util.Quantity.from_data_array(tendency[variable])
            for variable in variables
        }
    if communicator.tile.rank == 0:
        logger.info(f"Scattering nudging tendency to all ranks for tile {tile}")
    tendency_as_state = communicator.tile.scatter_state(tendency_as_state)
    if communicator.tile.rank == 0:
        logger.info(f"Adding nudging tendency to state for tile {tile}")
    for variable in variables:
        state[variable].view[:] += tendency_as_state[variable].view[:] * dt
    if communicator.tile.rank == 0:
        logger.info(f"Updating fv3gfs state for tile {tile}")
    fv3gfs.set_state(state)


if __name__ == "__main__":
    config = runtime.get_config()
    nudging_zarr_url = config["runtime"]["nudging_zarr_url"]
    mapper = fsspec.get_mapper(nudging_zarr_url)
    variables_to_nudge = config["runtime"]["variables_to_nudge"]
    dt = runtime.get_timestep()
    communicator = fv3gfs.CubedSphereCommunicator(
        MPI.COMM_WORLD, fv3gfs.CubedSpherePartitioner.from_namelist(config["namelist"])
    )
    # tile = fv3util.get_tile_index(rank, communicator.partitioner.total_ranks)
    rank = communicator.rank
    tile_index = communicator.partitioner.tile_index(rank)
    if rank == 0:
        logger.info(f"Nudging following variables: {variables_to_nudge}")
    if communicator.tile.rank == 0:
        logger.info(f"My rank is {rank} and my tile_rank is {communicator.tile.rank}")
        logger.info(f"Loading tile-{tile_index} nudging tendencies from {nudging_zarr_url}")
        ds_nudging = xr.open_zarr(mapper).isel(tile=tile_index).load()

    fv3gfs.initialize()
    for i in range(fv3gfs.get_step_count()):
        do_logging = rank == 0 and i % 10 == 0
        if do_logging:
            logger.info(f"Stepping dynamics for timestep {i}")
        fv3gfs.step_dynamics()
        if do_logging:
            logger.info(f"Computing physics routines for timestep {i}")
        fv3gfs.compute_physics()
        if do_logging:
            logger.info(f"Adding nudging tendency for timestep {i}")
        if communicator.tile.rank == 0:
            apply_nudging_tendency(variables_to_nudge, ds_nudging, dt, communicator)
        if do_logging:
            logger.info(f"Update atmospheric prognostic state for timestep {i}")
        fv3gfs.apply_physics()
    fv3gfs.cleanup()
