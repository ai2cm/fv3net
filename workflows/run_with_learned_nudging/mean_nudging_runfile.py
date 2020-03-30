import logging

import fv3gfs
import fv3util
from mpi4py import MPI
from fv3net import runtime
import fsspec
import xarray as xr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CF_TO_NUDGE = {
    "air_temperature": "t_dt_nudge",
    "specific_humidity": "q_dt_nudge",
    "pressure_thickness_of_atmospheric_layer": "delp_dt_nudge",
    "eastward_wind_after_physics": "u_dt_nudge",
    "northward_wind_after_physics": "v_dt_nudge",
}


def get_current_nudging_tendency(variables, time, ds_nudging):
    nudging_tendency = {}
    for variable in variables:
        nudging_variable_name = CF_TO_NUDGE[variable]
        nudging_tendency[variable] = ds_nudging[nudging_variable_name].sel(
            time=time, method="nearest"
        )
    return nudging_tendency


def apply_nudging_tendency(variables, ds_nudging, dt):
    state = fv3gfs.get_state(names=["time"] + variables)
    tendency_to_apply = get_current_nudging_tendency(
        variables, state["time"], ds_nudging
    )
    for variable in variables:
        state[variable].view[:] += tendency_to_apply[variable] * dt
    fv3gfs.set_state(state)


if __name__ == "__main__":
    rank = MPI.COMM_WORLD.Get_rank()
    config = runtime.get_config()
    nudging_zarr_url = config["runtime"]["nudging_zarr_url"]
    variables_to_nudge = config["runtime"]["variables_to_nudge"]
    dt = runtime.get_timestep()
    communicator = fv3gfs.CubedSphereCommunicator(
        MPI.COMM_WORLD, fv3gfs.CubedSpherePartitioner.from_namelist(config["namelist"])
    )
    tile = fv3util.get_tile_index(rank, communicator.get_total_ranks())
    mapper = fsspec.get_mapper(nudging_zarr_url)
    ds_nudging = xr.open_zarr(mapper).isel(tile=tile).load()

    if rank == 0:
        logger.info(f"Loaded nudging tendencies from {nudging_zarr_url}")
        logger.info(f"Nudging following variables: {variables_to_nudge}")

    fv3gfs.initialize()
    for i in range(fv3gfs.get_step_count()):
        if rank == 0:
            logger.info(f"Stepping dynamics for timestep {i}")
        fv3gfs.step_dynamics()
        if rank == 0:
            logger.info(f"Computing physics routines for timestep {i}")
        fv3gfs.compute_physics()
        if rank == 0:
            logger.info(f"Adding nudging tendency for timestep {i}")
        apply_nudging_tendency(variables_to_nudge, ds_nudging, dt)
        if rank == 0:
            logger.info(f"Update atmospheric prognostic state for timestep {i}")
        fv3gfs.apply_physics()
