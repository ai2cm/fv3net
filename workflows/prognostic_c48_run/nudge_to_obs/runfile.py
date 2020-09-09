import os
from typing import Sequence, Optional, MutableMapping
import functools
from datetime import datetime, timedelta
import yaml
import fsspec
import logging
import fv3gfs.util

if __name__ == "__main__":
    import fv3gfs.wrapper as wrapper
    from mpi4py import MPI
else:
    wrapper = None
    MPI = None

STORE_NAMES = [
    "x_wind",
    "y_wind",
    "eastward_wind",
    "northward_wind",
    "vertical_wind",
    "air_temperature",
    "specific_humidity",
    "time",
    "pressure_thickness_of_atmospheric_layer",
    "vertical_thickness_of_atmospheric_layer",
    "land_sea_mask",
    "surface_temperature",
    "surface_geopotential",
    "sensible_heat_flux",
    "latent_heat_flux",
    "total_precipitation",
    "surface_precipitation_rate",
    "total_sky_downward_shortwave_flux_at_surface",
    "total_sky_upward_shortwave_flux_at_surface",
    "total_sky_downward_longwave_flux_at_surface",
    "total_sky_upward_longwave_flux_at_surface",
    "total_sky_downward_shortwave_flux_at_top_of_atmosphere",
    "total_sky_upward_shortwave_flux_at_top_of_atmosphere",
    "total_sky_upward_longwave_flux_at_top_of_atmosphere",
]

RUN_DIR = os.path.dirname(os.path.realpath(__file__))


def get_timestep(config):
    return timedelta(seconds=config["namelist"]["coupler_nml"]["dt_atmos"])


class StageWriter:
    def __init__(
        self,
        root_dirname: str,
        partitioner,
        mode="w",
        times: Optional[Sequence[str]] = None,
    ):
        self._root_dirname = root_dirname
        self._monitors: MutableMapping[str, SubsetWriter] = {}
        self._mode = mode
        self.partitioner = partitioner
        self.times = times

    def store(self, time: datetime, state, stage: str):
        monitor = self._get_monitor(stage)
        monitor.store(time, state)

    def _get_monitor(self, stage_name: str):
        if stage_name not in self._monitors:
            store = master_only(
                lambda: fsspec.get_mapper(
                    os.path.join(self._root_dirname, stage_name + ".zarr")
                )
            )()
            monitor = wrapper.ZarrMonitor(
                store, self.partitioner, mode=self._mode, mpi_comm=MPI.COMM_WORLD
            )
            self._monitors[stage_name] = SubsetWriter(monitor, self.times)
        return self._monitors[stage_name]


class SubsetWriter:
    """Write only certain substeps"""

    def __init__(
        self, monitor: wrapper.ZarrMonitor, times: Optional[Sequence[str]] = None
    ):
        """

        Args:
            monitor: a stage monitor to use to store outputs
            times: an optional list of output times to store stages at. By
                default all times will be output.
        """
        self._monitor = monitor
        self._times = times
        self.time = None
        self.logger = logging.getLogger("SubsetStageWriter")
        self.logger.info(f"Saving stages at {self._times}")

    def _output_current_time(self, time: datetime) -> bool:
        if self._times is None:
            return True
        else:
            return time.strftime("%Y%m%d.%H%M%S") in self._times

    def store(self, time: datetime, state):
        if self._output_current_time(time):
            self._monitor.store(state)


def master_only(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        if MPI.COMM_WORLD.Get_rank() == 0:
            result = func(*args, **kwargs)
        else:
            result = None
        return MPI.COMM_WORLD.bcast(result)

    return wrapped


@master_only
def load_config(filename):
    with open("fv3config.yml", "r") as config_file:
        config = yaml.safe_load(config_file)
    return config


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = load_config("fv3config.yml")
    partitioner = wrapper.CubedSpherePartitioner.from_namelist(config["namelist"])
    communicator = wrapper.CubedSphereCommunicator(MPI.COMM_WORLD, partitioner)
    timestep = get_timestep(config)

    monitor = StageWriter(
        RUN_DIR,
        partitioner,
        mode="w",
        times=config["runfile_output"].get("output_times", None),
    )

    wrapper.initialize()
    for i in range(wrapper.get_step_count()):
        state = wrapper.get_state(names=STORE_NAMES)
        state["time"] += timestep  # consistent with Fortran diagnostic output times
        time = state["time"]
        monitor.store(time, state, stage="before_dynamics")
        wrapper.step_dynamics()
        monitor.store(
            time, wrapper.get_state(names=STORE_NAMES), stage="after_dynamics"
        )
        wrapper.step_physics()
        monitor.store(time, wrapper.get_state(names=STORE_NAMES), stage="after_physics")

    wrapper.cleanup()
