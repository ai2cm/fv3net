import os
from typing import Sequence, Optional, MutableMapping
import functools
from datetime import datetime, timedelta
import yaml
import fsspec
import logging
import numpy as np

if __name__ == "__main__":
    import fv3gfs
    from mpi4py import MPI
else:
    fv3gfs = None
    MPI = None

logger = logging.getLogger(__name__)

RADIATION_NAMES = [
    "total_sky_downward_shortwave_flux_at_surface",
    "total_sky_upward_shortwave_flux_at_surface",
    "total_sky_downward_longwave_flux_at_surface",
    "total_sky_upward_longwave_flux_at_surface",
    "total_sky_downward_shortwave_flux_at_top_of_atmosphere",
    "total_sky_upward_shortwave_flux_at_top_of_atmosphere",
    "total_sky_upward_longwave_flux_at_top_of_atmosphere",
]

# probably also store C-grid winds from physics?
STORE_NAMES = [
    "x_wind",
    "y_wind",
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
] + RADIATION_NAMES

TENDENCY_OUT_FILENAME = "tendencies.zarr"
RUN_DIR = os.path.dirname(os.path.realpath(__file__))
GRAVITY = 9.81
M_PER_MM = 1 / 1000
PRECIP_NAME = "total_precipitation"


def get_timestep(config):
    return timedelta(seconds=config["namelist"]["coupler_nml"]["dt_atmos"])


def get_timescales_from_config(config):
    return_dict = {}
    for name, hours in config["nudging"]["timescale_hours"].items():
        return_dict[name] = timedelta(seconds=int(hours * 60 * 60))
    return return_dict


def time_to_label(time):
    return (
        f"{time.year:04d}{time.month:02d}{time.day:02d}."
        f"{time.hour:02d}{time.minute:02d}{time.second:02d}"
    )


def test_time_to_label():
    time, label = datetime(2015, 1, 20, 6, 30, 0), "20150120.063000"
    result = time_to_label(time)
    assert result == label


def get_restart_directory(reference_dir, label):
    return os.path.join(reference_dir, label)


def get_reference_state(time, reference_dir, communicator, only_names):
    label = time_to_label(time)
    dirname = get_restart_directory(reference_dir, label)
    logger.debug(f"Restart dir: {dirname}")
    state = fv3gfs.open_restart(
        dirname, communicator, label=label, only_names=only_names
    )
    state["time"] = time
    return state


def nudge_to_reference(state, reference, timescales, timestep):
    tendencies = fv3gfs.apply_nudging(state, reference, timescales, timestep)
    tendencies = append_key_label(tendencies, "_tendency_due_to_nudging")
    tendencies["time"] = state["time"]
    return tendencies


def append_key_label(d, suffix):
    return_dict = {}
    for key, value in d.items():
        return_dict[key + suffix] = value
    return return_dict


def column_integrated_moistening(humidity_tendency, pressure_thickness):
    GRAVITY = 9.81
    moistening = np.sum(humidity_tendency * pressure_thickness, axis=0) / gravity
    moistening.units = "kg/m^2/s"
    return moistening


def add_nudging_moistening_to_precipitation(state, tendencies, timestep):
    return_names = list(tendencies.keys())
    if "specific_humidity" in tendencies:
        return_names = return_names.append(PRECIP_NAME)
        moistening = column_integrated_moistening(
            tendencies["specific_humidity"],
            state["pressure_thickness_of_atmospheric_layer"],
        )
        total_precip = state[PRECIP_NAME] - M_PER_MM * timestep * moistening
        total_precip = np.where(total_precip >= 0, total_precip, 0)
        state[PRECIP_NAME] = total_precip
    return return_names


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
            monitor = fv3gfs.ZarrMonitor(
                store, self.partitioner, mode=self._mode, mpi_comm=MPI.COMM_WORLD
            )
            self._monitors[stage_name] = SubsetWriter(monitor, self.times)
        return self._monitors[stage_name]


class SubsetWriter:
    """Write only certain substeps"""

    def __init__(
        self, monitor: fv3gfs.ZarrMonitor, times: Optional[Sequence[str]] = None
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
            self.logger.debug(f"Storing time: {time}")
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
    reference_dir = config["nudging"]["restarts_path"]
    partitioner = fv3gfs.CubedSpherePartitioner.from_namelist(config["namelist"])
    communicator = fv3gfs.CubedSphereCommunicator(MPI.COMM_WORLD, partitioner)
    nudging_timescales = get_timescales_from_config(config)
    nudging_names = list(nudging_timescales.keys())
    store_names = list(set(["time"] + nudging_names + STORE_NAMES))
    timestep = get_timestep(config)

    nudge = functools.partial(
        nudge_to_reference, timescales=nudging_timescales, timestep=timestep,
    )

    monitor = StageWriter(
        RUN_DIR,
        partitioner,
        mode="w",
        times=config["nudging"].get("output_times", None),
    )

    fv3gfs.initialize()
    for i in range(fv3gfs.get_step_count()):
        state = fv3gfs.get_state(names=store_names)
        start = datetime.utcnow()
        time = state["time"]

        monitor.store(time, state, stage="before_dynamics")
        fv3gfs.step_dynamics()
        monitor.store(time, fv3gfs.get_state(names=store_names), stage="after_dynamics")
        fv3gfs.step_physics()
        state = fv3gfs.get_state(names=store_names)
        monitor.store(time, state, stage="after_physics")
        fv3gfs.save_intermediate_restart_if_enabled()
        reference = get_reference_state(
            time, reference_dir, communicator, only_names=store_names
        )
        tendencies = nudge(state, reference)
        monitor.store(time, reference, stage="reference")
        monitor.store(time, tendencies, stage="nudging_tendencies")
        monitor.store(time, state, stage="after_nudging")

        updated_names = add_nudging_moistening_to_precip(state, tendencies, timestep)
        updated_state_members = {
            key: quantity for key, quantity in state.items() if key in updated_names
        }
        fv3gfs.set_state(updated_state_members)
    fv3gfs.cleanup()
