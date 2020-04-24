import os
import functools
from datetime import datetime, timedelta
import yaml
import fsspec

if __name__ == "__main__":
    import fv3gfs
    from mpi4py import MPI
else:
    fv3gfs = None
    MPI = None


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
]


def get_radiation_names():
    radiation_names = []
    for properties in fv3gfs.PHYSICS_PROPERTIES:
        if properties["container"] == "Radtend":
            radiation_names.append(properties["name"])
    assert len(radiation_names) > 0
    return radiation_names


if fv3gfs is not None:
    STORE_NAMES.extend(get_radiation_names())

TENDENCY_OUT_FILENAME = "tendencies.zarr"
RUN_DIR = os.path.dirname(os.path.realpath(__file__))


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


class StageMonitor:
    def __init__(self, root_dirname, partitioner, mode="w"):
        self._root_dirname = root_dirname
        self._monitors = {}
        self._mode = mode
        self.partitioner = partitioner

    def store(self, state, stage):
        monitor = self._get_monitor(stage)
        monitor.store(state)

    def _get_monitor(self, stage_name):
        if stage_name not in self._monitors:
            store = master_only(
                lambda: fsspec.get_mapper(
                    os.path.join(self._root_dirname, stage_name + ".zarr")
                )
            )()
            self._monitors[stage_name] = fv3gfs.ZarrMonitor(
                store, self.partitioner, mode=self._mode, mpi_comm=MPI.COMM_WORLD
            )
        return self._monitors[stage_name]


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

    monitor = StageMonitor(RUN_DIR, partitioner, mode="w",)

    fv3gfs.initialize()
    for i in range(fv3gfs.get_step_count()):
        state = fv3gfs.get_state(names=store_names)
        start = datetime.utcnow()
        monitor.store(state, stage="before_dynamics")
        fv3gfs.step_dynamics()
        monitor.store(fv3gfs.get_state(names=store_names), stage="after_dynamics")
        fv3gfs.step_physics()
        state = fv3gfs.get_state(names=store_names)
        monitor.store(state, stage="after_physics")
        fv3gfs.save_intermediate_restart_if_enabled()
        reference = get_reference_state(
            state["time"], reference_dir, communicator, only_names=store_names
        )
        tendencies = nudge(state, reference)
        monitor.store(reference, stage="reference")
        monitor.store(tendencies, stage="nudging_tendencies")
        monitor.store(state, stage="after_nudging")
        fv3gfs.set_state(state)
    fv3gfs.cleanup()
