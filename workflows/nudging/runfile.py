import os
import functools
from datetime import datetime, timedelta
import yaml

if __name__ == "__main__":
    import fv3gfs
    from mpi4py import MPI


TENDENCY_OUT_FILENAME = "tendencies.zarr"
RUN_DIR = os.path.dirname(os.path.realpath(__file__))


def get_timestep(config):
    return timedelta(seconds=config["namelist"]["coupler_nml"]["dt_atmos"])


def get_timescales_from_config(config):
    return_dict = {}
    for name, hours in config["nudging"]["timescale_hours"].items():
        return_dict[name] = timedelta(hours=hours)
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


def get_reference_state(time, reference_dir, partitioner, only_names):
    label = time_to_label(time)
    dirname = get_restart_directory(reference_dir, label)
    return fv3gfs.open_restart(dirname, partitioner, label=label, only_names=only_names)


def nudge_to_reference(state, partitioner, reference_dir, timescales, timestep):
    reference = get_reference_state(
        state["time"], reference_dir, partitioner, only_names=timescales.keys()
    )
    tendencies = fv3gfs.apply_nudging(state, reference, timescales, timestep)
    tendencies = append_key_label(tendencies, "_tendency_due_to_nudging")
    tendencies["time"] = state["time"]
    return tendencies


def append_key_label(d, suffix):
    return_dict = {}
    for key, value in d.items():
        return_dict[key + suffix] = value
    return return_dict


if __name__ == "__main__":
    rank = MPI.COMM_WORLD.Get_rank()
    current_dir = os.getcwd()
    with open("fv3config.yml", "r") as config_file:
        config = yaml.safe_load(config_file)
    reference_dir = config["nudging"]["restarts_path"]
    MPI.COMM_WORLD.barrier()  # wait for master rank to write run directory
    partitioner = fv3gfs.CubedSpherePartitioner.from_namelist(config["namelist"])
    nudging_timescales = get_timescales_from_config(config)
    timestep = get_timestep(config)

    nudge = functools.partial(
        nudge_to_reference,
        partitioner=partitioner,
        reference_dir=reference_dir,
        timescales=nudging_timescales,
        timestep=timestep,
    )

    tendency_monitor = fv3gfs.ZarrMonitor(
        os.path.join(RUN_DIR, TENDENCY_OUT_FILENAME),
        partitioner,
        mode="w",
        mpi_comm=MPI.COMM_WORLD,
    )

    fv3gfs.initialize()
    for i in range(fv3gfs.get_step_count()):
        fv3gfs.step_dynamics()
        fv3gfs.step_physics()
        fv3gfs.save_intermediate_restart_if_enabled()

        state = fv3gfs.get_state(names=["time"] + list(nudging_timescales.keys()))
        tendencies = nudge(state)
        tendency_monitor.store(tendencies)
        fv3gfs.set_state(state)
    fv3gfs.cleanup()
