import fv3gfs.util
from mpi4py import MPI
import shutil
import fsspec
import dataclasses
import cftime
import functools
from datetime import timedelta
import os
from typing import (
    Mapping,
    Iterable,
    Callable,
    Any,
    Dict,
    Optional,
)
import logging
from .interpolate import time_interpolate_func, label_to_time
from .names import STATE_NAME_TO_TENDENCY
from .types import State

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class NudgingConfig:
    """Nudging Configurations

    The prognostic run supports nudge-to-fine towards a dataset with a different
    sampling frequency than the model time step. The available nudging times
    should start with ``reference_initial_time`` and appear at a regular
    frequency of ``reference_frequency_seconds`` thereafter. These options are
    optional; if not provided the nudging data will be assumed to contain every
    time. The reference state will be linearly interpolated between the
    available time samples.

    Attributes:
        timescale_hours: mapping of variable names to timescales (in hours).

    Examples::

        NudgingConfig(
            timescale_hours={
                "air_temperature": 3,
                "specific_humidity": 3,
                "x_wind": 3,
                "y_wind": 3,
            },
            restarts_path="gs://some-bucket/some-path",
            reference_initial_time="20160801.001500",
            reference_frequency_seconds=900,
        )
    """

    restarts_path: str
    timescale_hours: Dict[str, float]
    restart_categories: Optional[Mapping[str, str]] = None
    # This should be required, but needs to be typed this way to preserve
    # backwards compatibility with existing yamls that don't specify it.
    # See https://github.com/ai2cm/fv3net/issues/1449
    # optional arguments needed for time interpolation
    reference_initial_time: Optional[str] = None
    reference_frequency_seconds: float = 900


def nudging_timescales_from_dict(timescales: Mapping) -> Mapping:
    return_dict = {}
    for name, hours in timescales.items():
        return_dict[name] = timedelta(seconds=int(hours * 60 * 60))
    return return_dict


def setup_get_reference_state(
    config: NudgingConfig,
    state_names: Iterable[str],
    tracer_metadata: Mapping,
    communicator: fv3gfs.util.CubedSphereCommunicator,
):
    """
    Configure the 'get_reference_function' for use in a nudged fv3gfs run.
    """
    reference_dir = config.restarts_path

    get_reference_state: Callable[[Any], State] = functools.partial(
        _get_reference_state,
        reference_dir=reference_dir,
        restart_categories=config.restart_categories,
        communicator=communicator,
        only_names=state_names,
        tracer_metadata=tracer_metadata,
    )

    initial_time_label = config.reference_initial_time
    if initial_time_label is not None:
        get_reference_state = time_interpolate_func(
            get_reference_state,
            initial_time=label_to_time(initial_time_label),
            frequency=timedelta(seconds=config.reference_frequency_seconds),
        )

    return get_reference_state


def _get_reference_state(
    time: cftime.DatetimeJulian,
    reference_dir: str,
    communicator: fv3gfs.util.CubedSphereCommunicator,
    only_names: Iterable[str],
    tracer_metadata: Mapping,
    restart_categories: Optional[Mapping[str, str]] = None,
):
    label = _time_to_label(time)
    dirname = os.path.join(reference_dir, label)

    localdir = "download"

    if MPI.COMM_WORLD.rank == 0:
        fs = fsspec.get_fs_token_paths(dirname)[0]
        fs.get(dirname, localdir, recursive=True)
        if restart_categories is not None:
            _rename_local_restarts(localdir, restart_categories)

    # need this for synchronization
    MPI.COMM_WORLD.barrier()

    state = fv3gfs.util.open_restart(
        localdir,
        communicator,
        label=label,
        only_names=only_names,
        tracer_properties=tracer_metadata,
    )

    # clean up the local directory
    # wait for other processes to finish using the data
    MPI.COMM_WORLD.barrier()
    if MPI.COMM_WORLD.rank == 0:
        shutil.rmtree(localdir)

    return _to_state_dataarrays(state)


def _rename_local_restarts(
    localdir: str, restart_categories: Mapping[str, str]
) -> None:
    files = os.listdir(localdir)
    for required_category, disk_category in restart_categories.items():
        for file in [file for file in files if disk_category in file]:
            existing_filepath = os.path.join(localdir, file)
            required_filepath = os.path.join(
                localdir, file.replace(disk_category, required_category)
            )
            os.rename(existing_filepath, required_filepath)


def _to_state_dataarrays(state: Mapping[str, fv3gfs.util.Quantity]) -> State:
    out: State = {}
    for var in state:
        out[var] = state[var].data_array
    return out


def _time_to_label(time: cftime.DatetimeJulian) -> str:
    return (
        f"{time.year:04d}{time.month:02d}{time.day:02d}."
        f"{time.hour:02d}{time.minute:02d}{time.second:02d}"
    )


def get_nudging_tendency(
    state: State, reference_state: State, nudging_timescales: Mapping[str, timedelta]
) -> State:
    """
    Return the nudging tendency of the given state towards the reference state
    according to the provided nudging timescales. Adapted from fv3gfs-util.nudging
    version, but for use with derived state mappings.
    Args:
        state (dict): A derived state dictionary.
        reference_state (dict): A reference state dictionary.
        nudging_timescales (dict): A dictionary whose keys are standard names and
            values are timedelta objects indicating the relaxation timescale for that
            variable.
    Returns:
        nudging_tendencies (dict): A dictionary whose keys are the tendency names
            associated with the state variables in names.STATE_NAME_TO_TENDENCY
            and values are xr.DataArray objects indicating the nudging tendency
            of that standard name.
    """
    return_dict: State = {}
    for name, timescale in nudging_timescales.items():
        var_state = state[name]
        var_reference = reference_state[name]
        return_data = (var_reference - var_state) / timescale.total_seconds()
        return_data = return_data.assign_attrs(
            {"units": f'{var_state.attrs.get("units", "")} s^-1'}
        )
        return_dict[STATE_NAME_TO_TENDENCY[name]] = return_data
    return return_dict
