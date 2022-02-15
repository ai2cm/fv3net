import pace.util
from mpi4py import MPI
import shutil
import fsspec
import xarray as xr
import dataclasses
import cftime
import functools
from datetime import timedelta
import os
from typing import (
    MutableMapping,
    Mapping,
    Iterable,
    Callable,
    Hashable,
    Any,
    Dict,
    Optional,
)
import logging
from runtime.names import STATE_NAME_TO_TENDENCY, MASK
from fv3kube import RestartCategoriesConfig

logger = logging.getLogger(__name__)

State = MutableMapping[Hashable, xr.DataArray]

# list of variables that will use nearest neighbor interpolation
# between times instead of linear interpolation
INTERPOLATE_NEAREST = [
    MASK,
]


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

    The reference state be interpolated between available time samples; by default
    linearly or using nearest neighbor intepolation for variables specified in
    INTERPOLATE_NEAREST.

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
    restart_categories: Optional[RestartCategoriesConfig] = None
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
    communicator: pace.util.CubedSphereCommunicator,
):
    """
    Configure the 'get_reference_function' for use in a nudged fv3gfs run.
    """
    reference_dir = config.restarts_path

    get_reference_state: Callable[[Any], Dict[Any, Any]] = functools.partial(
        _get_reference_state,
        reference_dir=reference_dir,
        restart_categories=config.restart_categories,
        communicator=communicator,
        only_names=state_names,
        tracer_metadata=tracer_metadata,
    )

    initial_time_label = config.reference_initial_time
    if initial_time_label is not None:
        get_reference_state = _time_interpolate_func(
            get_reference_state,
            initial_time=_label_to_time(initial_time_label),
            frequency=timedelta(seconds=config.reference_frequency_seconds),
        )

    return get_reference_state


def _get_reference_state(
    time: str,
    reference_dir: str,
    communicator: pace.util.CubedSphereCommunicator,
    only_names: Iterable[str],
    tracer_metadata: Mapping,
    restart_categories: Optional[RestartCategoriesConfig],
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

    state = pace.util.open_restart(
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
    localdir: str, restart_categories: RestartCategoriesConfig
) -> None:
    standard_restart_categories = RestartCategoriesConfig()
    files = os.listdir(localdir)
    for category_name in vars(restart_categories):
        disk_category = getattr(restart_categories, category_name)
        standard_category = getattr(standard_restart_categories, category_name)
        for file in [file for file in files if disk_category in file]:
            existing_filepath = os.path.join(localdir, file)
            standard_filepath = os.path.join(
                localdir, file.replace(disk_category, standard_category)
            )
            os.rename(existing_filepath, standard_filepath)


def _to_state_dataarrays(state: Mapping[str, pace.util.Quantity]) -> State:
    out: State = {}
    for var in state:
        out[var] = state[var].data_array
    return out


def _time_to_label(time: cftime.DatetimeJulian) -> str:
    return (
        f"{time.year:04d}{time.month:02d}{time.day:02d}."
        f"{time.hour:02d}{time.minute:02d}{time.second:02d}"
    )


def _label_to_time(time: str) -> cftime.DatetimeJulian:
    return cftime.DatetimeJulian(
        int(time[:4]),
        int(time[4:6]),
        int(time[6:8]),
        int(time[9:11]),
        int(time[11:13]),
        int(time[13:15]),
    )


def _time_interpolate_func(
    func: Callable[[cftime.DatetimeJulian], dict],
    frequency: timedelta,
    initial_time: cftime.DatetimeJulian,
) -> Callable[[cftime.DatetimeJulian], dict]:
    cached_func = functools.lru_cache(maxsize=2)(func)

    @functools.wraps(cached_func)
    def myfunc(time: cftime.DatetimeJulian) -> State:
        quotient = (time - initial_time) // frequency
        remainder = (time - initial_time) % frequency

        state: State = {}
        if remainder == timedelta(0):
            state.update(cached_func(time))
        else:
            begin_time = quotient * frequency + initial_time
            end_time = begin_time + frequency

            state_0 = cached_func(begin_time)
            state_1 = cached_func(end_time)

            state.update(
                _average_states(state_0, state_1, weight=(end_time - time) / frequency)
            )

        return state

    return myfunc


def _average_states(state_0: State, state_1: State, weight: float) -> State:
    common_keys = set(state_0) & set(state_1)
    out = {}
    for key in common_keys:
        if isinstance(state_1[key], xr.DataArray):
            with xr.set_options(keep_attrs=True):
                if key in INTERPOLATE_NEAREST:
                    out[key] = state_0[key] if weight >= 0.5 else state_1[key]
                else:
                    out[key] = (
                        state_0[key] * weight
                        + (1 - weight) * state_1[key]  # type: ignore
                    )
    return out


def get_nudging_tendency(
    state: State, reference_state: State, nudging_timescales: Mapping[str, timedelta]
) -> State:
    """
    Return the nudging tendency of the given state towards the reference state
    according to the provided nudging timescales. Adapted from pace.util.nudging
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
