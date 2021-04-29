import fv3gfs.util
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

logger = logging.getLogger(__name__)

SST_NAME = "ocean_surface_temperature"
TSFC_NAME = "surface_temperature"
MASK_NAME = "land_sea_mask"

State = MutableMapping[Hashable, xr.DataArray]


@dataclasses.dataclass
class NudgingConfig:
    """Nudging Configurations

    The runfile supports nudge-to-fine towards a dataset with a different sampling
    frequency than the model time step. The available nudging times should start
    with ``reference_initial_time`` and appear at a regular frequency of
    ``reference_frequency_seconds`` thereafter. These options are optional; if
    not provided the nudging data will be assumed to contain every time. The
    reference state will be linearly interpolated between the available time
    samples.

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

    timescale_hours: Dict[str, float]
    restarts_path: str
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

    get_reference_state: Callable[[Any], Dict[Any, Any]] = functools.partial(
        _get_reference_state,
        reference_dir=reference_dir,
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
    communicator: fv3gfs.util.CubedSphereCommunicator,
    only_names: Iterable[str],
    tracer_metadata: Mapping,
):
    label = _time_to_label(time)
    dirname = os.path.join(reference_dir, label)

    localdir = "download"

    if MPI.COMM_WORLD.rank == 0:
        fs = fsspec.get_fs_token_paths(dirname)[0]
        fs.get(dirname, localdir, recursive=True)

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
            out[key] = (
                state_0[key] * weight + (1 - weight) * state_1[key]  # type: ignore
            )
    return out


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
        nudging_tendencies (dict): A dictionary whose keys are standard names
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
        return_dict[name] = return_data
    return return_dict


def set_state_sst_to_reference(state: State, reference: State) -> State:
    """
    Set the sea surface and surface temperatures in a model state to values in
    a reference state. Useful for maintaining consistency between a nudged run
    and reference state.
    """
    state[SST_NAME] = _sst_from_reference(
        reference[TSFC_NAME], state[SST_NAME], state[MASK_NAME]
    )
    state[TSFC_NAME] = _sst_from_reference(
        reference[TSFC_NAME], state[TSFC_NAME], state[MASK_NAME]
    )
    return state


def _sst_from_reference(
    reference_surface_temperature: xr.DataArray,
    surface_temperature: xr.DataArray,
    land_sea_mask: xr.DataArray,
) -> xr.DataArray:
    return xr.where(
        land_sea_mask.values.round().astype("int") == 0,
        reference_surface_temperature,
        surface_temperature,
    )
