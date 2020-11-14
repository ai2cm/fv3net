import fv3gfs.util
import xarray as xr
import cftime
import functools
from datetime import timedelta
import os
from typing import MutableMapping, Mapping, Iterable, Callable, Hashable
import logging

logger = logging.getLogger(__name__)

SST_NAME = "ocean_surface_temperature"
TSFC_NAME = "surface_temperature"
MASK_NAME = "land_sea_mask"

State = MutableMapping[Hashable, xr.DataArray]


def nudging_timescales_from_dict(timescales: Mapping) -> Mapping:
    return_dict = {}
    for name, hours in timescales.items():
        return_dict[name] = timedelta(seconds=int(hours * 60 * 60))
    return return_dict


def setup_get_reference_state(config: Mapping, state_names: Iterable[str], comm):
    
    partitioner = fv3gfs.util.CubedSpherePartitioner.from_namelist(config["namelist"])
    communicator = fv3gfs.util.CubedSphereCommunicator(comm, partitioner)
    reference_dir = config['nudging']['restarts_path']
    
    get_reference_state = functools.partial(
        _get_reference_state,
        reference_dir=reference_dir,
        communicator=communicator,
        only_names=state_names,
    )
    
    initial_time_label = config["nudging"].get("reference_initial_time")
    if initial_time_label is not None:
        interpolated_get_reference_state = _time_interpolate_func(
            _get_reference_state,
            initial_time=_label_to_time(initial_time_label),
            frequency=timedelta(
                seconds=config["nudging"]["reference_frequency_seconds"]
            ),
        )
        
    return interpolated_get_reference_state


def _get_reference_state(time: str, reference_dir: str, communicator: fv3gfs.util.CubedSphereCommunicator, only_names: Iterable[str]):
    label = _time_to_label(time)
    dirname = os.path.join(reference_dir, label)
    logger.debug(f"Restart dir: {dirname}")
    state = fv3gfs.util.open_restart(
        dirname, communicator, label=label, only_names=only_names
    )
    return _to_state_dataarrays(state)


def _to_state_dataarrays(state: Mapping[str, fv3gfs.util.Quantity]) -> State:
    out = {}
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

            state.update(_average_states(
                state_0, state_1, weight=(end_time - time) / frequency
            ))

        state["time"] = time
        return state

    return myfunc


def _average_states(state_0: State, state_1: State, weight: float) -> State:
    common_keys = set(state_0) & set(state_1)
    out = {}
    for key in common_keys:
        if isinstance(state_1[key], xr.DataArray):
            out[key] = state_0[key] * weight + (1 - weight) * state_1[key] # type: ignore
    return out


def nudging_tendency(state: State, reference: State, timescales: Mapping) -> State:
    tendency = fv3gfs.util.get_nudging_tendencies(state, reference, timescales)
    tendency = _append_key_label(tendency, "_tendency_due_to_nudging")
    return tendency
            
            
def _append_key_label(d, suffix):
    return_dict = {}
    for key, value in d.items():
        return_dict[key + suffix] = value
    return return_dict


def set_state_sst_to_reference(state, reference):
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