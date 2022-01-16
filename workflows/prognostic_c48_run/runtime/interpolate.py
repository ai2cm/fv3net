from .types import State
from .names import MASK
import xarray as xr
import cftime
from datetime import timedelta
import functools
from typing import Callable


# list of variables that will use nearest neighbor interpolation
# between times instead of linear interpolation
INTERPOLATE_NEAREST = [
    MASK,
]


def time_interpolate_func(
    func: Callable[[cftime.DatetimeJulian], State],
    frequency: timedelta,
    initial_time: cftime.DatetimeJulian,
) -> Callable[[cftime.DatetimeJulian], State]:
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


def label_to_time(time: str) -> cftime.DatetimeJulian:
    return cftime.DatetimeJulian(
        int(time[:4]),
        int(time[4:6]),
        int(time[6:8]),
        int(time[9:11]),
        int(time[11:13]),
        int(time[13:15]),
    )
