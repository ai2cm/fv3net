from .types import State
import xarray as xr
import cftime
from datetime import timedelta
import functools
from typing import Callable


def time_interpolate_func(
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
                out[key] = (
                    state_0[key] * weight + (1 - weight) * state_1[key]  # type: ignore
                )
    return out
