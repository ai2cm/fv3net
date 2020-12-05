from typing import Callable, Union, Mapping

import xarray as xr

from ._base import GeoMapper

Time = str


class ValMap(GeoMapper):
    """Mapper which applies a function to the values of another

    Inspired by ``toolz.valmap``
    """

    def __init__(self, func: Callable[[xr.Dataset], xr.Dataset], mapper: GeoMapper):
        self.mapper = mapper
        self.func = func

    def __getitem__(self, key):
        return self.func(self.mapper[key])

    def keys(self):
        return self.mapper.keys()


class KeyMap(GeoMapper):
    """Mapper which applies a function to the keys of a GeoMapper

    Inspired by ``toolz.keymap``
    """

    def __init__(self, func: Callable[[xr.Dataset], xr.Dataset], mapper: GeoMapper):
        self.mapper = mapper
        self.func = func

    def __getitem__(self, key):
        return self.mapper[self._key_translation[key]]

    @property
    def _key_translation(self):
        return {self.func(key): key for key in self.mapper}

    def keys(self):
        return self._key_translation.keys()


class SubsetTimes(GeoMapper):
    """
    Sort and subset a timestep-based mapping to skip spin-up and limit
    the number of available times.
    """

    def __init__(
        self, i_start: int, n_times: Union[int, None], data: Mapping[str, xr.Dataset],
    ):
        timestep_keys = list(data.keys())
        timestep_keys.sort()

        i_end = None if n_times is None else i_start + n_times
        self._keys = timestep_keys[slice(i_start, i_end)]
        self._data = data

    def keys(self):
        return set(self._keys)

    def __getitem__(self, time: Time):
        if time not in self._keys:
            raise KeyError("Time {time} not found in SubsetTimes mapper.")
        return self._data[time]
