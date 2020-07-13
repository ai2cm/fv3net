from typing import Callable

import xarray as xr

from ._base import GeoMapper


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
