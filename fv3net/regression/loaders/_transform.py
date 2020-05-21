from typing import Mapping, Tuple
import xarray as xr
from toolz import groupby

Time = str
Tile = int
K = Tuple[Time, Tile]


class GroupByTime:
    def __init__(self, tiles: Mapping[K, xr.Dataset]) -> Mapping[K, xr.Dataset]:
        def fn(key):
            _, tile = key
            return tile

        self._tiles = tiles
        self._time_lookup = groupby(fn, self._tiles.keys())

    def keys(self):
        return self._time_lookup.keys()

    def __len__(self):
        return len(self.keys())

    def __getitem__(self, time: Time) -> xr.Dataset:
        # TODO generalize this function
        tiles = list(range(1, 7))
        datasets = [self._tiles[key] for key in self._time_lookup[time]]
        return xr.concat(datasets, dim="tile").assign_coords(tile=tiles)
