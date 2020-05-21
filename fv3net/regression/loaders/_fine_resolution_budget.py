import os
import re
import vcm
import xarray as xr


class FineResolutionBudgetTiles:
    """An Mapping interface to a fine-res-q1-q2 dataset"""

    def __init__(self, url):
        super(FineResolutionBudget, self).__init__()
        fs = vcm.cloud.get_fs(url)
        self._url = url
        self.files = fs.glob(os.path.join(url, "*.nc"))

        if len(self.files) == 0:
            raise ValueError("No file detected")

    def _parse_file(self, url):
        pattern = r"tile(.)\.nc"
        match = re.search(pattern, url)
        date = vcm.parse_timestep_str_from_path(url)
        tile = match.group(1)
        return date, int(tile)

    def __getitem__(self, key: str) -> xr.Dataset:
        return xr.open_dataset(self._find_file(key))

    def _find_file(self, key):
        return [file for file in self.files if self._parse_file(file) == key][-1]

    def keys(self):
        return [self._parse_file(file) for file in self.files]

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.keys())


class FineResolutionBudget:
    # TODO this name is too specific. This operation will work for any Mapping keyed by
    # (time, tile) tuples

    def __init__(self, tiles: FineResolutionBudgetTiles) -> "FineResolutionBudget":
        self._tiles = tiles

    def __getitem__(self, time: str) -> xr.Dataset:
        tiles = list(range(1, 7))
        datasets = [self._tiles[(time, tile)] for tile in tiles]
        return xr.concat(datasets, dim="tile").assign_coords(tile=tiles)

    @classmethod
    def from_url(cls, url: str) -> "FineResolutionBudget":
        return cls(FineResolutionBudgetTiles(url))
