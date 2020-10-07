from ._base import GeoMapper
import xarray as xr
import collections
import vcm
import os
import fv3gfs.util._properties

RESTART_RENAMES = {
    data["restart_name"]: std_name
    for (std_name, data) in fv3gfs.util._properties.RESTART_PROPERTIES.items()
}


class CoarsenedDataMapper(GeoMapper, collections.abc.Mapping):
    def __init__(self, url):
        self._url = url
        self._fs = vcm.cloud.get_fs(url)
        self._keys = None
        self._n = 0

    def __getitem__(self, key: str) -> xr.Dataset:
        ds = vcm.open_restarts(os.path.join(self._url, key))
        ds = ds.squeeze("file_prefix")
        renames = {
            "grid_xt": fv3gfs.util.X_DIM,
            "grid_yt": fv3gfs.util.Y_DIM,
            "pfull": fv3gfs.util.Z_DIM,
            "sphum": "specific_humidity",
        }
        renames.update(
            {name: value for (name, value) in RESTART_RENAMES.items() if name in ds}
        )
        ds = ds.rename(renames)
        return ds

    def keys(self):
        if self._keys is None:
            self._keys = sorted(list(self._fs.ls(self._url)))
        return self._keys

    def __len__(self):
        return len(self.keys())

    def __iter__(self):
        self._n = 0
        return self

    def __next__(self):
        if self._n <= len(self):
            result = list(self.keys)[self._n]
            self._n += 1
            return result
        else:
            raise StopIteration
