import pandas as pd
import numpy as np
import xarray as xr
import vcm


from ..constants import TIME_NAME, TIME_FMT
from .._utils import standardize_zarr_time_coord


class GeoMapper:
    def __init__(self, *args):
        raise NotImplementedError("Don't use the base class!")

    def __len__(self):
        return len(self.keys())

    def __iter__(self):
        return iter(self.keys())

    def __getitem__(self, key: str) -> xr.Dataset:
        raise NotImplementedError()

    def keys(self):
        raise NotImplementedError()


class LongRunMapper(GeoMapper):
    """
    Basic mapper across the time dimension for any long-form
    simulation output.
    
    This mapper uses slightly different initialization from the
    base GeoMapper class (this takes a dataset instead of a url) because
    run information for all timesteps already exists within
    a single file, i.e., no filesystem grouping is necessary to get
    an item.
    """

    def __init__(self, ds):
        self.ds = standardize_zarr_time_coord(ds)

    def __getitem__(self, key: str) -> xr.Dataset:
        dt64 = np.datetime64(vcm.parse_datetime_from_str(key))
        return self.ds.sel({TIME_NAME: dt64})

    def keys(self):
        return [
            time.strftime(TIME_FMT)
            for time in pd.to_datetime(self.ds[TIME_NAME].values)
        ]
